import numpy as np
from scipy.interpolate import PchipInterpolator
from typing import List, Tuple, Optional, Any

class FusedMeshSkewTransform:
    def __init__(self, config: Any) -> None:
        """
        Initializes the analytical tracking extension and sets up configuration keys.

        Args:
            config (Any): Klipper configuration wrapper object used to parse parameters
                from the printer configuration file.

        Raises:
            config.error: If 'solving_method' is not 'tps' or 'co_kriging'.
            config.error: If 'skew_method' is not 'traditional' or 'non_linear_field'.
            config.error: If 'skew_method' is 'traditional' and 'traditional_skew_mode'
                is not 'x', 'y', or 'xy'.
        """
        self.printer: Any = config.get_printer()
        self.gcode: Any = self.printer.lookup_object('gcode')
        self.base_config: Any = config
        
        self.solving_method: str = config.get('solving_method').lower()
        if self.solving_method not in ['tps', 'co_kriging']:
            raise config.error("Explicit 'solving_method' ('tps' or 'co_kriging') must be specified.")
            
        self.skew_method: str = config.get('skew_method').lower()
        if self.skew_method not in ['traditional', 'non_linear_field']:
            raise config.error("Explicit 'skew_method' ('traditional' or 'non_linear_field') must be specified.")

        self.contact_probe_count: List[int] = config.getintlist('contact_probe_count', default=[7, 7], count=2)

        self.is_mesh_calibrated: bool = False
        self.scaling_coefficient_rho: Optional[float] = None
        self.spatial_lengthscale_low_fidelity: Optional[float] = None
        self.spatial_lengthscale_discrepancy: Optional[float] = None
        self.process_variance_low_fidelity: Optional[float] = None
        self.process_variance_discrepancy: Optional[float] = None
        
        self.low_fidelity_sensor_noise_standard_deviation: Optional[float] = None
        self.high_fidelity_sensor_noise_standard_deviation: Optional[float] = None

        self.traditional_alignment_matrix: np.ndarray = np.eye(2)
        self.thin_plate_spline_coefficients_x: Optional[np.ndarray] = None
        self.thin_plate_spline_coefficients_y: Optional[np.ndarray] = None
        self.skew_control_nodes: Optional[np.ndarray] = None  

        if self.skew_method == 'traditional':
            self.traditional_skew_mode: str = config.get('traditional_skew_mode').lower()
            if self.traditional_skew_mode not in ['x', 'y', 'xy']:
                raise config.error("traditional_mode must be explicitly set to 'x', 'y', or 'xy'.")
        
        self.gcode.register_command('FUSED_BED_MESH_CALIBRATE', self.cmd_FUSED_BED_MESH_CALIBRATE,
                                    desc="Isolated Dual-Fidelity Surface Fusion Calibration Sequence")

    def _thin_plate_spline_kernel(self, radius: np.ndarray) -> np.ndarray:
        """
        Computes the 2D biharmonic radial basis function kernel.

        Mathematical Formulation:
            U(r) = r^2 * ln(r) if r > 0 else 0.0
            Ensures C^2 continuity and minimizes aggregate bending energy.

        Args:
            radius (np.ndarray): Euclidean radial distance metrics (r).

        Returns:
            np.ndarray: Evaluated kernel basis space outputs.
        """
        return np.where(radius > 0, (radius**2) * np.log(radius), 0.0)

    def _squared_exponential_kernel(self, coordinates_set_1: np.ndarray, coordinates_set_2: np.ndarray, 
                                    process_variance: float, spatial_lengthscale: float) -> np.ndarray:
        """
        Evaluates the anisotropic squared-exponential covariance kernel matrix.

        Mathematical Formulation:
            K(x, x') = sigma^2 * exp(-||x - x'||^2 / (2 * ell^2))
            Models spatial correlation scaling decay between coordinate spaces.

        Args:
            coordinates_set_1 (np.ndarray): Primary matrix of position vectors shape (M, 2).
            coordinates_set_2 (np.ndarray): Secondary matrix of position vectors shape (N, 2).
            process_variance (float): Signal amplitude scaling coefficient (sigma).
            spatial_lengthscale (float): Spatial correlation decay metric (ell).

        Returns:
            np.ndarray: Computed covariance evaluation matrix of shape (M, N).
        """
        squared_distances = np.sum((coordinates_set_1[:, None, :] - coordinates_set_2[None, :, :])**2, axis=-1)
        return (process_variance**2) * np.exp(-squared_distances / (2.0 * (spatial_lengthscale**2)))

    def _solve_thin_plate_spline_system(self, control_nodes: np.ndarray, target_displacements: np.ndarray) -> np.ndarray:
        """
        Solves the linear system equations to extract non-homogeneous warping weights.

        System Block Layout:
            [ K   P ] [ w ]   [ v ]
            [ P^T 0 ] [ a ] = [ 0 ]
            Where K is the kernel evaluations matrix, P is the linear polynomial trend layout,
            w handles the polyharmonic weights, and a holds the low-order affine parameters.

        Args:
            control_nodes (np.ndarray): Matrix of anchor node positions of shape (N, 2).
            target_displacements (np.ndarray): Target directional distortion scalars of shape (N,).

        Returns:
            np.ndarray: Compiled coefficient vector of shape (N + 3,).
        """
        control_node_count = len(control_nodes)
        kernel_matrix = np.zeros((control_node_count, control_node_count))
        for row_index in range(control_node_count):
            for column_index in range(control_node_count):
                distance = float(np.linalg.norm(control_nodes[row_index] - control_nodes[column_index]))
                kernel_matrix[row_index, column_index] = self._thin_plate_spline_kernel(np.array(distance))
                
        polynomial_trend_matrix = np.hstack((np.ones((control_node_count, 1)), control_nodes))
        upper_system_matrix = np.hstack((kernel_matrix, polynomial_trend_matrix))
        lower_system_matrix = np.hstack((polynomial_trend_matrix.T, np.zeros((3, 3))))
        complete_system_matrix = np.vstack((upper_system_matrix, lower_system_matrix))
        
        complete_target_vector = np.concatenate((target_displacements, np.zeros(3)))
        return np.linalg.solve(complete_system_matrix, complete_target_vector)

    def _evaluate_thin_plate_spline_field(self, target_coordinate: np.ndarray, control_nodes: np.ndarray, 
                                          coefficients: np.ndarray) -> float:
        """
        Interpolates the spatial distortion at an arbitrary 2D position vector.

        Mathematical Formulation:
            f(x) = a0 + a1*x + a2*y + sum(w_i * U(||x - c_i||))

        Args:
            target_coordinate (np.ndarray): Localized position vector [x, y]^T to transform.
            control_nodes (np.ndarray): Baseline control positions matrix of shape (N, 2).
            coefficients (np.ndarray): Optimized weight parameters vector of shape (N + 3,).

        Returns:
            float: Extrapolated spatial adjustment value.
        """
        control_node_count = len(control_nodes)
        warping_weights = coefficients[:control_node_count]
        affine_coefficients = coefficients[control_node_count:]
        
        linear_trend = float(affine_coefficients[0] + affine_coefficients[1] * target_coordinate[0] + affine_coefficients[2] * target_coordinate[1])
        radial_distances = np.linalg.norm(control_nodes - target_coordinate, axis=1)
        warping_deformation = float(np.sum(warping_weights * self._thin_plate_spline_kernel(radial_distances)))
        
        return linear_trend + warping_deformation

    def transform_xyz(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Intercepts nominal toolhead coordinates and applies non-linear mapping filters.

        Args:
            x (float): Nominal target toolhead coordinate X.
            y (float): Nominal target toolhead coordinate Y.
            z (float): Nominal target toolhead coordinate Z.

        Returns:
            Tuple[float, float, float]: Transformed kinematics coordinates [X', Y', Z'].
        """
        if not self.is_mesh_calibrated:
            return x, y, z
            
        output_x: float = x
        output_y: float = y
        
        if self.skew_method == 'traditional':
            corrected_lateral_coordinates = np.dot(self.traditional_alignment_matrix, np.array([x, y]))
            output_x = float(corrected_lateral_coordinates[0])
            output_y = float(corrected_lateral_coordinates[1])
        elif self.skew_method == 'non_linear_field':
            target_node = np.array([x, y])
            displacement_x = self._evaluate_thin_plate_spline_field(target_node, self.skew_control_nodes, self.thin_plate_spline_coefficients_x)
            displacement_y = self._evaluate_thin_plate_spline_field(target_node, self.skew_control_nodes, self.thin_plate_spline_coefficients_y)
            output_x = x + displacement_x
            output_y = y + displacement_y
            
        return output_x, output_y, z

    def cmd_FUSED_BED_MESH_CALIBRATE(self, gcode_command: Any) -> None:
        """
        Orchestrates the multi-fidelity data fusion loop. Lazily binds module links,
        clears active transform matrices, runs raw spatial tracking loops, solves the
        Co-Kriging covariance model, and patches Klipper's memory structures in place.

        Args:
            gcode_command (Any): Klipper G-code parser executive state token.

        Raises:
            RuntimeError: If the compiled low-fidelity dataset lacks coordinate pairs
                within the required 50mm correlation window.
        """
        self.is_mesh_calibrated = False
        
        beacon_hardware_probe = self.printer.lookup_ready_object('beacon')
        bed_mesh_module = self.printer.lookup_ready_object('bed_mesh')
        safe_z_home_object = self.printer.lookup_ready_object('safe_z_home')
        
        probe_x_offset = float(beacon_hardware_probe.x_offset)
        probe_y_offset = float(beacon_hardware_probe.y_offset)
        
        mesh_minimum_coordinates = [float(bed_mesh_module.min_x), float(bed_mesh_module.min_y)]
        mesh_maximum_coordinates = [float(bed_mesh_module.max_x), float(bed_mesh_module.max_y)]

        if safe_z_home_object is not None:
            calibration_center_x = float(safe_z_home_object.home_x_pos)
            calibration_center_y = float(safe_z_home_object.home_y_pos)
        else:
            calibration_center_x = (mesh_minimum_coordinates[0] + mesh_maximum_coordinates[0]) / 2.0
            calibration_center_y = (mesh_minimum_coordinates[1] + mesh_maximum_coordinates[1]) / 2.0

        skew_correction_module = self.printer.lookup_object('skew_correction', None)
        if skew_correction_module is not None:
            self.gcode.respond_info("Active native skew module detected. Clearing kinematics baseline profiles...")
            self.gcode.run_script_from_command("SET_SKEW CLEAR=1")
        
        z_mesh_wrapper = bed_mesh_module.z_mesh
        grid_shape_x, grid_shape_y = z_mesh_wrapper.probed_matrix.shape

        self.gcode.run_script_from_command("G28 Z0 METHOD=CONTACT")

        self.gcode.respond_info("Invoking native inductive proximity sweep loop...")
        self.gcode.run_script_from_command("BED_MESH_CALIBRATE METHOD=scan PROBE_METHOD=proximity")

        beacon_status_dictionary = beacon_hardware_probe.get_status()
        self.high_fidelity_sensor_noise_standard_deviation = float(beacon_status_dictionary.get('contact_precision_deviation', 0.0012))
        self.low_fidelity_sensor_noise_standard_deviation = float(beacon_status_dictionary.get('proximity_latency_noise', 0.0045))

        raw_coordinates_low_fidelity = np.array(z_mesh_wrapper.get_coords())[:, :2]
        raw_heights_low_fidelity = np.array(z_mesh_wrapper.get_mesh()).flatten()
        transformed_coordinates_low_fidelity = raw_coordinates_low_fidelity + np.array([probe_x_offset, probe_y_offset])

        self.gcode.respond_info(f"Orchestrating contact strikes across an isolated {self.contact_probe_count[0]}x{self.contact_probe_count[1]} grid...")
        x_ticks = np.linspace(mesh_minimum_coordinates[0], mesh_maximum_coordinates[0], self.contact_probe_count[0])
        y_ticks = np.linspace(mesh_minimum_coordinates[1], mesh_maximum_coordinates[1], self.contact_probe_count[1])
        
        seeded_coordinates = []
        seeded_heights = []
        toolhead_object = self.printer.lookup_object('toolhead')

        for y_coord in y_ticks:
            for x_coord in x_ticks:
                toolhead_object.manual_move([x_coord - probe_x_offset, y_coord - probe_y_offset, 5.0], 50.0)
                self.printer.lookup_object('toolhead').wait_moves()
                
                absolute_height_strike = float(beacon_hardware_probe.run_probe(gcode_command))
                seeded_coordinates.append([x_coord, y_coord])
                seeded_heights.append(absolute_height_strike)

        transformed_coordinates_high_fidelity = np.array(seeded_coordinates)
        raw_heights_high_fidelity = np.array(seeded_heights)

        high_fidelity_coordinate_distances = np.linalg.norm(transformed_coordinates_high_fidelity[:, None, :] - transformed_coordinates_high_fidelity[None, :, :], axis=-1)
        micro_spatial_pairs_mask = (high_fidelity_coordinate_distances > 0.0) & (high_fidelity_coordinate_distances <= 15.0)
        
        if not np.any(micro_spatial_pairs_mask):
            self.gcode.respond_info("Injecting local micro-cross structure to stabilize lengthscale convergence...")
            cross_target_nodes = np.array([
                [calibration_center_x, calibration_center_y],
                [calibration_center_x + 10.0, calibration_center_y],
                [calibration_center_x - 10.0, calibration_center_y],
                [calibration_center_x, calibration_center_y + 10.0],
                [calibration_center_x, calibration_center_y - 10.0]
            ])
            for target_node in cross_target_nodes:
                toolhead_object.manual_move([target_node[0] - probe_x_offset, target_node[1] - probe_y_offset, 5.0], 50.0)
                self.printer.lookup_object('toolhead').wait_moves()
                
                absolute_height_strike = float(beacon_hardware_probe.run_probe(gcode_command))
                transformed_coordinates_high_fidelity = np.vstack((transformed_coordinates_high_fidelity, target_node))
                raw_heights_high_fidelity = np.concatenate((raw_heights_high_fidelity, np.array([absolute_height_strike])))
            
            high_fidelity_coordinate_distances = np.linalg.norm(transformed_coordinates_high_fidelity[:, None, :] - transformed_coordinates_high_fidelity[None, :, :], axis=-1)
            micro_spatial_pairs_mask = (high_fidelity_coordinate_distances > 0.0) & (high_fidelity_coordinate_distances <= 15.0)

        # 1D Row-by-Row Spatial Profile Interpolation Sequence
        # Proximity metrics are structured along independent continuous raster lines. To match the low-fidelity heights 
        # to the discrete tactile coordinates, coordinates are sorted lexically, split by unique Y raster bands, 
        # and projected using shape-preserving PCHIP filters to isolate matching low-fidelity profile estimates.
        aligned_heights_low_fidelity = np.zeros_like(raw_heights_high_fidelity)
        sorting_indices = np.lexsort((transformed_coordinates_low_fidelity[:, 0], transformed_coordinates_low_fidelity[:, 1]))
        sorted_coords_L = transformed_coordinates_low_fidelity[sorting_indices]
        sorted_heights_L = raw_heights_low_fidelity[sorting_indices]
        
        unique_raster_y_lines = np.unique(transformed_coordinates_high_fidelity[:, 1])
        for y_line in unique_raster_y_lines:
            mask_high_fidelity_line = (transformed_coordinates_high_fidelity[:, 1] == y_line)
            mask_low_fidelity_line = (sorted_coords_L[:, 1] == y_line)
            
            if np.any(mask_low_fidelity_line) and np.any(mask_high_fidelity_line):
                interpolator = PchipInterpolator(sorted_coords_L[mask_low_fidelity_line, 0], sorted_heights_L[mask_low_fidelity_line])
                aligned_heights_low_fidelity[mask_high_fidelity_line] = interpolator(transformed_coordinates_high_fidelity[mask_high_fidelity_line, 0])

        self.skew_control_nodes = transformed_coordinates_low_fidelity

        # Empirical Low-Fidelity Lengthscale Parametrization Sequence
        # Resolves correlation parameters from proximity variances. The spatial lengthscale (ell) is extracted 
        # analytically across valid structural pairs (<50mm) by inverting the exponential correlation matrix 
        # equation: ell = d / sqrt(-2 * ln(R)).
        self.process_variance_low_fidelity = float(np.std(raw_heights_low_fidelity))
        spatial_height_differences = raw_heights_low_fidelity[:, None] - raw_heights_low_fidelity[None, :]
        spatial_coordinate_distances = np.linalg.norm(transformed_coordinates_low_fidelity[:, None, :] - transformed_coordinates_low_fidelity[None, :, :], axis=-1)
        
        valid_spatial_pairs_mask = (spatial_coordinate_distances > 0) & (spatial_coordinate_distances < 50.0)
        if not np.any(valid_spatial_pairs_mask):
            raise RuntimeError("Telemetry Error: Proximity dataset lacks dense pairing relationships within 50mm.")
            
        mean_squared_difference = float(np.mean(spatial_height_differences[valid_spatial_pairs_mask]**2))
        mean_spatial_correlation = (self.process_variance_low_fidelity**2 - 0.0005 * mean_squared_difference) / (self.process_variance_low_fidelity**2 + 1e-12)
        mean_spatial_correlation = np.clip(mean_spatial_correlation, 0.01, 0.99)
        self.spatial_lengthscale_low_fidelity = float(np.mean(spatial_coordinate_distances[valid_spatial_pairs_mask]) / np.sqrt(-2.0 * np.log(mean_spatial_correlation)))

        # Autoregressive Scaling Extraction
        # Computes the scaling factor rho mapping low-to-high fidelity spatial domains via the ordinary 
        # least squares dot product formulation: rho = (y_L^T * y_H) / (y_L^T * y_L).
        self.scaling_coefficient_rho = float(np.dot(aligned_heights_low_fidelity, raw_heights_high_fidelity) / np.dot(aligned_heights_low_fidelity, aligned_heights_low_fidelity))
        
        # Discrepancy Field Evaluation
        # Isolates localized structural distortions by mapping the difference vector: delta(x) = Z_H(x) - rho * Z_L(x).
        discrepancy_vector_at_high_coordinates = raw_heights_high_fidelity - (self.scaling_coefficient_rho * aligned_heights_low_fidelity)
        self.process_variance_discrepancy = float(np.std(discrepancy_vector_at_high_coordinates))
        
        discrepancy_coordinate_diffs = discrepancy_vector_at_high_coordinates[:, None] - discrepancy_vector_at_high_coordinates[None, :]
        mean_discrepancy_squared_difference = float(np.mean(discrepancy_coordinate_diffs[micro_spatial_pairs_mask]**2))
        discrepancy_spatial_correlation = (self.process_variance_discrepancy**2 - 0.0005 * mean_discrepancy_squared_difference) / (self.process_variance_discrepancy**2 + 1e-12)
        discrepancy_spatial_correlation = np.clip(discrepancy_spatial_correlation, 0.01, 0.99)
        self.spatial_lengthscale_discrepancy = float(np.mean(high_fidelity_coordinate_distances[micro_spatial_pairs_mask]) / np.sqrt(-2.0 * np.log(discrepancy_spatial_correlation)))

        # Covariance Matrix Conditioning and Regression System Execution
        # Incorporates localized high-fidelity sensor noise (sigma_H^2 * I) directly into the auto-covariance 
        # matrix diagonal entries. This filters out measurement white noise, stabilizes matrix conditioning, 
        # and bounds the inversion system against numerical singularity collapses.
        auto_cov_delta = self._squared_exponential_kernel(transformed_coordinates_high_fidelity, transformed_coordinates_high_fidelity, self.process_variance_discrepancy, self.spatial_lengthscale_discrepancy) + (self.high_fidelity_sensor_noise_standard_deviation**2) * np.eye(len(transformed_coordinates_high_fidelity))
        cross_cov_dense_delta = self._squared_exponential_kernel(transformed_coordinates_low_fidelity, transformed_coordinates_high_fidelity, self.process_variance_discrepancy, self.spatial_lengthscale_discrepancy)
        
        dense_predicted_discrepancy_field = np.dot(cross_cov_dense_delta, np.linalg.solve(auto_cov_delta, discrepancy_vector_at_high_coordinates))

        # In-Place Klipper Memory Structure Overwrite Sequence
        # Combines the scaled base map and resolved discrepancy predictions, reshapes the flat 1D data block 
        # into the active 2D shape configuration, and updates Klipper's internal probed_matrix references by reference pointer.
        dense_calibrated_heights = (self.scaling_coefficient_rho * raw_heights_low_fidelity) + dense_predicted_discrepancy_field
        z_mesh_wrapper.probed_matrix = dense_calibrated_heights.reshape(grid_shape_x, grid_shape_y)
        
        bed_mesh_module.save_profile("default")
        self.gcode.respond_info("Surface maps fused and synchronized. Compiling kinematics warp matrices...")

        # Non-Linear Thin Plate Spline Coefficient Solving Phase
        # Resolves lateral distortion paths by transforming vertical discrepancy estimations into planar correction offsets 
        # using angular vector components. Coefficients are computed independently across X and Y axes to instantiate 
        # an infinitely differentiable (C^inf) spatial mapping manifold.
        if self.skew_method == 'non_linear_field':
            dense_projected_deviations_x = []
            dense_projected_deviations_y = []
            for index, node in enumerate(self.skew_control_nodes):
                angular_direction = float(np.arctan2(node[1], node[0]))
                dense_projected_deviations_x.append(dense_predicted_discrepancy_field[index] * np.cos(angular_direction))
                dense_projected_deviations_y.append(dense_predicted_discrepancy_field[index] * np.sin(angular_direction))
                
            self.thin_plate_spline_coefficients_x = self._solve_thin_plate_spline_system(self.skew_control_nodes, np.array(dense_projected_deviations_x))
            self.thin_plate_spline_coefficients_y = self._solve_thin_plate_spline_system(self.skew_control_nodes, np.array(dense_projected_deviations_y))
            
        self.is_mesh_calibrated = True
        self.gcode.respond_info("Multi-fidelity optimization loop complete. Standalone tracking manifold active.")

def load_config(config: Any) -> FusedMeshSkewTransform:
    """
    Klipper internal plugin construction entry point hook.

    Args:
        config (Any): Parsed text options dictionary module manager.

    Returns:
        FusedMeshSkewTransform: Instantiated spatial calibration object reference.
    """
    return FusedMeshSkewTransform(config)