import numpy as np
from scipy.interpolate import PchipInterpolator
from typing import List, Tuple, Optional, Any

class FusedMeshSkewTransform:
    def __init__(self, config: Any) -> None:
        """
        Initializes the analytical extension instance. Registers distinct, namespaced
        G-code entry points to preserve core Klipper runtime hygiene.
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
        self.mesh_minimum_coordinates: Optional[List[float]] = None
        self.mesh_maximum_coordinates: Optional[List[float]] = None

        if self.skew_method == 'traditional':
            self.traditional_skew_mode: str = config.get('traditional_skew_mode').lower()
            if self.traditional_skew_mode not in ['x', 'y', 'xy']:
                raise config.error("traditional_mode must be explicitly set to 'x', 'y', or 'xy'.")

        self.probe_x_offset: Optional[float] = None
        self.probe_y_offset: Optional[float] = None
        self.calibration_center_x: Optional[float] = None
        self.calibration_center_y: Optional[float] = None
        
        self.printer.register_event_handler("klippy:connect", self.handle_connect)
        self.gcode.register_command('COMPUTE_SPATIAL_GEOMETRY_TRANSFORM', self.cmd_COMPUTE_SPATIAL_GEOMETRY_TRANSFORM,
                                    desc="Isolate gantry variations and finalize non-linear step filters")

    def handle_connect(self) -> None:
        """
        Runs post-initialization to inherit structural configurations and map
        toolhead offset vectors from the active machine state.
        """
        self.beacon_hardware_probe = self.printer.lookup_object('beacon', None)
        if self.beacon_hardware_probe is None:
            raise self.printer.config_error("Hardware Initialization Error: No active Beacon device found.")
            
        self.probe_x_offset = float(self.beacon_hardware_probe.x_offset)
        self.probe_y_offset = float(self.beacon_hardware_probe.y_offset)

        try:
            pconfig = self.printer.lookup_object('configfile')
            bed_mesh_config = pconfig.config.getsection('bed_mesh')
        except Exception:
            raise self.printer.config_error("Configuration Error: [bed_mesh] section must be defined.")
            
        self.mesh_minimum_coordinates = bed_mesh_config.getfloatlist('mesh_min', count=2)
        self.mesh_maximum_coordinates = bed_mesh_config.getfloatlist('mesh_max', count=2)

        safe_z_home_object = self.printer.lookup_object('safe_z_home', None)
        if safe_z_home_object is not None:
            self.calibration_center_x = float(safe_z_home_object.home_x_pos)
            self.calibration_center_y = float(safe_z_home_object.home_y_pos)
        else:
            self.calibration_center_x = (self.mesh_minimum_coordinates[0] + self.mesh_maximum_coordinates[0]) / 2.0
            self.calibration_center_y = (self.mesh_minimum_coordinates[1] + self.mesh_maximum_coordinates[1]) / 2.0

    def _thin_plate_spline_kernel(self, radius: np.ndarray) -> np.ndarray:
        return np.where(radius > 0, (radius**2) * np.log(radius), 0.0)

    def _squared_exponential_kernel(self, coordinates_set_1: np.ndarray, coordinates_set_2: np.ndarray, 
                                    process_variance: float, spatial_lengthscale: float) -> np.ndarray:
        squared_distances = np.sum((coordinates_set_1[:, None, :] - coordinates_set_2[None, :, :])**2, axis=-1)
        return (process_variance**2) * np.exp(-squared_distances / (2.0 * (spatial_lengthscale**2)))

    def _solve_thin_plate_spline_system(self, control_nodes: np.ndarray, target_displacements: np.ndarray) -> np.ndarray:
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
        control_node_count = len(control_nodes)
        warping_weights = coefficients[:control_node_count]
        affine_coefficients = coefficients[control_node_count:]
        
        linear_trend = float(affine_coefficients[0] + affine_coefficients[1] * target_coordinate[0] + affine_coefficients[2] * target_coordinate[1])
        radial_distances = np.linalg.norm(control_nodes - target_coordinate, axis=1)
        warping_deformation = float(np.sum(warping_weights * self._thin_plate_spline_kernel(radial_distances)))
        
        return linear_trend + warping_deformation

    def transform_xyz(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
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

    def cmd_COMPUTE_SPATIAL_GEOMETRY_TRANSFORM(self, gcode_command: Any) -> None:
        """
        Processes active low-fidelity tracking vectors, generates a programmatic
        touch mapping array, solves the stochastic parameters, and updates memory references.
        """
        self.is_mesh_calibrated = False
        
        bed_mesh_module = self.printer.lookup_object('bed_mesh', None)
        if bed_mesh_module is None:
            gcode_command.respond_error("Mesh Error: Active bed_mesh object reference tracking failed.")
            return

        z_mesh_wrapper = bed_mesh_module.z_mesh
        grid_shape_x, grid_shape_y = z_mesh_wrapper.probed_matrix.shape

        beacon_status_dictionary = self.beacon_hardware_probe.get_status()
        self.high_fidelity_sensor_noise_standard_deviation = float(beacon_status_dictionary.get('contact_precision_deviation', 0.0012))
        self.low_fidelity_sensor_noise_standard_deviation = float(beacon_status_dictionary.get('proximity_latency_noise', 0.0045))

        # 1. Extract the raw low-fidelity coordinate and height fields populated by the macro sweep
        raw_coordinates_low_fidelity = np.array(z_mesh_wrapper.get_coords())[:, :2]
        raw_heights_low_fidelity = np.array(z_mesh_wrapper.get_mesh()).flatten()
        transformed_coordinates_low_fidelity = raw_coordinates_low_fidelity + np.array([self.probe_x_offset, self.probe_y_offset])

        # 2. Programmatically execute a sparse 7x7 nozzle contact loop to map true references
        self.gcode.respond_info("Orchestrating discrete reference touches across a sparse 7x7 matrix...")
        x_ticks = np.linspace(self.mesh_minimum_coordinates[0], self.mesh_maximum_coordinates[0], 7)
        y_ticks = np.linspace(self.mesh_minimum_coordinates[1], self.mesh_maximum_coordinates[1], 7)
        
        seeded_coordinates = []
        seeded_heights = []
        toolhead_object = self.printer.lookup_object('toolhead')

        for y_coord in y_ticks:
            for x_coord in x_ticks:
                # Compensate for carriage offset vectors to ensure accurate physical tool placement
                toolhead_object.manual_move([x_coord - self.probe_x_offset, y_coord - self.probe_y_offset, 5.0], 50.0)
                self.printer.lookup_object('toolhead').wait_moves()
                
                absolute_height_strike = float(self.beacon_hardware_probe.run_probe(gcode_command))
                seeded_coordinates.append([x_coord, y_coord])
                seeded_heights.append(absolute_height_strike)

        transformed_coordinates_high_fidelity = np.array(seeded_coordinates)
        raw_heights_high_fidelity = np.array(seeded_heights)

        # 3. Handle localized data clusters to protect matrix stability
        high_fidelity_coordinate_distances = np.linalg.norm(transformed_coordinates_high_fidelity[:, None, :] - transformed_coordinates_high_fidelity[None, :, :], axis=-1)
        micro_spatial_pairs_mask = (high_fidelity_coordinate_distances > 0.0) & (high_fidelity_coordinate_distances <= 15.0)
        
        if not np.any(micro_spatial_pairs_mask):
            self.gcode.respond_info("Injecting autonomous 5-point micro-cross around safe Z home coordinates...")
            cross_target_nodes = np.array([
                [self.calibration_center_x, self.calibration_center_y],
                [self.calibration_center_x + 10.0, self.calibration_center_y],
                [self.calibration_center_x - 10.0, self.calibration_center_y],
                [self.calibration_center_x, self.calibration_center_y + 10.0],
                [self.calibration_center_x, self.calibration_center_y - 10.0]
            ])
            for target_node in cross_target_nodes:
                toolhead_object.manual_move([target_node[0] - self.probe_x_offset, target_node[1] - self.probe_y_offset, 5.0], 50.0)
                self.printer.lookup_object('toolhead').wait_moves()
                
                absolute_height_strike = float(self.beacon_hardware_probe.run_probe(gcode_command))
                transformed_coordinates_high_fidelity = np.vstack((transformed_coordinates_high_fidelity, target_node))
                raw_heights_high_fidelity = np.concatenate((raw_heights_high_fidelity, np.array([absolute_height_strike])))
            
            high_fidelity_coordinate_distances = np.linalg.norm(transformed_coordinates_high_fidelity[:, None, :] - transformed_coordinates_high_fidelity[None, :, :], axis=-1)
            micro_spatial_pairs_mask = (high_fidelity_coordinate_distances > 0.0) & (high_fidelity_coordinate_distances <= 15.0)

        # 4. Perform shape-preserving PCHIP track alignment
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

        # 5. Evaluate the Gaussian Process Co-Kriging covariance field
        self.process_variance_low_fidelity = float(np.std(raw_heights_low_fidelity))
        spatial_height_differences = raw_heights_low_fidelity[:, None] - raw_heights_low_fidelity[None, :]
        spatial_coordinate_distances = np.linalg.norm(transformed_coordinates_low_fidelity[:, None, :] - transformed_coordinates_low_fidelity[None, :, :], axis=-1)
        
        valid_spatial_pairs_mask = (spatial_coordinate_distances > 0) & (spatial_coordinate_distances < 50.0)
        mean_squared_difference = float(np.mean(spatial_height_differences[valid_spatial_pairs_mask]**2))
        mean_spatial_correlation = (self.process_variance_low_fidelity**2 - 0.0005 * mean_squared_difference) / (self.process_variance_low_fidelity**2 + 1e-12)
        mean_spatial_correlation = np.clip(mean_spatial_correlation, 0.01, 0.99)
        self.spatial_lengthscale_low_fidelity = float(np.mean(spatial_coordinate_distances[valid_spatial_pairs_mask]) / np.sqrt(-2.0 * np.log(mean_spatial_correlation)))

        self.scaling_coefficient_rho = float(np.dot(aligned_heights_low_fidelity, raw_heights_high_fidelity) / np.dot(aligned_heights_low_fidelity, aligned_heights_low_fidelity))
        
        discrepancy_vector_at_high_coordinates = raw_heights_high_fidelity - (self.scaling_coefficient_rho * aligned_heights_low_fidelity)
        self.process_variance_discrepancy = float(np.std(discrepancy_vector_at_high_coordinates))
        
        discrepancy_coordinate_diffs = discrepancy_vector_at_high_coordinates[:, None] - discrepancy_vector_at_high_coordinates[None, :]
        mean_discrepancy_squared_difference = float(np.mean(discrepancy_coordinate_diffs[micro_spatial_pairs_mask]**2))
        discrepancy_spatial_correlation = (self.process_variance_discrepancy**2 - 0.0005 * mean_discrepancy_squared_difference) / (self.process_variance_discrepancy**2 + 1e-12)
        discrepancy_spatial_correlation = np.clip(discrepancy_spatial_correlation, 0.01, 0.99)
        self.spatial_lengthscale_discrepancy = float(np.mean(high_fidelity_coordinate_distances[micro_spatial_pairs_mask]) / np.sqrt(-2.0 * np.log(discrepancy_spatial_correlation)))

        auto_cov_delta = self._squared_exponential_kernel(transformed_coordinates_high_fidelity, transformed_coordinates_high_fidelity, self.process_variance_discrepancy, self.spatial_lengthscale_discrepancy) + (self.high_fidelity_sensor_noise_standard_deviation**2) * np.eye(len(transformed_coordinates_high_fidelity))
        cross_cov_dense_delta = self._squared_exponential_kernel(transformed_coordinates_low_fidelity, transformed_coordinates_high_fidelity, self.process_variance_discrepancy, self.spatial_lengthscale_discrepancy)
        
        dense_predicted_discrepancy_field = np.dot(cross_cov_dense_delta, np.linalg.solve(auto_cov_delta, discrepancy_vector_at_high_coordinates))

        # 6. Mutate the active probed matrix reference addresses in place
        dense_calibrated_heights = (self.scaling_coefficient_rho * raw_heights_low_fidelity) + dense_predicted_discrepancy_field
        z_mesh_wrapper.probed_matrix = dense_calibrated_heights.reshape(grid_shape_x, grid_shape_y)
        
        # Force Klipper to compute updated internal interpolation splines using the new grid values
        bed_mesh_module.save_profile("default")
        self.gcode.respond_info("Dual-fidelity surface mapping completed. Global tracking parameters synchronized.")

        # 7. Isolate and solve Thin Plate Spline coefficients for non-linear lateral transformations
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

def load_config(config: Any) -> FusedMeshSkewTransform:
    return FusedMeshSkewTransform(config)