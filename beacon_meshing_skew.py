import numpy as np
from scipy.interpolate import griddata
from typing import List, Tuple, Optional, Any

class FusedMeshSkewTransform:
    def __init__(self, config: Any) -> None:
        """Initializes the analytical tracking extension."""
        self.printer: Any = config.get_printer()
        self.gcode: Any = self.printer.lookup_object('gcode')
        self.base_config: Any = config
        
        self.solving_method: str = config.get('solving_method', 'co_kriging').lower()
        self.skew_method: str = config.get('skew_method', 'non_linear_field').lower()

        self.is_mesh_calibrated: bool = False
        
        # Buffer for the Contact Anchor Data
        self.contact_coordinates: Optional[np.ndarray] = None
        self.contact_heights: Optional[np.ndarray] = None

        # Transformation Data
        self.traditional_alignment_matrix: np.ndarray = np.eye(2)
        self.thin_plate_spline_coefficients_x: Optional[np.ndarray] = None
        self.thin_plate_spline_coefficients_y: Optional[np.ndarray] = None
        self.skew_control_nodes: Optional[np.ndarray] = None  
        
        # Register the two synchronous math execution endpoints
        self.gcode.register_command('_SAVE_HIGH_FIDELITY_MESH', self.cmd_SAVE_HIGH_FIDELITY_MESH,
                                    desc="Buffers contact anchors before inductive pass")
        self.gcode.register_command('_COMPUTE_KINEMATIC_MANIFOLD', self.cmd_COMPUTE_KINEMATIC_MANIFOLD,
                                    desc="Executes Co-Kriging Fusion and TPS Space Warping")

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
        """Intercepts nominal toolhead coordinates and applies non-linear mapping filters."""
        if not self.is_mesh_calibrated or self.skew_method == 'none':
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

    def cmd_SAVE_HIGH_FIDELITY_MESH(self, gcode_command: Any) -> None:
        """Captures the output of the native Beacon contact probing pass into module memory."""
        bed_mesh_module = self.printer.lookup_ready_object('bed_mesh')
        z_mesh_wrapper = bed_mesh_module.z_mesh
        
        # Save raw coordinates and heights directly from Klipper's native generator
        self.contact_coordinates = np.array(z_mesh_wrapper.get_coords())[:, :2]
        self.contact_heights = np.array(z_mesh_wrapper.get_mesh()).flatten()
        
        self.gcode.respond_info("High-fidelity contact anchor mesh saved to fusion buffer.")

    def cmd_COMPUTE_KINEMATIC_MANIFOLD(self, gcode_command: Any) -> None:
        """Executes pure mathematical fusion using the saved contact anchors and the active proximity scan."""
        if self.contact_coordinates is None or self.contact_heights is None:
            raise gcode_command.error("Fusion error: No contact mesh data in buffer. Orchestration Macro failed.")
            
        self.is_mesh_calibrated = False
        beacon_hardware_probe = self.printer.lookup_ready_object('beacon')
        bed_mesh_module = self.printer.lookup_ready_object('bed_mesh')
        z_mesh_wrapper = bed_mesh_module.z_mesh
        
        # Load the newly generated dense inductive scan topology
        scan_coordinates = np.array(z_mesh_wrapper.get_coords())[:, :2]
        scan_heights = np.array(z_mesh_wrapper.get_mesh()).flatten()
        grid_shape_x, grid_shape_y = z_mesh_wrapper.probed_matrix.shape

        beacon_status_dictionary = beacon_hardware_probe.get_status()
        std_dev_hf = float(beacon_status_dictionary.get('contact_precision_deviation', 0.0012))
        std_dev_lf = float(beacon_status_dictionary.get('proximity_latency_noise', 0.0045))

        self.skew_control_nodes = scan_coordinates

        if self.solving_method == 'co_kriging':
            # Mathematically robust 2D topological interpolation (Scipy Cubic Fallback to Nearest)
            aligned_heights_lf = griddata(scan_coordinates, scan_heights, self.contact_coordinates, method='cubic')
            nans = np.isnan(aligned_heights_lf)
            if np.any(nans):
                aligned_heights_lf[nans] = griddata(scan_coordinates, scan_heights, self.contact_coordinates[nans], method='nearest')
                
            # Discrepancy Matrix Solvers
            rho = float(np.dot(aligned_heights_lf, self.contact_heights) / np.dot(aligned_heights_lf, aligned_heights_lf))
            discrepancy_vector = self.contact_heights - (rho * aligned_heights_lf)
            variance_discrepancy = float(np.std(discrepancy_vector))
            
            hf_coordinate_distances = np.linalg.norm(self.contact_coordinates[:, None, :] - self.contact_coordinates[None, :, :], axis=-1)
            mean_hf_distance = np.mean(hf_coordinate_distances[hf_coordinate_distances > 0])
            lengthscale_discrepancy = mean_hf_distance / np.sqrt(-2.0 * np.log(0.5))  # Normalized mid-band correlation
            
            # Covariance System Inversion
            auto_cov = self._squared_exponential_kernel(self.contact_coordinates, self.contact_coordinates, variance_discrepancy, lengthscale_discrepancy) + (std_dev_hf**2) * np.eye(len(self.contact_coordinates))
            cross_cov = self._squared_exponential_kernel(scan_coordinates, self.contact_coordinates, variance_discrepancy, lengthscale_discrepancy)
            
            dense_predicted_discrepancy = np.dot(cross_cov, np.linalg.solve(auto_cov, discrepancy_vector))
            dense_calibrated_heights = (rho * scan_heights) + dense_predicted_discrepancy

        elif self.solving_method == 'tps':
            dense_calibrated_heights = np.zeros(len(scan_coordinates))
            tps_coefficients_z = self._solve_thin_plate_spline_system(self.contact_coordinates, self.contact_heights)
            for index, node in enumerate(scan_coordinates):
                dense_calibrated_heights[index] = self._evaluate_thin_plate_spline_field(node, self.contact_coordinates, tps_coefficients_z)
            
            dense_predicted_discrepancy = dense_calibrated_heights - scan_heights

        # Overwrite active Klipper bed mesh topology in memory
        z_mesh_wrapper.probed_matrix = dense_calibrated_heights.reshape(grid_shape_x, grid_shape_y)
        bed_mesh_module.save_profile("default")

        if self.skew_method == 'none':
            self.gcode.respond_info("Vertical topology fused. Skew method is 'none', manifold skipped.")
            return

        # Solve lateral Non-Linear TPS Field using discrepancy vectors
        if self.skew_method == 'non_linear_field':
            dense_projected_dev_x = []
            dense_projected_dev_y = []
            for index, node in enumerate(self.skew_control_nodes):
                angular_direction = float(np.arctan2(node[1], node[0]))
                dense_projected_dev_x.append(dense_predicted_discrepancy[index] * np.cos(angular_direction))
                dense_projected_dev_y.append(dense_predicted_discrepancy[index] * np.sin(angular_direction))
                
            self.thin_plate_spline_coefficients_x = self._solve_thin_plate_spline_system(self.skew_control_nodes, np.array(dense_projected_dev_x))
            self.thin_plate_spline_coefficients_y = self._solve_thin_plate_spline_system(self.skew_control_nodes, np.array(dense_projected_dev_y))
            
            self.is_mesh_calibrated = True
            self.gcode.respond_info("Kinematic manifold compiled. TPS skew active.")

def load_config(config: Any) -> FusedMeshSkewTransform:
    return FusedMeshSkewTransform(config)