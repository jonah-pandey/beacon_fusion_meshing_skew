import numpy as np
from scipy.interpolate import PchipInterpolator
from typing import List, Tuple, Optional, Any

class FusedMeshSkewTransform:
    def __init__(self, config: Any) -> None:
        """
        Initializes the transform instance and caches the initialization ConfigWrapper.
        Postpones cross-section option reads to handle_connect to honor Klipper's
        internal sub-module initialization order.
        """
        self.printer: Any = config.get_printer()
        self.gcode: Any = self.printer.lookup_object('gcode')
        
        # Cache the initial ConfigWrapper safely to serve as our root configuration hook
        self.base_config: Any = config
        
        self.solving_method: str = config.get('solving_method').lower()
        if self.solving_method not in ['tps', 'co_kriging']:
            raise config.error("Explicit 'solving_method' ('tps' or 'co_kriging') must be specified.")
            
        self.skew_method: str = config.get('skew_method').lower()
        if self.skew_method not in ['traditional', 'non_linear_field']:
            raise config.error("Explicit 'skew_method' ('traditional' or 'non_linear_field') must be specified.")

        self.dense_probe_count: List[int] = config.getintlist('dense_probe_count', count=2)

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
        self.gcode.register_command('CALIBRATE_SPATIAL_GEOMETRY', self.cmd_CALIBRATE_SPATIAL_GEOMETRY,
                                    desc="Sync topography tracks and calculate dual-fidelity variables")

    def handle_connect(self) -> None:
        """
        Runs on the Klippy connect hook after all system objects are constructed.
        Safely extracts configurations across sections to establish a clean data baseline.
        """
        self.beacon_hardware_probe = self.printer.lookup_object('beacon', None)
        if self.beacon_hardware_probe is None:
            raise self.printer.config_error("Hardware Initialization Error: No active Beacon device found.")
            
        self.probe_x_offset = float(self.beacon_hardware_probe.x_offset)
        self.probe_y_offset = float(self.beacon_hardware_probe.y_offset)

        # Inherit the target bed boundaries using Klipper's native cross-section lookup
        try:
            bed_mesh_config = self.base_config.getsection('bed_mesh')
        except Exception:
            raise self.printer.config_error("Configuration Error: [bed_mesh] section must be defined to inherit parameters.")
            
        self.mesh_minimum_coordinates = bed_mesh_config.getfloatlist('mesh_min', count=2)
        self.mesh_maximum_coordinates = bed_mesh_config.getfloatlist('mesh_max', count=2)

        safe_z_home_object = self.printer.lookup_object('safe_z_home', None)
        if safe_z_home_object is not None:
            self.calibration_center_x = float(safe_z_home_object.home_x)
            self.calibration_center_y = float(safe_z_home_object.home_y)
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
            raise RuntimeError("Kinematics Aborted: Spatial constants are uncalibrated. Run CALIBRATE_SPATIAL_GEOMETRY.")
            
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

    def cmd_CALIBRATE_SPATIAL_GEOMETRY(self, gcode_command: Any) -> None:
        if self.is_mesh_calibrated:
            return

        bed_mesh_module = self.printer.lookup_object('bed_mesh', None)
        if bed_mesh_module is None:
            gcode_command.respond_error("Mesh Error: No active bed_mesh structure initialized.")
            return

        beacon_status_dictionary = self.beacon_hardware_probe.get_status()
        
        fetched_noise_high_fidelity = beacon_status_dictionary.get('contact_precision_deviation', None)
        if fetched_noise_high_fidelity is not None:
            self.high_fidelity_sensor_noise_standard_deviation = float(fetched_noise_high_fidelity)
        else:
            accuracy_samples = []
            for _ in range(10):
                z_strike = float(self.beacon_hardware_probe.run_probe(gcode_command))
                accuracy_samples.append(z_strike)
            self.high_fidelity_sensor_noise_standard_deviation = float(np.std(accuracy_samples))

        fetched_noise_low_fidelity = beacon_status_dictionary.get('proximity_latency_noise', None)
        if fetched_noise_low_fidelity is not None:
            self.low_fidelity_sensor_noise_standard_deviation = float(fetched_noise_low_fidelity)
        else:
            toolhead_object = self.printer.lookup_object('toolhead')
            toolhead_object.manual_move([self.calibration_center_x - self.probe_x_offset, self.calibration_center_y - self.probe_y_offset, 5.0], 50.0)
            self.printer.lookup_object('toolhead').wait_moves()
            
            static_time_series_stream = []
            for _ in range(50):
                static_time_series_stream.append(float(self.beacon_hardware_probe.get_status()['probed_z']))
            self.low_fidelity_sensor_noise_standard_deviation = float(np.std(static_time_series_stream))

        raw_coordinates_low_fidelity = np.array(bed_mesh_module.z_mesh.get_coords())[:, :2]
        raw_heights_low_fidelity = np.array(bed_mesh_module.z_mesh.get_mesh())
        raw_coordinates_high_fidelity = np.array(bed_mesh_module.contact_points)[:, :2]
        raw_heights_high_fidelity = np.array(bed_mesh_module.contact_heights)
        
        transformed_coordinates_low_fidelity = raw_coordinates_low_fidelity + np.array([self.probe_x_offset, self.probe_y_offset])
        transformed_coordinates_high_fidelity = raw_coordinates_high_fidelity + np.array([self.probe_x_offset, self.probe_y_offset])
        
        low_fidelity_boundary_mask = (transformed_coordinates_low_fidelity[:, 0] >= self.mesh_minimum_coordinates[0]) & \
                                     (transformed_coordinates_low_fidelity[:, 0] <= self.mesh_maximum_coordinates[0]) & \
                                     (transformed_coordinates_low_fidelity[:, 1] >= self.mesh_minimum_coordinates[1]) & \
                                     (transformed_coordinates_low_fidelity[:, 1] <= self.mesh_maximum_coordinates[1])
        transformed_coordinates_low_fidelity = transformed_coordinates_low_fidelity[low_fidelity_boundary_mask]
        raw_heights_low_fidelity = raw_heights_low_fidelity[low_fidelity_boundary_mask]
        
        high_fidelity_boundary_mask = (transformed_coordinates_high_fidelity[:, 0] >= self.mesh_minimum_coordinates[0]) & \
                                      (transformed_coordinates_high_fidelity[:, 0] <= self.mesh_maximum_coordinates[0]) & \
                                      (transformed_coordinates_high_fidelity[:, 1] >= self.mesh_minimum_coordinates[1]) & \
                                      (transformed_coordinates_high_fidelity[:, 1] <= self.mesh_maximum_coordinates[1])
        transformed_coordinates_high_fidelity = transformed_coordinates_high_fidelity[high_fidelity_boundary_mask]
        raw_heights_high_fidelity = raw_heights_high_fidelity[high_fidelity_boundary_mask]

        high_fidelity_coordinate_distances = np.linalg.norm(transformed_coordinates_high_fidelity[:, None, :] - transformed_coordinates_high_fidelity[None, :, :], axis=-1)
        micro_spatial_pairs_mask = (high_fidelity_coordinate_distances > 0.0) & (high_fidelity_coordinate_distances <= 15.0)
        
        if not np.any(micro_spatial_pairs_mask):
            toolhead_object = self.printer.lookup_object('toolhead')
            
            cross_target_nodes = np.array([
                [self.calibration_center_x, self.calibration_center_y],
                [self.calibration_center_x + 10.0, self.calibration_center_y],
                [self.calibration_center_x - 10.0, self.calibration_center_y],
                [self.calibration_center_x, self.calibration_center_y + 10.0],
                [self.calibration_center_x, self.calibration_center_y - 10.0]
            ])
            
            seeded_coordinates = []
            seeded_heights = []
            
            for target_node in cross_target_nodes:
                toolhead_object.manual_move([target_node[0] - self.probe_x_offset, target_node[1] - self.probe_y_offset, 5.0], 50.0)
                self.printer.lookup_object('toolhead').wait_moves()
                
                absolute_height_strike = float(self.beacon_hardware_probe.run_probe(gcode_command))
                seeded_coordinates.append(target_node)
                seeded_heights.append(absolute_height_strike)
                
            transformed_coordinates_high_fidelity = np.vstack((transformed_coordinates_high_fidelity, np.array(seeded_coordinates)))
            raw_heights_high_fidelity = np.concatenate((raw_heights_high_fidelity, np.array(seeded_heights)))
            
            high_fidelity_coordinate_distances = np.linalg.norm(transformed_coordinates_high_fidelity[:, None, :] - transformed_coordinates_high_fidelity[None, :, :], axis=-1)
            micro_spatial_pairs_mask = (high_fidelity_coordinate_distances > 0.0) & (high_fidelity_coordinate_distances <= 15.0)

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

        if self.skew_method == 'non_linear_field':
            dense_projected_deviations_x = []
            dense_projected_deviations_y = []
            
            for index, node in enumerate(self.skew_control_nodes):
                angular_direction = float(np.arctan2(node[1], node[0]))
                dense_projected_deviations_x.append(dense_predicted_discrepancy_field[index] * np.cos(angular_direction))
                dense_projected_deviations_y.append(dense_predicted_discrepancy_field[index] * np.sin(angular_direction))
                
            self.thin_plate_spline_coefficients_x = self._solve_thin_plate_spline_system(self.skew_control_nodes, np.array(dense_projected_deviations_x))
            self.thin_plate_spline_coefficients_y = self._solve_thin_plate_spline_system(self.skew_control_nodes, np.array(dense_projected_deviations_y))
            
        elif self.skew_method == 'traditional':
            xy_distance = gcode_command.get_float('XY_DIST', default=None)
            xz_distance = gcode_command.get_float('XZ_DIST', default=None)
            yz_distance = gcode_command.get_float('YZ_DIST', default=None)
            
            if xy_distance is None or xz_distance is None or yz_distance is None:
                raise RuntimeError("Kinematics Error: Traditional skew evaluation requires explicit definitions for 'XY_DIST', 'XZ_DIST', and 'YZ_DIST'.")
            
            cosine_gamma = (xy_distance**2 + xz_distance**2 - yz_distance**2) / (2.0 * xy_distance * xz_distance)
            sine_gamma = np.sqrt(1.0 - cosine_gamma**2)
            tangent_alpha = cosine_gamma / sine_gamma
            
            if self.traditional_skew_mode == 'xy':
                self.traditional_alignment_matrix = np.array([[1.0, -tangent_alpha], [0.0, 1.0 / sine_gamma]])
            elif self.traditional_skew_mode == 'x':
                self.traditional_alignment_matrix = np.array([[1.0, -tangent_alpha], [0.0, 1.0]])
            elif self.traditional_skew_mode == 'y':
                self.traditional_alignment_matrix = np.array([[1.0, 0.0], [-tangent_alpha, 1.0 / sine_gamma]])
                
        self.is_mesh_calibrated = True

    def cmd_RECALIBRATE_SPATIAL_GEOMETRY(self, gcode_command: Any) -> None:
        self.is_mesh_calibrated = False
        self.cmd_CALIBRATE_SPATIAL_GEOMETRY(gcode_command)

def load_config(config: Any) -> FusedMeshSkewTransform:
    return FusedMeshSkewTransform(config)