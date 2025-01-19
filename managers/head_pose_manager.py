import numpy as np
from scipy.spatial.transform import Rotation as R

import estimators.base_estimator
import observers.subject
import time



class HeadPoseManager(observers.subject.Subject):
    """
    Manages head pose estimation, calibration, and additional functionality.
    """

    def __init__(self, estimator: estimators.base_estimator.BaseHeadPoseEstimator, zone_threshold=np.radians(15), notify_interval=0.1):
        """
        Initialize the manager.

        Args:
            estimator: An instance of MediaPipeHeadPoseEstimator.
            zone_threshold: Radius of the allowable zone (in radians).
        """
        super().__init__()
        self.estimator = estimator
        self.calibration_matrix = np.eye(3)  # Identity matrix for neutral pose
        self.raw_recorded_directions = []  # Store calibrated matrices
        self.zone_threshold = zone_threshold  # Allowable error for zone checking
        self.current_calibrated_rotation_matrix = np.eye(3)
        self.current_raw_rotation_matrix = np.eye(3)

        self.last_notify_time = 0  # Track the last notification time
        self.notify_interval = notify_interval  # Notification interval in seconds

    def calibrate(self):
        """
        Calibrate the current head orientation as the neutral pose.

        Args:
            rotation_vector: Rotation vector from the estimator.
        """

        self.calibration_matrix = np.linalg.inv(self.current_raw_rotation_matrix)

        print("Calibration complete! Current pose set as neutral.")

    def process_frame(self, frame):
        """
        Process a frame, apply calibration, and return results.

        Args:
            frame: The current video frame.

        Returns:
            Tuple: Contains calibrated and raw pose data.
        """
        raw_data = self.estimator.estimate(frame)
        rotation_vector = raw_data.get("rotation_vector")
        if rotation_vector is None:
            return None

        # Apply calibration
        rotation_matrix = R.from_rotvec(rotation_vector).as_matrix()
        calibrated_matrix = self.calibration_matrix @ rotation_matrix

        self.current_calibrated_rotation_matrix = calibrated_matrix
        self.current_raw_rotation_matrix = rotation_matrix

        calibrated_rotation_vector = R.from_matrix(calibrated_matrix).as_rotvec()

        current_time = time.time()
        if current_time - self.last_notify_time > self.notify_interval:
            self.last_notify_time = current_time
            self.notify_observers(
                {"rotation_vector": calibrated_rotation_vector,
                 "is_looking_within_zone": self.is_looking_within_zone()})

        return calibrated_matrix

    def reset_recorded_looking_directions(self):
        self.raw_recorded_directions = []
        print("reset the recorded looking directions")

    def record_looking_direction(self):
        """
        Record a head direction for later reference, calibrated to the current neutral pose.

        Args:
            rotation_vector: Current rotation vector from the estimator.
        """

        self.raw_recorded_directions.append(self.current_raw_rotation_matrix)
        print(f"Direction {len(self.raw_recorded_directions)} recorded!")

    def is_looking_within_zone(self):
        """
        Check if the current head pose is within the defined zone.

        Args:
            rotation_vector: Current rotation vector from the estimator.

        Returns:
            bool: True if within the zone, False otherwise.
        """

        # The forward direction vector in the calibrated space
        current_direction = self.current_calibrated_rotation_matrix @ np.array([0, 0, 1])

        # Check against recorded directions
        for raw_recorded_matrix in self.raw_recorded_directions:
            recorded_direction = self.calibration_matrix @ raw_recorded_matrix @ np.array([0, 0, 1])

            # Compute angle between vectors
            angle = np.arccos(np.clip(np.dot(current_direction, recorded_direction), -1.0, 1.0))

            if angle < self.zone_threshold:
                return True
        return False

    def set_zone_threshold(self, threshold_degrees):
        """
        Dynamically update the zone threshold.

        Args:
            threshold_degrees: New threshold in degrees.
        """
        self.zone_threshold = np.radians(threshold_degrees)
        print(f"Zone threshold updated to {threshold_degrees} degrees.")

    def get_calibrated_recorded_directions(self):
        return [self.calibration_matrix @ direction for direction in self.raw_recorded_directions]


