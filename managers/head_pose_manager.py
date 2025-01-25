import numpy as np
from scipy.spatial.transform import Rotation as R
import estimators.base_estimator
import observers.subject
import time


class HeadPoseManager(observers.subject.Subject):
    """
    Manages head pose estimation, calibration, and zone tracking functionality.
    """

    def __init__(self, estimator: estimators.base_estimator.BaseHeadPoseEstimator, zone_threshold=np.radians(15), notify_interval=0.1):
        """
        Initialize the head pose manager with the specified estimator and parameters.

        Args:
            estimator (BaseHeadPoseEstimator): Instance for head pose estimation.
            zone_threshold (float): Threshold for the allowable zone (in radians).
            notify_interval (float): Interval for notifying observers (in seconds).
        """
        super().__init__()
        self.estimator = estimator
        self.calibration_matrix = np.eye(3)  # Identity matrix for neutral pose calibration
        self.raw_recorded_directions = []  # List of calibrated rotation matrices
        self.zone_threshold = zone_threshold
        self.current_calibrated_rotation_matrix = np.eye(3)
        self.current_raw_rotation_matrix = np.eye(3)
        self.last_notify_time = 0  # Timestamp of the last notification
        self.notify_interval = notify_interval

    def calibrate(self):
        """
        Calibrate the current head orientation as the neutral pose.
        """
        self.calibration_matrix = np.linalg.inv(self.current_raw_rotation_matrix)
        print("Calibration complete! Current pose set as neutral.")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a video frame, apply calibration, and return the calibrated rotation matrix.

        Args:
            frame (np.ndarray): Current video frame to be processed.

        Returns:
            np.ndarray: Calibrated rotation matrix. None if estimation fails.
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

        # Notify observers periodically
        current_time = time.time()
        if current_time - self.last_notify_time > self.notify_interval:
            self.last_notify_time = current_time
            self.notify_observers(
                {"rotation_vector": calibrated_rotation_vector,
                 "is_looking_within_zone": self.is_looking_within_zone()})

        return calibrated_matrix

    def reset_recorded_looking_directions(self):
        """
        Reset the list of recorded looking directions.
        """
        self.raw_recorded_directions = []
        print("Reset the recorded looking directions.")

    def record_looking_direction(self):
        """
        Record the current head direction for later reference.
        """
        self.raw_recorded_directions.append(self.current_raw_rotation_matrix)
        print(f"Direction {len(self.raw_recorded_directions)} recorded!")

    def is_looking_within_zone(self) -> bool:
        """
        Check if the current head pose is within the defined allowable zone.

        Returns:
            bool: True if the current pose is within the zone, False otherwise.
        """
        current_direction = self.current_calibrated_rotation_matrix @ np.array([0, 0, 1])

        for raw_recorded_matrix in self.raw_recorded_directions:
            recorded_direction = self.calibration_matrix @ raw_recorded_matrix @ np.array([0, 0, 1])

            # Compute the angle between the current and recorded directions
            angle = np.arccos(np.clip(np.dot(current_direction, recorded_direction), -1.0, 1.0))
            if angle < self.zone_threshold:
                return True
        return False

    def set_zone_threshold(self, threshold_degrees: float):
        """
        Update the zone threshold dynamically.

        Args:
            threshold_degrees (float): New threshold in degrees.
        """
        self.zone_threshold = np.radians(threshold_degrees)
        print(f"Zone threshold updated to {threshold_degrees} degrees.")

    def get_calibrated_recorded_directions(self) -> list:
        """
        Retrieve the list of calibrated recorded head directions.

        Returns:
            list: Calibrated recorded rotation matrices.
        """
        return [self.calibration_matrix @ direction for direction in self.raw_recorded_directions]
