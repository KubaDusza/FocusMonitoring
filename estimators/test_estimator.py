import mediapipe as mp
import cv2
import numpy as np
from collections import deque
from estimators.base_estimator import BaseHeadPoseEstimator
from scipy.spatial.transform import Rotation as R


class MediaPipeHeadPoseEstimator(BaseHeadPoseEstimator):
    """Concrete implementation using MediaPipe for head pose estimation with improvements."""

    def __init__(self):
        super().__init__()
        self.face_mesh = None
        self.drawing_utils = None
        self.drawing_spec = None
        self.landmarks = None
        self.rotation_buffer = deque(maxlen=5)  # For temporal smoothing
        self.translation_buffer = deque(maxlen=5)
        self.initialize_kalman_filter()

    def initialize(self) -> None:
        """Initialize MediaPipe Face Mesh."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True
        )
        self.drawing_utils = mp.solutions.drawing_utils
        self.drawing_spec = self.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

    def initialize_kalman_filter(self):
        """Initialize Kalman Filter for head pose smoothing."""
        self.kalman = cv2.KalmanFilter(6, 3)
        self.kalman.measurementMatrix = np.eye(3, 6, dtype=np.float32)
        self.kalman.transitionMatrix = np.eye(6, dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-3
        self.kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-1

    def kalman_filter_update(self, rotation_vector):
        """Apply Kalman Filter smoothing."""
        measurement = np.array(rotation_vector, dtype=np.float32).reshape(3, 1)
        self.kalman.correct(measurement)
        prediction = self.kalman.predict()
        return prediction.flatten()

    def estimate(self, frame: np.ndarray) -> dict:
        """
        Estimate head pose and calculate rotation and translation vectors.

        Args:
            frame (np.ndarray): The input video frame.
        Returns:
            dict: Rotation and translation vectors.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            self.landmarks = results.multi_face_landmarks[0]
            self._solve_head_pose(frame)

        return self.get_head_pose()

    def _solve_head_pose(self, frame: np.ndarray) -> None:
        """SolvePnP to calculate head pose using multiple landmarks and improved visibility handling."""
        keypoint_ids = [1, 33, 61, 152, 263, 291, 10, 234, 454]  # More landmarks
        model_points = np.array([
            [0.0, 0.0, 0.0],  # Nose tip
            [-225.0, 170.0, -135.0],  # Left eye
            [-150.0, -150.0, -125.0],  # Left mouth corner
            [0.0, -330.0, -65.0],  # Chin
            [225.0, 170.0, -135.0],  # Right eye
            [150.0, -150.0, -125.0],  # Right mouth corner
            [0.0, 0.0, 0.0],  # Forehead center
            [-300.0, 0.0, -100.0],  # Left side of face
            [300.0, 0.0, -100.0]  # Right side of face
        ])

        keypoints_2d = []
        valid_model_points = []

        # Improved landmark visibility check
        for idx, point in enumerate(keypoint_ids):
            landmark = self.landmarks.landmark[point]

            # Adjusted threshold for more leniency
            if (landmark.visibility > 0.1 or  # Lower visibility threshold
                    (0 <= landmark.x <= 1 and 0 <= landmark.y <= 1)):  # Backup check for position inside frame
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                keypoints_2d.append([x, y])
                valid_model_points.append(model_points[idx])

        # Ensure a minimum number of landmarks
        if len(keypoints_2d) < 4:
            print("Not enough visible landmarks for stable estimation.")
            return

        # Convert to NumPy arrays
        keypoints_2d = np.array(keypoints_2d, dtype="double")
        valid_model_points = np.array(valid_model_points, dtype="double")

        # Camera calibration
        focal_length = frame.shape[1]
        center = (frame.shape[1] / 2, frame.shape[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        # SolvePnP for pose estimation
        _, rotation_vector, translation_vector = cv2.solvePnP(
            valid_model_points, keypoints_2d, camera_matrix, None
        )

        # Apply Kalman filter and smoothing
        self.rotation_buffer.append(rotation_vector)
        #smoothed_rotation = np.mean(self.rotation_buffer, axis=0)
        #self.rotation_vector = self.kalman_filter_update(smoothed_rotation)
        self.rotation_vector = rotation_vector
        self.translation_vector = translation_vector

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw face mesh and head pose axes on the frame."""
        if self.landmarks and self.rotation_vector is not None and self.translation_vector is not None:
            self.drawing_utils.draw_landmarks(
                frame,
                self.landmarks,
                self.mp_face_mesh.FACEMESH_CONTOURS,
                self.drawing_spec,
                self.drawing_spec
            )

            # Extract only the first three components for rotation vector
            rotation_vector = np.array(self.rotation_vector[:3], dtype=np.float64).reshape(3, 1)
            translation_vector = np.array(self.translation_vector, dtype=np.float64).reshape(3, 1)

            # Define the axis for the pose visualization
            axis = np.float32([
                [0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]
            ])

            camera_matrix = np.array([
                [frame.shape[1], 0, frame.shape[1] / 2],
                [0, frame.shape[1], frame.shape[0] / 2],
                [0, 0, 1]
            ], dtype="double")

            # Ensure valid input matrices before projecting points
            try:
                img_points, _ = cv2.projectPoints(
                    axis, rotation_vector, translation_vector, camera_matrix, None
                )
            except cv2.error as e:
                print(f"Error during projectPoints: {e}")
                return frame

            # Draw the pose axis lines on the frame
            origin = tuple(np.int32(img_points[0].ravel()))
            frame = cv2.line(frame, origin, tuple(np.int32(img_points[1].ravel())), (0, 0, 255), 2)  # X-axis
            frame = cv2.line(frame, origin, tuple(np.int32(img_points[2].ravel())), (0, 255, 0), 2)  # Y-axis
            frame = cv2.line(frame, origin, tuple(np.int32(img_points[3].ravel())), (255, 0, 0), 2)  # Z-axis

        return frame

    def get_head_pose(self) -> dict:
        """Return rotation and translation vectors."""
        return {
            "rotation_vector": self.rotation_vector,
            "translation_vector": self.translation_vector
        }
