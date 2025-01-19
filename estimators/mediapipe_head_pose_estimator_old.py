# estimators/mediapipe_head_pose_estimator.py
import mediapipe as mp
import cv2
import numpy as np
from estimators.base_estimator import BaseHeadPoseEstimator


class MediaPipeHeadPoseEstimator(BaseHeadPoseEstimator):
    """Concrete implementation using MediaPipe for head pose estimation."""

    def __init__(self):
        super().__init__()
        self.face_mesh = None
        self.drawing_utils = None
        self.landmarks = None

    def initialize(self) -> None:
        """Initialize MediaPipe Face Mesh."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True
        )
        self.drawing_utils = mp.solutions.drawing_utils
        self.drawing_spec = self.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

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
        """SolvePnP to calculate head pose."""
        model_points = np.array([
            [0.0, 0.0, 0.0],
            [0.0, -330.0, -65.0],
            [-225.0, 170.0, -135.0],
            [225.0, 170.0, -135.0],
            [-150.0, -150.0, -125.0],
            [150.0, -150.0, -125.0]
        ])

        self.keypoints_2d = np.array([
            (self.landmarks.landmark[1].x * frame.shape[1], self.landmarks.landmark[1].y * frame.shape[0]),
            (self.landmarks.landmark[152].x * frame.shape[1], self.landmarks.landmark[152].y * frame.shape[0]),
            (self.landmarks.landmark[33].x * frame.shape[1], self.landmarks.landmark[33].y * frame.shape[0]),
            (self.landmarks.landmark[263].x * frame.shape[1], self.landmarks.landmark[263].y * frame.shape[0]),
            (self.landmarks.landmark[61].x * frame.shape[1], self.landmarks.landmark[61].y * frame.shape[0]),
            (self.landmarks.landmark[291].x * frame.shape[1], self.landmarks.landmark[291].y * frame.shape[0])
        ], dtype="double")

        focal_length = frame.shape[1]
        center = (frame.shape[1] / 2, frame.shape[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        # SolvePnP to estimate head pose
        _, self.rotation_vector, self.translation_vector = cv2.solvePnP(
            model_points, self.keypoints_2d, camera_matrix, None
        )

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw face mesh and head pose axes on the frame."""
        if self.landmarks:
            self.drawing_utils.draw_landmarks(
                frame,
                self.landmarks,
                self.mp_face_mesh.FACEMESH_CONTOURS,
                self.drawing_spec,
                self.drawing_spec
            )

            axis = np.float32([
                [0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]
            ])

            camera_matrix = np.array([
                [frame.shape[1], 0, frame.shape[1] / 2],
                [0, frame.shape[1], frame.shape[0] / 2],
                [0, 0, 1]
            ], dtype="double")



            img_points, _ = cv2.projectPoints(
                axis, self.rotation_vector, self.translation_vector, camera_matrix, None
            )

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


