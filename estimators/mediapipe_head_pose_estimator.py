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
        self.rotation_vector = None
        self.translation_vector = None

    def initialize(self) -> None:
        """Initialize MediaPipe Face Mesh."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.drawing_utils = mp.solutions.drawing_utils
        self.drawing_spec = self.drawing_utils.DrawingSpec(
            color=(0, 255, 0), thickness=1, circle_radius=1
        )

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

    def print_landmark_visibilities(self):
        """Print the visibility of the selected landmarks for debugging."""
        landmark_indices = [1, 152, 33, 263, 61, 291]
        print("Landmark Visibilities:")

        if self.landmarks:
            for idx in landmark_indices:
                landmark = self.landmarks.landmark[idx]
                print(f"Landmark {idx}: Visibility = {landmark.visibility:.2f}")
        else:
            print("No landmarks detected.")

    def _solve_head_pose(self, frame: np.ndarray) -> None:
        """SolvePnP to calculate head pose."""
        # Define 3D model points corresponding to the selected landmarks
        model_points = np.array(
            [
                [0.0, 0.0, 0.0],  # Nose tip
                [0.0, -330.0, -65.0],  # Chin
                [-225.0, 170.0, -135.0],  # Left eye left corner
                [225.0, 170.0, -135.0],  # Right eye right corner
                [-150.0, -150.0, -125.0],  # Left Mouth corner
                [150.0, -150.0, -125.0],  # Right Mouth corner
                [-300.0, 0.0, -150.0],  # Left cheek
                [300.0, 0.0, -150.0],  # Right cheek
            ]
        )

        # Extract 2D coordinates of the corresponding landmarks
        self.keypoints_2d = np.array(
            [
                (
                    self.landmarks.landmark[1].x * frame.shape[1],
                    self.landmarks.landmark[1].y * frame.shape[0],
                ),  # Nose tip
                (
                    self.landmarks.landmark[152].x * frame.shape[1],
                    self.landmarks.landmark[152].y * frame.shape[0],
                ),  # Chin
                (
                    self.landmarks.landmark[33].x * frame.shape[1],
                    self.landmarks.landmark[33].y * frame.shape[0],
                ),
                # Left eye left corner
                (
                    self.landmarks.landmark[263].x * frame.shape[1],
                    self.landmarks.landmark[263].y * frame.shape[0],
                ),
                # Right eye right corner
                (
                    self.landmarks.landmark[61].x * frame.shape[1],
                    self.landmarks.landmark[61].y * frame.shape[0],
                ),
                # Left Mouth corner
                (
                    self.landmarks.landmark[291].x * frame.shape[1],
                    self.landmarks.landmark[291].y * frame.shape[0],
                ),
                # Right Mouth corner
                (
                    self.landmarks.landmark[234].x * frame.shape[1],
                    self.landmarks.landmark[234].y * frame.shape[0],
                ),
                # Left cheek
                (
                    self.landmarks.landmark[454].x * frame.shape[1],
                    self.landmarks.landmark[454].y * frame.shape[0],
                ),
                # Right cheek
            ],
            dtype="double",
        )

        # Camera internals
        focal_length = frame.shape[1]
        center = (frame.shape[1] / 2, frame.shape[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype="double",
        )

        # SolvePnP to estimate head pose
        success, self.rotation_vector, self.translation_vector = cv2.solvePnP(
            model_points, self.keypoints_2d, camera_matrix, None
        )

        if not success:
            print("Head pose estimation failed.")

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw face mesh and head pose axes on the frame."""
        if self.landmarks:
            self.drawing_utils.draw_landmarks(
                frame,
                self.landmarks,
                self.mp_face_mesh.FACEMESH_CONTOURS,
                self.drawing_spec,
                self.drawing_spec,
            )

            axis = np.float32([[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]])

            camera_matrix = np.array(
                [
                    [frame.shape[1], 0, frame.shape[1] / 2],
                    [0, frame.shape[1], frame.shape[0] / 2],
                    [0, 0, 1],
                ],
                dtype="double",
            )

            img_points, _ = cv2.projectPoints(
                axis, self.rotation_vector, self.translation_vector, camera_matrix, None
            )

            origin = tuple(np.int32(img_points[0].ravel()))
            frame = cv2.line(
                frame, origin, tuple(np.int32(img_points[1].ravel())), (0, 0, 255), 2
            )  # X-axis
            frame = cv2.line(
                frame, origin, tuple(np.int32(img_points[2].ravel())), (0, 255, 0), 2
            )  # Y-axis
            frame = cv2.line(
                frame, origin, tuple(np.int32(img_points[3].ravel())), (255, 0, 0), 2
            )  # Z-axis

        return frame

    def get_head_pose(self) -> dict:
        """Return rotation and translation vectors."""
        if self.rotation_vector is None:
            return {"rotation_vector": None, "translation_vector": None}

        return {
            "rotation_vector": self.rotation_vector.flatten(),
            "translation_vector": self.translation_vector.flatten(),
        }
