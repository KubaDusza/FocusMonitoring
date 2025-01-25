# estimators/base_estimator.py
from abc import ABC, abstractmethod
import numpy as np


class BaseEstimator(ABC):
    """Abstract base class for estimators providing a template for posture and head pose analysis."""

    def __init__(self):
        self.keypoints_2d = None
        self.keypoints_3d = None

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the estimator."""
        pass

    @abstractmethod
    def estimate(self, frame: np.ndarray) -> dict:
        """
        Process a frame and return estimation results.

        Args:
            frame (np.ndarray): Input video frame.
        Returns:
            dict: Rotation and translation vectors.
        """
        pass

    @abstractmethod
    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw estimation results on the frame.

        Args:
            frame (np.ndarray): The frame on which to draw.
        Returns:
            np.ndarray: The modified frame with drawings.
        """
        pass

    def get_keypoints_2d(self):
        """Get the last computed 2D keypoints."""
        return self.keypoints_2d

    def get_keypoints_3d(self):
        """Get the last computed 3D keypoints."""
        return self.keypoints_3d


class BaseHeadPoseEstimator(BaseEstimator):
    """Base class for head pose estimators, extending BaseEstimator."""

    def __init__(self):
        super().__init__()
        self.rotation_vector = None
        self.translation_vector = None

    @abstractmethod
    def get_head_pose(self) -> dict:
        """
        Return head pose data including rotation and translation vectors.

        Returns:
            dict: Contains 'rotation_vector' and 'translation_vector'.
        """
        pass



