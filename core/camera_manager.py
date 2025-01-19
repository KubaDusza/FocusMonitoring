import cv2

class CameraManager:
    """A flexible camera manager to handle multiple cameras and video feeds."""
    MAX_CAMERA_COUNT = 2  # Maximum number of cameras to scan for


    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None

    def start_camera(self):
        """Start the camera feed."""
        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera index {self.camera_index}")

    def get_frame(self):
        """Capture a frame and return it as RGB."""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                raise RuntimeError("Failed to capture a frame.")
        raise RuntimeError("Camera is not started.")



    def stop_camera(self):
        """Stop the camera and release resources."""
        if self.cap:
            self.cap.release()
            self.cap = None

    def switch_camera(self, new_index):
        """Switch to a different camera."""
        self.stop_camera()
        self.camera_index = new_index
        self.start_camera()

    def switch_to_next_camera(self):
        """Switch between available cameras."""
        self.switch_camera((self.camera_index + 1) % self.MAX_CAMERA_COUNT)



