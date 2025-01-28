# Head Pose Estimation and Visualization Program

## Overview

This program provides real-time head pose estimation, visualization, and zone adherence tracking using MediaPipe's face mesh module. It employs design patterns such as Observer, Subject, and Strategy to decouple components and enable extensibility. The program captures live video, processes head pose data, visualizes the results, and generates analytical insights (heatmaps and graphs).

---

## Classes and Design Patterns

### 1. **Managers**

### **`CameraManager`**
**Design Pattern**: Encapsulation of hardware control  
Manages camera operations, including starting/stopping the camera and switching between multiple cameras.

- **`start_camera()`**: Initializes the video capture. Raises an error if the camera cannot be opened.
- **`get_frame()`**: Captures and processes a frame (downscaling and flipping). Returns the frame in RGB format.
- **`stop_camera()`**: Releases the camera resource.
- **`switch_camera(index: int)`**: Switches to a specific camera index or cycles through available cameras.

### **`HeadPoseManager`**
**Design Pattern**: Subject in the Observer pattern  
Handles head pose estimation, calibration, and zone adherence tracking. Manages notifications to observers (e.g., heatmap and graph generators).

- **`process_frame(frame: np.ndarray) -> RotationMatrix`**:  
  - Estimates head pose from the frame.
  - Applies calibration to compute the relative head pose.
  - Converts rotation vectors to rotation matrices using Rodrigues' rotation formula.
  - Notifies observers with the calibrated rotation vector and zone status.

- **`calibrate()`**:  
  - Sets the current head orientation as neutral by computing the inverse of the current rotation matrix.
  - Updates the calibration matrix for future transformations.

- **`record_looking_direction()`**:  
  - Records the current head direction after calibration. Stores raw rotation matrices.

- **`reset_recorded_looking_directions()`**:  
  - Clears all recorded head directions.

- **`is_looking_within_zone()` -> `bool`**:  
  - Checks whether the current head pose is within the defined threshold zone.
  - Uses the dot product to compute the angle between the current calibrated direction and recorded directions.
  - Returns `True` if the angle is less than the threshold.

- **`set_zone_threshold(threshold: float)`**: Updates the angular threshold for zone adherence.

- **`get_calibrated_recorded_directions()` -> `List[RotationMatrix]`**:  
  - Returns the calibrated directions for all recorded head poses.

---

### 2. **Estimators**

### **`BaseHeadPoseEstimator`**
**Design Pattern**: Abstract Base Class  
Defines a common interface for head pose estimators.

- **`initialize()`**: Abstract method to initialize the estimator.
- **`estimate(frame: np.ndarray) -> dict`**: Abstract method to estimate head pose.
- **`draw(frame: np.ndarray) -> np.ndarray`**: Abstract method to visualize estimation results.

### **`MediaPipeHeadPoseEstimator`**
**Design Pattern**: Strategy (Concrete implementation)  
Implements head pose estimation using MediaPipe's Face Mesh module.

- **`initialize()`**:  
  - Initializes MediaPipe Face Mesh with parameters for real-time tracking.

- **`estimate(frame: np.ndarray) -> dict`**:  
  - Converts the frame to RGB and processes it using MediaPipe.
  - Extracts facial landmarks and computes rotation and translation vectors using `cv2.solvePnP`.
  - Uses a 3D head model to map 2D landmarks to 3D space.

- **`_solve_head_pose(frame: np.ndarray)`**:  
  - Defines 3D model points for key facial landmarks (e.g., nose tip, eyes, and mouth).
  - Maps detected 2D landmarks to 3D space using camera intrinsic parameters.
  - Solves for rotation and translation vectors using the `cv2.solvePnP` algorithm.

- **`draw(frame: np.ndarray) -> np.ndarray`**:  
  - Draws the facial landmarks and head pose axes (X, Y, Z) on the frame.

- **`get_head_pose() -> dict`**:  
  - Returns the computed rotation and translation vectors.

---

### 3. **Observers**

### **`Observer` (Abstract Class)**
**Design Pattern**: Abstract Base Class for Observer pattern  
Defines the `update(data: dict)` method to receive notifications from subjects.

### **`Subject` (Abstract Class)**

**Design Pattern**: Subject in the Observer pattern  
The `Subject` class serves as a base for managing observers and notifying them of updates. It provides mechanisms to register, remove, and notify observers.

- **`__init__()`**:  
  Initializes an empty list of observers.

- **`add_observer(observer: Observer)`**:  
  Adds an observer to the list.  
  **Args**:  
  - `observer` (Observer): An instance of a class implementing the `Observer` interface.

- **`remove_observer(observer: Observer)`**:  
  Removes an observer from the list.  
  **Args**:  
  - `observer` (Observer): The observer to be removed.

- **`notify_observers(data: dict)`**:  
  Notifies all registered observers by calling their `update()` method and passing the `data`.  
  **Args**:  
  - `data` (dict): The data to be passed to the observers, typically containing state updates such as rotation vectors or zone adherence status.


### **`HeatmapGenerator`**
**Design Pattern**: Observer  
Generates heatmaps of looking directions based on the rotation vector.

- **`update(data: dict)`**:  
  - Records the new rotation vector data.
  
- **`record_data(rotation_vector: np.ndarray)`**:  
  - Projects the rotation vector onto a 2D plane (Z=constant) using matrix transformations.

- **`generate_heatmap()`**:  
  - Computes a density heatmap of recorded looking directions.
  - Visualizes optional focus areas as cones with angular thresholds.

### **`GraphGenerator`**
**Design Pattern**: Observer  
Generates time-based graphs for zone adherence and gaze duration.

- **`update(data: dict)`**:  
  - Tracks timestamps and zone adherence status in time series.
  
- **`plot_all_graphs()`**:  
  - Visualizes cumulative zone adherence, gaze duration, and sliding window adherence.

---

### 4. **Visualizer**

### **`HeadPoseVisualizer`**
Visualizes the 3D head pose and recorded directions using Matplotlib.

- **`set_visualization_rotation(angles: List[float])`**:  
  - Applies a global rotation to the visualization.

- **`plot_head_pose(rotation_matrix: np.ndarray, directions: List[np.ndarray], threshold: float, placeholder)`**:  
  - Renders the calibrated head pose, recorded directions, and cones indicating focus zones.

---

### 5. **Main**

#### **`setup()`**
Initializes session state objects, including the camera manager, head pose estimator, heatmap generator, and graph generator. Observers are registered to the head pose manager.

#### **`main()`**
- Captures user inputs from the Streamlit interface.
- Manages camera operations and frame processing.
- Updates and renders visualizations, heatmaps, and graphs.
- Tracks user interactions for head pose calibration and direction recording.
