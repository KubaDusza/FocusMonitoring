# 🎯 Focus Monitoring System 🖥️

A system for real-time head-pose estimation, focus tracking, and data visualization. It allows users to monitor attention zones, generate heatmaps of looking directions, and analyze focus over time.

## Features 🚀
- **Head-Pose Calibration:** Set a neutral head pose for accurate tracking.
- **Looking Direction Recording:** Record multiple predefined directions.
- **Focus Zone Tracking:** Monitor if the user's gaze is within defined zones.
- **Heatmap Visualization:** Generate 2D heatmaps of looking directions.
- **Attention Analysis:** Graphs to analyze time spent focusing on specific zones.

## Technologies Used 🛠️
- **Python**
- **Streamlit:** Web interface
- **MediaPipe:** Head-pose estimation
- **Matplotlib:** Visualization
- **NumPy:** Mathematical operations
- **Scipy:** Rotation and transformations

## Installation 🖥️
Follow these steps to set up the system locally:

1. Clone the repository:
    ```bash
    git clone https://github.com/KubaDusza/FocusMonitoring.git
    cd FocusMonitoring
    ```
2. Create a virtual environment and activate it:
    - On MacOS/Linux:
      ```bash
      python3 -m venv venv && source venv/bin/activate
      ```
    - On Windows:
      ```bash
      python -m venv venv && venv\Scripts\activate
      ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the app:
    ```bash
    streamlit run app.py
    ```

## How to Use 🤓
1. **Calibration:**
   - Position your head in a neutral pose and click the "Calibrate" button.
2. **Define Zones:**
   - Record specific looking directions by clicking "Record Direction."
3. **Monitor Focus:**
   - The system will track if your gaze stays within the predefined zones.
4. **Generate Heatmaps:**
   - Use the "Generate Heatmap" button to visualize gaze distribution.
5. **Analyze Attention:**
   - View time-series graphs for focus percentage, average gaze duration, and adherence over time.

## TODOs 📋
- make it more usable
- add recording buttons
- make the zone selecting better
- reduce lag
---

