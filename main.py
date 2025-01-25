import streamlit as st
import matplotlib.pyplot as plt

from managers.camera_manager import CameraManager
from managers.head_pose_manager import HeadPoseManager

from estimators.mediapipe_head_pose_estimator import MediaPipeHeadPoseEstimator

from visualizer.head_pose_visualizer import HeadPoseVisualizer

from observers.heatmap_generator import HeatmapGenerator
from observers.graph_generator import GraphGenerator


def setup():
    if "camera_manager" not in st.session_state:
        st.session_state.camera_manager = CameraManager(camera_index=0)

    if "head_pose_estimator" not in st.session_state:
        st.session_state.head_pose_estimator = MediaPipeHeadPoseEstimator()
        st.session_state.head_pose_estimator.initialize()

    if "visualizer" not in st.session_state:
        st.session_state.visualizer = HeadPoseVisualizer([90, 0, 0])

    if "heatmap_generator" not in st.session_state:
        st.session_state.heatmap_generator = HeatmapGenerator()

    if "graph_generator" not in st.session_state:
        st.session_state.graph_generator = GraphGenerator()

    if "head_pose_manager" not in st.session_state:
        st.session_state.head_pose_manager = HeadPoseManager(
            st.session_state.head_pose_estimator,
            notify_interval=0.5,
        )
        st.session_state.head_pose_manager.add_observer(
            st.session_state.heatmap_generator
        )
        st.session_state.head_pose_manager.add_observer(
            st.session_state.graph_generator
        )


def main():
    setup()
    st.title("Head Pose Estimation with Zone Tracking")


    camera_manager: CameraManager = st.session_state.camera_manager
    head_pose_manager: HeadPoseManager = st.session_state.head_pose_manager
    visualizer: HeadPoseVisualizer = st.session_state.visualizer
    heatmap_generator: HeatmapGenerator = st.session_state.heatmap_generator
    graph_generator: GraphGenerator = st.session_state.graph_generator

    FRAME_SKIP = 2
    PLOT_SKIP = 2

    enable_camera = st.toggle("Enable Camera", value=True)
    switch_camera = st.button("Switch Camera")
    show_face_mesh = st.checkbox("Show Face Mesh Overlay", value=True)
    show_3d_visualization = st.checkbox("Show 3D Head Pose", value=True)
    calibrate_button = st.button("Calibrate Head Pose")

    col1, col2 = st.columns(2)
    record_direction_button = col1.button("Record Looking Direction")
    reset_directions_button = col2.button("Reset Recorded Directions")
    generate_heatmap_button = col1.button("Generate a heatmap")
    generate_graphs_button = col2.button("Generate graphs")

    threshold_slider = st.slider("Zone Threshold (degrees)", 1, 90, 15)
    head_pose_manager.set_zone_threshold(threshold_slider)

    placeholder = st.empty()
    data_placeholder = st.empty()

    if enable_camera:
        camera_manager.start_camera()

        frame_count = 0
        last_calibrated_matrix = None

        while enable_camera:

            if frame_count % FRAME_SKIP*PLOT_SKIP == 0:
                frame_count = 0 # resetting the frame count

            frame_placeholder, plot_placeholder = placeholder.columns(2)
            heatmap_placeholder, graph_placeholder = data_placeholder.columns(2)

            try:
                frame = camera_manager.get_frame()
                frame_count += 1

                if frame_count % FRAME_SKIP == 0:

                    if switch_camera:
                        camera_manager.switch_to_next_camera()
                        switch_camera = False

                    calibrated_matrix = head_pose_manager.process_frame(frame)
                    if calibrated_matrix is not None:
                        last_calibrated_matrix = calibrated_matrix

                    if calibrate_button and last_calibrated_matrix is not None:
                        head_pose_manager.calibrate()
                        calibrate_button = False

                    if record_direction_button and last_calibrated_matrix is not None:
                        head_pose_manager.record_looking_direction()
                        record_direction_button = False

                    if reset_directions_button:
                        head_pose_manager.reset_recorded_looking_directions()
                        reset_directions_button = False

                    if generate_heatmap_button:
                        heatmap_generator.generate_heatmap(
                            heatmap_placeholder,
                            head_pose_manager.get_calibrated_recorded_directions(),
                            head_pose_manager.zone_threshold,
                        )
                        generate_heatmap_button = False

                    if generate_graphs_button:
                        graph_generator.plot_all_graphs(graph_placeholder)
                        st.write(len(graph_generator.time_series))
                        generate_graphs_button = False

                if show_face_mesh:
                    frame = head_pose_manager.estimator.draw(frame)

                frame_placeholder.image(frame, channels="RGB", use_container_width=True)

                if (
                    show_3d_visualization
                    and last_calibrated_matrix is not None
                    and frame_count % PLOT_SKIP == 0
                ):
                    visualizer.plot_head_pose(
                        last_calibrated_matrix,
                        head_pose_manager.get_calibrated_recorded_directions(),
                        head_pose_manager.zone_threshold,
                        plot_placeholder
                    )

                if head_pose_manager.is_looking_within_zone():
                    frame_placeholder.success("Looking inside the defined zone!")
                else:
                    frame_placeholder.warning("Looking outside the zone!")

            except RuntimeError as e:
                st.error(f"Camera Error: {e}")

        camera_manager.stop_camera()


if __name__ == "__main__":
    main()
