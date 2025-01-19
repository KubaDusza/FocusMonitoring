import streamlit as st
from core.camera_manager import CameraManager
from estimators.mediapipe_head_pose_estimator import MediaPipeHeadPoseEstimator
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


def setup():
    if "camera_manager" not in st.session_state:
        st.session_state.camera_manager = CameraManager(camera_index=1)

    if "head_pose_estimator" not in st.session_state:
        st.session_state.head_pose_estimator = MediaPipeHeadPoseEstimator()
        st.session_state.head_pose_estimator.initialize()

    if "calibration_matrix" not in st.session_state:
        st.session_state.calibration_matrix = np.eye(3)

    if "recorded_directions" not in st.session_state:
        st.session_state.recorded_directions = []

    if "zone_threshold" not in st.session_state:
        st.session_state.zone_threshold = np.radians(15)


def calibrate_axis_corrections(rotation_vector):
    rotation_matrix = R.from_rotvec(rotation_vector).as_matrix()
    st.session_state.calibration_matrix = np.linalg.inv(rotation_matrix)

    st.success("Calibration complete! Current pose set as neutral.")


def plot_3d_pose(rotation_vector, plot_placeholder, colored_face_index=1):
    rotation_matrix = R.from_rotvec(rotation_vector).as_matrix()
    corrected_rotation_matrix = st.session_state.calibration_matrix @ rotation_matrix

    # Apply additional corrections for axis alignment
    x_axis_correction = R.from_euler('x', 0, degrees=True).as_matrix()
    y_axis_correction = R.from_euler('y', 0, degrees=True).as_matrix()
    corrected_rotation_matrix = y_axis_correction @ x_axis_correction @ corrected_rotation_matrix

    # Define a cube centered at the origin
    cube_vertices = np.array([
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5]
    ])

    rotated_vertices = cube_vertices @ corrected_rotation_matrix.T

    faces = [
        [rotated_vertices[j] for j in [0, 1, 2, 3]],
        [rotated_vertices[j] for j in [4, 5, 6, 7]],
        [rotated_vertices[j] for j in [0, 1, 5, 4]],
        [rotated_vertices[j] for j in [2, 3, 7, 6]],
        [rotated_vertices[j] for j in [1, 2, 6, 5]],
        [rotated_vertices[j] for j in [4, 7, 3, 0]]
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    face_colors = ['cyan'] * 6
    face_colors[colored_face_index] = 'red'

    for idx, face in enumerate(faces):
        ax.add_collection3d(
            Poly3DCollection([face], facecolors=face_colors[idx], linewidths=1, edgecolors='k', alpha=.75))

    # Draw the smiley face
    #if colored_face_index == 1:
    face_center = np.mean(faces[colored_face_index], axis=0)
    left_eye = face_center + np.array([-0.2, 0.2, 0]) @ corrected_rotation_matrix.T
    right_eye = face_center + np.array([0.2, 0.2, 0]) @ corrected_rotation_matrix.T
    mouth_points = np.array([
        [0.2, -0.2, 0],
        [0, -0.3, 0],
        [-0.2, -0.2, 0]
    ]) @ corrected_rotation_matrix.T + face_center

    ax.scatter(*left_eye, color='black', s=100)
    ax.scatter(*right_eye, color='black', s=100)
    ax.add_collection3d(Line3DCollection([mouth_points], color='black', linewidths=2))

        # Apply additional corrections for axis alignment
    #x_axis_correction = R.from_euler('x', 0, degrees=True).as_matrix()
    #y_axis_correction = R.from_euler('y', 0, degrees=True).as_matrix()
    #corrected_rotation_matrix = y_axis_correction @ x_axis_correction @ corrected_rotation_matrix

    # ✅ Draw Recorded Directions as Arrows
    for recorded_vector in st.session_state.recorded_directions:
        recorded_matrix = R.from_rotvec(recorded_vector).as_matrix() @ st.session_state.calibration_matrix
        recorded_direction = recorded_matrix @ np.array([0, 0, 1])

        # Draw an arrow for each recorded direction
        ax.quiver(0, 0, 0, recorded_direction[0], recorded_direction[1], recorded_direction[2],
                  color='orange', arrow_length_ratio=0.1)

    # Set plot limits and labels
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plot_placeholder.pyplot(fig)


def record_looking_direction(rotation_vector):

    #rotation_vector = rotation_vector @ st.session_state.calibration_matrix

    """Records a new looking direction"""
    if len(st.session_state.recorded_directions) < 4:
        st.session_state.recorded_directions.append(rotation_vector)
        st.success(f"Direction {len(st.session_state.recorded_directions)} recorded!")
    else:
        st.warning("You have already recorded 4 directions.")


def is_looking_within_zone(rotation_vector):
    #rotation_vector = rotation_vector @ st.session_state.calibration_matrix

    """Check if the current head pose is inside the defined zone"""
    current_direction = R.from_rotvec(rotation_vector.flatten()).as_matrix() @ np.array([0, 0, 1])

    for recorded_vector in st.session_state.recorded_directions:
        recorded_direction = R.from_rotvec(recorded_vector).as_matrix() @ np.array([0, 0, 1])

        # Compute angle between vectors
        angle = np.arccos(np.clip(np.dot(current_direction, recorded_direction), -1.0, 1.0))

        if angle < st.session_state.zone_threshold:
            return True
    return False


def main():
    setup()
    st.title("Head Pose Estimation with Zone Tracking")
    st.write(st.session_state.calibration_matrix)

    camera_manager = st.session_state.camera_manager
    head_pose_estimator = st.session_state.head_pose_estimator

    enable_camera = st.toggle("Enable Camera", value=True)
    show_face_mesh = st.checkbox("Show Face Mesh Overlay", value=True)
    show_3d_visualization = st.checkbox("Show 3D Head Pose", value=True)
    calibrate_button = st.button("Calibrate Head Pose")
    record_direction_button = st.button("Record Looking Direction")

    placeholder = st.empty()

    if enable_camera:
        camera_manager.start_camera()

        while enable_camera:
            frame_placeholder, plot_placeholder = placeholder.columns(2)

            try:
                frame = camera_manager.get_frame()
                head_pose_data = head_pose_estimator.estimate(frame)

                if calibrate_button and head_pose_data["rotation_vector"] is not None:
                    calibrate_axis_corrections(head_pose_data["rotation_vector"].flatten())
                    calibrate_button = False

                if record_direction_button and head_pose_data["rotation_vector"] is not None:
                    record_looking_direction(head_pose_data["rotation_vector"].flatten())
                    record_direction_button = False

                if show_face_mesh:
                    frame = head_pose_estimator.draw(frame)
                frame_placeholder.image(frame, channels="RGB", use_container_width=True)

                if show_3d_visualization and head_pose_data["rotation_vector"] is not None:
                    plot_3d_pose(head_pose_data["rotation_vector"].flatten(), plot_placeholder)

                # Zone Check
                if is_looking_within_zone(head_pose_data["rotation_vector"]):
                    frame_placeholder.success("Looking inside the defined zone!")
                else:
                    frame_placeholder.warning("Looking outside the zone!")

            except RuntimeError as e:
                st.error(f"Camera Error: {e}")

        camera_manager.stop_camera()


if __name__ == "__main__":
    main()
