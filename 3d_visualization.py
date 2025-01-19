
import streamlit as st
from core.camera_manager import CameraManager
from estimators.mediapipe_head_pose_estimator import MediaPipeHeadPoseEstimator

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R


def setup():
    if "camera_manager" not in st.session_state:
        st.session_state.camera_manager = CameraManager(camera_index=1)

    if "head_pose_estimator" not in st.session_state:
        st.session_state.head_pose_estimator = MediaPipeHeadPoseEstimator()
        st.session_state.head_pose_estimator.initialize()

    if "axis_corr" not in st.session_state:
        st.session_state.axis_corr = {"x": -90, "y": 0, "z": 180}


def calibrate_axis_corrections(rotation_vector):
    """
    Calibrate the axis corrections based on the current head pose.
    Adjusts the correction values to make the current pose the neutral pose.
    """
    current_rotation = R.from_rotvec(rotation_vector[:3]).as_euler('xyz', degrees=True)
    # Offset from the default neutral pose
    st.session_state.axis_corr["x"] -= current_rotation[0]
    st.session_state.axis_corr["y"] -= current_rotation[1]
    st.session_state.axis_corr["z"] -= current_rotation[2]

    st.success("Calibration complete! Current pose set as neutral.")


def plot_3d_pose(rotation_vector, plot_placeholder, colored_face_index=0, x_axis_corr=-90, y_axis_corr=0, z_axis_corr=180):
    """
    Plots a 3D pose cube with one face colored differently and a smiley face on it.

    Args:
        rotation_vector: The rotation vector representing the pose.
        plot_placeholder: Streamlit placeholder for displaying the plot.
        colored_face_index: Index of the face to color differently (0 to 5).
    """
    # Convert rotation vector to rotation matrix
    rotation = R.from_rotvec(rotation_vector[:3])
    rotation_matrix = rotation.as_matrix()

    # Apply the axis corrections
    x_axis_correction = R.from_euler('x', x_axis_corr, degrees=True).as_matrix()
    y_axis_correction = R.from_euler('y', y_axis_corr, degrees=True).as_matrix()
    z_axis_correction = R.from_euler('z', z_axis_corr, degrees=True).as_matrix()
    corrected_rotation_matrix = y_axis_correction @ z_axis_correction @ x_axis_correction @ rotation_matrix

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

    # Apply the corrected rotation to the cube vertices
    rotated_vertices = cube_vertices @ corrected_rotation_matrix.T

    # Define the 6 faces of the cube
    faces = [
        [rotated_vertices[j] for j in [0, 1, 2, 3]],  # Front
        [rotated_vertices[j] for j in [4, 5, 6, 7]],  # Back (target face for smiley)
        [rotated_vertices[j] for j in [0, 1, 5, 4]],  # Bottom
        [rotated_vertices[j] for j in [2, 3, 7, 6]],  # Top
        [rotated_vertices[j] for j in [1, 2, 6, 5]],  # Right
        [rotated_vertices[j] for j in [4, 7, 3, 0]]   # Left
    ]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Colors for the faces
    face_colors = ['cyan'] * 6
    face_colors[colored_face_index] = 'red'

    for idx, face in enumerate(faces):
        ax.add_collection3d(
            Poly3DCollection([face], facecolors=face_colors[idx], linewidths=1, edgecolors='k', alpha=.75))

    # Set plot limits and labels
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plot_placeholder.pyplot(fig)


def main():
    setup()
    st.title("Head Pose Estimation with 3D Visualization")

    camera_manager = st.session_state.camera_manager
    head_pose_estimator = st.session_state.head_pose_estimator

    # Streamlit Controls
    enable_camera = st.toggle("Enable Camera", value=True)
    show_face_mesh = st.checkbox("Show Face Mesh Overlay", value=True)
    show_3d_visualization = st.checkbox("Show 3D Head Pose", value=True)
    switch_camera = st.button("Switch Camera")

    # New Calibration Button
    calibrate_button = st.button("Calibrate Head Pose")

    if switch_camera:
        camera_manager.switch_to_next_camera()

    placeholder = st.empty()
    data_placeholder = st.empty()

    if enable_camera:
        camera_manager.start_camera()

        while enable_camera:
            frame_placeholder, plot_placeholder = placeholder.columns(2)
            try:
                frame = camera_manager.get_frame()
                head_pose_data = head_pose_estimator.estimate(frame)

                # If calibration button is pressed, adjust corrections based on current pose
                if calibrate_button and head_pose_data["rotation_vector"] is not None:
                    calibrate_axis_corrections(head_pose_data["rotation_vector"].flatten())
                    calibrate_button = False

                # Draw face mesh overlay
                if show_face_mesh:
                    frame = head_pose_estimator.draw(frame)
                frame_placeholder.image(frame, channels="RGB", use_container_width=True)

                # 3D Visualization
                if show_3d_visualization and head_pose_data["rotation_vector"] is not None:
                    plot_3d_pose(
                        head_pose_data["rotation_vector"].flatten(),
                        plot_placeholder,
                        x_axis_corr=st.session_state.axis_corr["x"],
                        y_axis_corr=st.session_state.axis_corr["y"],
                        z_axis_corr=st.session_state.axis_corr["z"],
                    )

            except RuntimeError as e:
                st.error(f"Camera Error: {e}")

        camera_manager.stop_camera()


if __name__ == "__main__":
    main()
