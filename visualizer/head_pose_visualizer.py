import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


class HeadPoseVisualizer:
    """
    A class to visualize head poses, recorded directions, and threshold cones.
    """

    def __init__(self, visualization_rotation = None):
        if visualization_rotation is None:
            self.visualization_rotation = np.eye(3)  # Global visualization rotation
        else:
            self.set_visualization_rotation(visualization_rotation)

    def set_visualization_rotation(self, euler_angles_degrees):
        """
        Set the rotation of the entire visualization.

        Args:
            euler_angles_degrees: List or array of [x, y, z] rotation angles in degrees.
        """
        from scipy.spatial.transform import Rotation as R
        self.visualization_rotation = R.from_euler('xyz', euler_angles_degrees, degrees=True).as_matrix()

    def plot_head_pose(self, calibrated_rotation_matrix, recorded_directions, zone_threshold):
        """
        Plot the head pose, recorded directions, and cones.

        Args:
            calibrated_rotation_matrix: Current calibrated head pose rotation matrix.
            recorded_directions: List of rotation matrices for recorded directions.
            zone_threshold: Threshold angle (in radians) for the cones.
            plot_placeholder: If provided, the figure will be plotted there (e.g., in Streamlit).
        """

        calibrated_rotation_matrix = self.visualization_rotation @ calibrated_rotation_matrix
        recorded_directions = [self.visualization_rotation @ direction for direction in recorded_directions]

        # Create the 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Draw components
        self._draw_head(ax, calibrated_rotation_matrix)
        self._draw_smile(ax, calibrated_rotation_matrix)
        self._draw_recorded_directions(ax, recorded_directions, zone_threshold)

        # Set plot limits and labels
        self._set_plot_bounds(ax)

        return fig

    def _draw_head(self, ax, rotation_matrix):
        """
        Draw the 3D cube representing the head.
        """
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
        rotated_vertices = cube_vertices @ rotation_matrix.T

        faces = [
            [rotated_vertices[j] for j in [0, 1, 2, 3]],
            [rotated_vertices[j] for j in [4, 5, 6, 7]],
            [rotated_vertices[j] for j in [0, 1, 5, 4]],
            [rotated_vertices[j] for j in [2, 3, 7, 6]],
            [rotated_vertices[j] for j in [1, 2, 6, 5]],
            [rotated_vertices[j] for j in [4, 7, 3, 0]]
        ]

        # Draw the cube
        face_colors = ['cyan'] * 6
        face_colors[1] = 'red'  # Highlight the front face
        for idx, face in enumerate(faces):
            ax.add_collection3d(
                Poly3DCollection([face], facecolors=face_colors[idx], linewidths=1, edgecolors='k', alpha=.75)
            )

    def _draw_smile(self, ax, rotation_matrix):
        """
        Draw the smiley face on the front of the cube.
        """
        # Compute the front face center
        front_face_center = np.mean([
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5]
        ], axis=0) @ rotation_matrix.T

        # Smiley face components
        left_eye = front_face_center + np.array([-0.2, 0.2, 0]) @ rotation_matrix.T
        right_eye = front_face_center + np.array([0.2, 0.2, 0]) @ rotation_matrix.T
        mouth_points = np.array([
            [0.2, -0.2, 0],
            [0, -0.3, 0],
            [-0.2, -0.2, 0]
        ]) @ rotation_matrix.T + front_face_center

        # Draw the smiley face
        ax.scatter(*left_eye, color='black', s=100)
        ax.scatter(*right_eye, color='black', s=100)
        ax.add_collection3d(Line3DCollection([mouth_points], color='black', linewidths=2))

    def _draw_recorded_directions(self, ax, recorded_directions, zone_threshold):
        """
        Draw recorded directions as arrows and cones.
        """
        for recorded_matrix in recorded_directions:
            direction_vector = recorded_matrix @ np.array([0, 0, 1])

            # Draw arrow
            ax.quiver(0, 0, 0, direction_vector[0], direction_vector[1], direction_vector[2],
                      color='orange', arrow_length_ratio=0.1)

            # Draw cone
            self._draw_cone(ax, direction_vector, zone_threshold)

    def _draw_cone(self, ax, direction, cone_angle):
        """
        Draw a cone to visualize the threshold zone.
        """
        from scipy.spatial.transform import Rotation as R
        cone_height = 1.0
        theta = np.linspace(0, 2 * np.pi, 30)
        circle_radius = np.tan(cone_angle) * cone_height

        circle_x = circle_radius * np.cos(theta)
        circle_y = circle_radius * np.sin(theta)
        circle_z = np.ones_like(circle_x) * cone_height

        circle_points = np.stack([circle_x, circle_y, circle_z], axis=1)
        rotation = R.align_vectors([direction], [[0, 0, 1]])[0].as_matrix()
        rotated_points = circle_points @ rotation.T

        ax.plot_trisurf(rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2], color='orange', alpha=0.3)

    def _set_plot_bounds(self, ax):
        """
        Set the 3D plot bounds and labels.
        """
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
