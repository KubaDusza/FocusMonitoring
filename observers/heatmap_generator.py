import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from observers.observer import Observer


class HeatmapGenerator(Observer):
    def __init__(self, save_path="heatmap.png", plane_distance=1.0):
        super().__init__()
        """
        Initialize the heatmap generator.

        Args:
            save_path (str): Path to save the heatmap image.
            plane_distance (float): Distance of the imaginary plane in front of the origin.
        """
        self.save_path = save_path
        self.data = []  # Store rotation vectors over time
        self.plane_distance = plane_distance

    def update(self, data):
        self.record_data(data["rotation_vector"])

    def record_data(self, rotation_vector):
        # Convert rotation vector to a 3D direction vector
        direction = R.from_rotvec(rotation_vector).as_matrix() @ np.array([0, 0, 1])

        # Scale the direction vector to intersect the plane
        scale_factor = self.plane_distance / direction[2]  # Ensure z = plane_distance
        point_on_plane = direction * scale_factor

        # Store the (x, y) components
        self.data.append(point_on_plane[:2])

    def _project_cone_to_plane(self, direction, zone_threshold):
        """
        Project the threshold cone around a given direction onto a 2D plane.

        Args:
            direction (np.ndarray): 3D unit vector representing the direction.
            zone_threshold (float): Cone angle (in radians).

        Returns:
            np.ndarray: Array of 2D points representing the projected cone boundary.
        """
        from scipy.spatial.transform import Rotation as R

        # Align the cone axis to the direction vector
        rotation_to_direction = R.align_vectors([direction], [[0, 0, 1]])[0]

        # Generate points on the cone boundary
        num_points = 100
        azimuths = np.linspace(0, 2 * np.pi, num_points)
        cone_boundary_points = []

        for azimuth in azimuths:
            # Generate a point on the cone surface in local space
            local_cone_point = R.from_euler('yz', [zone_threshold, azimuth], degrees=False).apply([0, 0, 1])

            # Rotate the local point into the global space
            global_cone_point = rotation_to_direction.apply(local_cone_point)

            # Project onto the plane
            scale_factor = self.plane_distance / global_cone_point[2]  # Ensure z = plane_distance
            x, y, _ = global_cone_point * scale_factor

            x = min(max(x, -1.5), 1.5)
            y = min(max(y, -1.5), 1.5)

            cone_boundary_points.append((x,y))

        return np.array(cone_boundary_points)

    def generate_heatmap(self, placeholder=None, recorded_directions=None, zone_threshold=np.radians(15)):
        """
        Generate and save a heatmap of the looking directions, including zone overlays.

        Args:
            placeholder: Streamlit placeholder for display (optional).
            recorded_directions: List of 3D rotation matrices for recorded directions.
            zone_threshold: Angular threshold in radians for the zones.
        """
        if not self.data:
            raise ValueError("No data available for generating heatmap.")

        # Convert the data to a numpy array for processing
        points = np.array(self.data)

        # Filter points within the range [-1.5, 1.5]
        points = points[(np.abs(points[:, 0]) <= 1.5) & (np.abs(points[:, 1]) <= 1.5)]

        # Create a 2D histogram
        heatmap, xedges, yedges = np.histogram2d(
            points[:, 0], points[:, 1], bins=100, range=[[-1.5, 1.5], [-1.5, 1.5]]
        )

        # Plot the heatmap
        fig, ax = plt.subplots(figsize=(8, 8))  # Increased figure size for better visualization
        c = ax.imshow(
            heatmap.T, origin="lower", extent=[-1.5, 1.5, -1.5, 1.5], cmap="hot", interpolation="nearest"
        )
        fig.colorbar(c, ax=ax, label="Density")

        # Add labels and title
        ax.set_title("Looking Direction Heatmap")
        ax.set_xlabel("X (horizontal)")
        ax.set_ylabel("Y (vertical)")

        # Overlay recorded zones
        if recorded_directions is not None:
            for recorded_matrix in recorded_directions:
                # Get the forward direction
                forward_direction = recorded_matrix @ np.array([0, 0, 1])

                # Project the cone to the plane
                cone_boundary_2d = self._project_cone_to_plane(forward_direction, zone_threshold)

                # Draw the projected cone boundary
                ax.fill(cone_boundary_2d[:, 0], cone_boundary_2d[:, 1], color='blue', alpha=0.3, label="Zone")

        # Save the heatmap to a file
        plt.savefig(self.save_path)

        plt.close(fig)  # Close the figure to free resources

        # Display the heatmap in Streamlit if a placeholder is provided
        if placeholder:
            placeholder.pyplot(fig)
