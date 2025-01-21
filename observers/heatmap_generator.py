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
        self.data = []
        self.plane_distance = plane_distance

    def update(self, data):
        """
        Observer update method; receives new rotation_vector data and records it.
        """
        self.record_data(data["rotation_vector"])

    def record_data(self, rotation_vector):
        """
        Convert the rotation vector to a 3D direction and project it onto
        the plane at z = self.plane_distance. Store the resulting (x, y).
        """
        direction = R.from_rotvec(rotation_vector).as_matrix() @ np.array([0, 0, 1])

        if abs(direction[2]) < 1e-9:
            return

        scale_factor = self.plane_distance / direction[2]
        point_on_plane = direction * scale_factor

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
        rotation_to_direction = R.align_vectors([direction], [[0, 0, 1]])[0]

        # Generate points on the cone boundary
        num_points = 100
        azimuths = np.linspace(0, 2 * np.pi, num_points)
        cone_boundary_points = []

        for azimuth in azimuths:
            local_cone_point = R.from_euler(
                "yz", [zone_threshold, azimuth], degrees=False
            ).apply([0, 0, 1])

            global_cone_point = rotation_to_direction.apply(local_cone_point)

            if abs(global_cone_point[2]) < 1e-9:
                continue
            scale_factor = self.plane_distance / global_cone_point[2]
            x, y, _ = global_cone_point * scale_factor

            x = np.clip(x, -1.5, 1.5)
            y = np.clip(y, -1.5, 1.5)

            cone_boundary_points.append((x, y))

        return np.array(cone_boundary_points)

    def generate_heatmap(
        self,
        placeholder=None,
        recorded_directions=None,
        zone_threshold=np.radians(15),
        focus_areas=None,
    ):
        """
        Generate and save a heatmap of the looking directions, including optional zone overlays.

        Args:
            placeholder: Streamlit placeholder for display (optional).
            recorded_directions: List of 3D rotation matrices for recorded directions (optional).
            zone_threshold: Default angular threshold in radians for cones, if not specified per focus area.
            focus_areas: A list of dictionaries describing arbitrary focus cones. Example item:
                {
                    "direction": np.array([0, 0, 1]),   # The direction for the focus area
                    "label": "Forward",                 # Legend label
                    "color": "blue",                    # Overlay color
                    "threshold": np.radians(15)         # Override default threshold (optional)
                }
        """
        if not self.data:
            raise ValueError("No data available for generating heatmap.")

        points = np.array(self.data)

        points = points[(np.abs(points[:, 0]) <= 1.5) & (np.abs(points[:, 1]) <= 1.5)]

        heatmap, xedges, yedges = np.histogram2d(
            points[:, 0], points[:, 1], bins=100, range=[[-1.5, 1.5], [-1.5, 1.5]]
        )

        fig, ax = plt.subplots(figsize=(8, 8))
        c = ax.imshow(
            heatmap.T,
            origin="lower",
            extent=[-1.5, 1.5, -1.5, 1.5],
            cmap="hot",
            interpolation="nearest",
        )
        fig.colorbar(c, ax=ax, label="Density")

        ax.set_title("Looking Direction Heatmap")
        ax.set_xlabel("X (horizontal)")
        ax.set_ylabel("Y (vertical)")

        if focus_areas is not None:
            for idx, focus_area in enumerate(focus_areas):
                # Extract or set defaults
                direction = focus_area["direction"]
                label = focus_area.get("label", f"Focus {idx+1}")
                color = focus_area.get("color", "blue")
                threshold = focus_area.get("threshold", zone_threshold)

                projected_cone = self._project_cone_to_plane(direction, threshold)

                ax.fill(
                    projected_cone[:, 0],
                    projected_cone[:, 1],
                    color=color,
                    alpha=0.25,
                    label=label,
                )

        if recorded_directions is not None:
            for idx, recorded_matrix in enumerate(recorded_directions):
                forward_direction = recorded_matrix @ np.array([0, 0, 1])
                cone_boundary_2d = self._project_cone_to_plane(
                    forward_direction, zone_threshold
                )
                ax.fill(
                    cone_boundary_2d[:, 0],
                    cone_boundary_2d[:, 1],
                    color="magenta",
                    alpha=0.3,
                    label=f"Recorded Zone {idx+1}",
                )

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        plt.savefig(self.save_path)
        plt.close(fig)

        if placeholder:
            placeholder.pyplot(fig)
