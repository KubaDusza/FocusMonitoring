import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from collections import deque
from scipy.spatial.transform import Rotation as R
from observers.observer import Observer


class GraphGenerator(Observer):
    def __init__(self, sliding_window_size=100):
        self.time_series = []
        self.zone_series = []
        self.gaze_durations = []
        self.sliding_window = deque(maxlen=sliding_window_size)
        self.sliding_window_percentages = []

    def update(self, data):
        """
        Update the graphs based on the notification from the manager.
        """
        current_time = time.time()
        is_in_zone = data.get("is_looking_within_zone", False)

        self.time_series.append(current_time)
        self.zone_series.append(1 if is_in_zone else 0)

        if is_in_zone:

            if not self.gaze_durations or (
                len(self.zone_series) >= 2 and self.zone_series[-2] == 0
            ):
                self.gaze_durations.append(1)
            else:
                self.gaze_durations[-1] += 1

        # Update sliding window
        self.sliding_window.append(is_in_zone)
        self.sliding_window_percentages.append(np.mean(self.sliding_window) * 100)

    def _format_time_series(self):
        """
        Convert Unix timestamps to human-readable time strings.
        """
        return [
            datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            for ts in self.time_series
        ]

    def plot_all_graphs(self, placeholder=None):
        if placeholder:
            col1, col2 = placeholder.columns(2)
            self.plot_time_spent(col1)
            self.plot_average_gaze_duration(col1)
            self.plot_zone_adherence(col2)
            self.plot_sliding_window_adherence(col2)

    def plot_time_spent(self, placeholder=None):
        """
        Plot the cumulative percentage of time spent in the zone.
        """
        if not self.time_series:
            return
        if len(self.time_series) < 5:
            return

        plt.figure(figsize=(6, 3))
        percentages = (
            np.cumsum(self.zone_series) / np.arange(1, len(self.zone_series) + 1) * 100
        )
        time_labels = self._format_time_series()

        plt.plot(self.time_series, percentages, label="Time Spent in Zone (%)")
        plt.xlabel("Time (HH:MM:SS)")
        plt.ylim(0, 100)

        if len(self.time_series) >= 5:
            step = max(len(self.time_series) // 5, 1)
            plt.xticks(self.time_series[::step], time_labels[::step], rotation=45)

        plt.ylabel("Percentage (%)")
        plt.title("Cumulative Time Spent in Zone")
        plt.grid()
        plt.savefig("time_spent_in_zone.png")
        if placeholder:
            placeholder.pyplot(plt)
        plt.close()

    def plot_zone_adherence(self, placeholder=None):
        """
        Plot zone adherence over time as a step function.
        """
        if not self.time_series:
            return
        if len(self.time_series) < 2:
            return

        plt.figure(figsize=(6, 3))
        time_labels = self._format_time_series()

        plt.step(
            self.time_series, self.zone_series, where="post", label="Zone Adherence"
        )
        plt.xlabel("Time (HH:MM:SS)")
        # ADDED: legend
        plt.legend()  # ADDED

        step = max(len(self.time_series) // 5, 1)
        plt.xticks(self.time_series[::step], time_labels[::step], rotation=45)

        plt.ylabel("In Zone (1) / Out of Zone (0)")
        plt.title("Zone Adherence Over Time")
        plt.grid()
        plt.savefig("zone_adherence_over_time.png")
        if placeholder:
            placeholder.pyplot(plt)
        plt.close()

    def plot_average_gaze_duration(self, placeholder=None):
        """
        Plot the average gaze duration over time.
        """
        if not self.gaze_durations:
            return
        if len(self.gaze_durations) < 1:
            return

        plt.figure(figsize=(6, 3))
        avg_gaze_durations = np.cumsum(self.gaze_durations) / np.arange(
            1, len(self.gaze_durations) + 1
        )

        plt.plot(avg_gaze_durations, label="Avg Gaze Duration (s)")
        plt.xlabel("Number of Gaze Events")
        plt.ylabel("Duration (s)")
        plt.title("Average Gaze Duration Over Time")
        plt.grid()

        plt.legend()
        plt.savefig("average_gaze_duration.png")
        if placeholder:
            placeholder.pyplot(plt)
        plt.close()

    def plot_sliding_window_adherence(self, placeholder=None):
        """
        Plot the adherence percentage over a sliding window.
        """
        if not self.sliding_window_percentages:
            return
        if len(self.sliding_window_percentages) < 1:
            return

        plt.figure(figsize=(6, 3))
        plt.plot(self.sliding_window_percentages, label="Zone Adherence (%)")
        plt.xlabel("Sliding Window Events")

        plt.ylim(0, 100)
        plt.legend()

        plt.ylabel("Adherence Percentage (%)")
        plt.title("Zone Adherence Percentage (Sliding Window)")
        plt.grid()
        plt.savefig("sliding_window_adherence.png")
        if placeholder:
            placeholder.pyplot(plt)
        plt.close()
