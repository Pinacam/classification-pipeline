"""
Project I - On-Device Classification
EMA Filtering Graphs

Description:
This program reads the CSV files containing raw and EMA-filtered
sensor readings collected from the ESP32.

It compares the raw IR readings with the filtered IR readings
for four lighting conditions: shade, LED, fluorescent, and sun.

The graphs help show how the EMA filter reduces noise and
smooths sudden changes in sensor readings.

Each graph is saved as a PDF in the figures/generated folder.
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Find the project folders
project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data"
figure_folder = project_folder / "figures" / "generated"

# Create the figures folder if it doesn't exist
figure_folder.mkdir(parents=True, exist_ok=True)

# Our four lighting conditions
conditions = ["shade", "led", "fluorescent", "sun"]

# Create one graph for each lighting condition
for condition in conditions:

    # Read the CSV measurements
    file_path = data_folder / f"ema_{condition}.csv"
    data = pd.read_csv(file_path)

    # Use the recorded sample order for the x-axis
    samples = range(1, len(data) + 1)

    # Plot raw and filtered IR readings
    plt.figure(figsize=(9, 4.5))

    plt.plot(
        samples,
        data["raw_ir"],
        label="Raw IR",
        color="lightgray",
        marker=".",
        linewidth=1.5
    )

    plt.plot(
        samples,
        data["filtered_ir"],
        label="EMA-filtered IR",
        color="blue",
        linewidth=2
    )

    plt.title(f"Raw vs. EMA-Filtered IR — {condition.capitalize()}")
    plt.xlabel("Sample Number")
    plt.ylabel("IR Sensor Reading")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save the graph
    output_file = figure_folder / f"ema_{condition}.pdf"
    plt.savefig(output_file)

    plt.show()

    print(f"Saved: {output_file}")