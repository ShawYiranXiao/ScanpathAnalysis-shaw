# Import necessary libraries
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

# Load stimulus image and raw gaze data
stimulus_suffix = 'jpeg'
stimulus_name = 'i1011319098'

# Load the image to use as the background for the plot
image_path = f'../Datasets/MIT1003/ALLSTIMULI/{stimulus_name}.{stimulus_suffix}'
img = mpimg.imread(image_path)

# Load the main data file (replace 'i1011319098' with the specific image filename as needed)
subject_data = scipy.io.loadmat(f'../Datasets/MIT1003/DATA/ajs/{stimulus_name}.mat')

# Extract the key data points from the loaded .mat structure
points = subject_data[stimulus_name][0][0][4][0][0][2]

# Filter out points where x or y values are below zero
filtered_points = points[(points[:, 0] >= 0) & (points[:, 1] >= 0)]

# Flip x and y coordinates as preferred by the user
x = filtered_points[:, 0]
y = filtered_points[:, 1]

# Sampling rate is 240 Hz (4.167 ms interval)
time_interval = 1 / 240  # in seconds

# Define parameters for fixation detection
velocity_threshold = 1000  # Velocity threshold in pixels/second for I-VT
dispersion_threshold = 25  # Spatial dispersion threshold in pixels for I-DT
min_fixation_duration = 0.1  # Minimum fixation duration in seconds

# Helper function: Calculate point-to-point velocities (I-VT)
def calculate_velocity(x, y, time_interval):
    velocities = []
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        dy = y[i] - y[i - 1]
        velocity = np.sqrt(dx**2 + dy**2) / time_interval
        velocities.append(velocity)
    return velocities

# Helper function: Apply I-VT algorithm
def apply_ivt(x, y, time_interval, velocity_threshold, min_fixation_duration):
    velocities = calculate_velocity(x, y, time_interval)
    fixations, saccades = [], []
    current_fixation = []
    
    for i in range(len(velocities)):
        if velocities[i] < velocity_threshold:
            current_fixation.append((x[i], y[i]))
        else:
            if len(current_fixation) > 0:
                if len(current_fixation) * time_interval >= min_fixation_duration:
                    fixations.append(np.array(current_fixation))
                else:
                    saccades.extend(current_fixation)
                current_fixation = []
    return fixations, saccades

# Helper function: Apply I-DT algorithm
def apply_idt(x, y, time_interval, dispersion_threshold, min_fixation_duration):
    fixations, saccades = [], []
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and max(x[i:j + 1]) - min(x[i:j + 1]) < dispersion_threshold and \
              max(y[i:j + 1]) - min(y[i:j + 1]) < dispersion_threshold:
            j += 1
        if (j - i) * time_interval >= min_fixation_duration:
            fixations.append(np.column_stack((x[i:j], y[i:j])))
        else:
            saccades.extend([(x[k], y[k]) for k in range(i, j)])
        i = j
    return fixations, saccades

# Apply I-VT and I-DT algorithms
ivt_fixations, ivt_saccades = apply_ivt(x, y, time_interval, velocity_threshold, min_fixation_duration)
idt_fixations, idt_saccades = apply_idt(x, y, time_interval, dispersion_threshold, min_fixation_duration)

# Plot raw gaze data, I-VT, and I-DT results
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# 1. Raw Gaze Data
axes[0].imshow(img)
axes[0].plot(x, y, marker='o', color='red', linewidth=2, markersize=5, label='Raw Gaze Path')  # Raw gaze path
axes[0].set_xlim(0, img.shape[1])
axes[0].set_ylim(img.shape[0], 0)
axes[0].set_title("Raw Gaze Data")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")

# 2. I-VT Fixations and Saccades
axes[1].imshow(img)
axes[1].plot(x, y, color='red', linewidth=1, linestyle='-', alpha=0.6)  # Underlying gaze path
# Plot I-VT fixations in blue
for fixation in ivt_fixations:
    axes[1].plot(fixation[:, 0], fixation[:, 1], marker='o', color='blue', linewidth=0, markersize=5, label='I-VT Fixation')
# Plot I-VT saccades in red
ivt_saccades = np.array(ivt_saccades)
if len(ivt_saccades) > 0:
    axes[1].plot(ivt_saccades[:, 0], ivt_saccades[:, 1], marker='x', color='green', linewidth=1, markersize=6, label='I-VT Saccade')
axes[1].set_xlim(0, img.shape[1])
axes[1].set_ylim(img.shape[0], 0)
axes[1].set_title("I-VT: Fixations (Blue) and Saccades (Red)")
axes[1].set_xlabel("X")
axes[1].set_ylabel("Y")

# 3. I-DT Fixations and Saccades
axes[2].imshow(img)
axes[2].plot(x, y, color='gray', linewidth=1, linestyle='-', alpha=0.6)  # Underlying gaze path
# Plot I-DT fixations in green
for fixation in idt_fixations:
    axes[2].plot(fixation[:, 0], fixation[:, 1], marker='o', color='blue', linewidth=0, markersize=5, label='I-DT Fixation')
# Plot I-DT saccades in red
idt_saccades = np.array(idt_saccades)
if len(idt_saccades) > 0:
    axes[2].plot(idt_saccades[:, 0], idt_saccades[:, 1], marker='x', color='red', linewidth=1, markersize=6, label='I-DT Saccade')
axes[2].set_xlim(0, img.shape[1])
axes[2].set_ylim(img.shape[0], 0)
axes[2].set_title("I-DT: Fixations (Green) and Saccades (Red)")
axes[2].set_xlabel("X")
axes[2].set_ylabel("Y")

# Show all plots
plt.tight_layout()
plt.show()
