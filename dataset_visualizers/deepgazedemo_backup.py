import sys; sys.path.insert(0, "..")
import streamlit as st
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
import deepgaze_pytorch
import os

# Streamlit App Configuration
st.title("DeepGazeIII with I-DT and I-VT Fixation Detection")
import numpy as np

def calculate_velocity(x, y, time_interval):
    """Calculate point-to-point velocities for I-VT."""
    velocities = []
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        dy = y[i] - y[i - 1]
        velocity = np.sqrt(dx**2 + dy**2) / time_interval
        velocities.append(velocity)
    return velocities

def apply_ivt(x, y, time_interval, velocity_threshold, min_fixation_duration):
    """Apply the I-VT algorithm to detect fixations."""
    velocities = calculate_velocity(x, y, time_interval)
    labels = np.zeros(len(x))  # Initialize labels with 0s for saccades
    current_fixation = []
    
    for i in range(len(velocities)):
        if velocities[i] < velocity_threshold:
            current_fixation.append(i + 1)  # +1 because velocities array is shorter by 1 element
        else:
            if len(current_fixation) * time_interval >= min_fixation_duration:
                labels[current_fixation] = 1  # Label as fixation
            current_fixation = []
    return labels

def apply_idt(x, y, time_interval, dispersion_threshold, min_fixation_duration):
    """Apply the I-DT algorithm to detect fixations."""
    labels = np.zeros(len(x))  # Initialize labels with 0s for saccades
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and (max(x[i:j + 1]) - min(x[i:j + 1])) < dispersion_threshold and \
                            (max(y[i:j + 1]) - min(y[i:j + 1])) < dispersion_threshold:
            j += 1
        if (j - i) * time_interval >= min_fixation_duration:
            labels[i:j] = 1  # Label as fixation
        i = j
    return labels

def get_fixation(x, y, time_interval, velocity_threshold=1000, dispersion_threshold=25, min_fixation_duration=0.1):
    """
    Perform I-DT and I-VT analysis and label points as fixations or saccades.
    
    Parameters:
    - x: List or numpy array of x positions.
    - y: List or numpy array of y positions.
    - time_interval: Time interval between points in seconds.
    - velocity_threshold: Velocity threshold for I-VT.
    - dispersion_threshold: Dispersion threshold for I-DT.
    - min_fixation_duration: Minimum duration to be considered a fixation.
    
    Returns:
    - fixation_sections: List of fixation sections, each being a list of points.
    - average_x: List of averaged x positions for each fixation section.
    - average_y: List of averaged y positions for each fixation section.
    """
    # Apply both I-VT and I-DT to label fixations
    ivt_labels = apply_ivt(x, y, time_interval, velocity_threshold, min_fixation_duration)
    idt_labels = apply_idt(x, y, time_interval, dispersion_threshold, min_fixation_duration)

    # OR operation between I-VT and I-DT labels to detect final fixations
    fixation_labels = np.logical_or(ivt_labels, idt_labels).astype(int)

    # Identify fixation sections based on the combined labels
    fixation_sections = []
    current_fixation = []

    for i in range(len(fixation_labels)):
        if fixation_labels[i] == 1:
            current_fixation.append((x[i], y[i]))
        else:
            if current_fixation:
                fixation_sections.append(current_fixation)
                current_fixation = []
    # Append the last fixation section if it exists
    if current_fixation:
        fixation_sections.append(current_fixation)

    # Calculate average x and y positions for each fixation section
    average_x = [np.mean([point[0] for point in section]) for section in fixation_sections]
    average_y = [np.mean([point[1] for point in section]) for section in fixation_sections]

    return fixation_sections, average_x, average_y


# Load the pretrained DeepGazeIII model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(DEVICE)


# Set the paths for the stimuli and data folders
stimuli_folder = '../Datasets/MIT1003/ALLSTIMULI/'
data_folder = '../Datasets/MIT1003/DATA/'

# List all the image files in the stimuli folder
image_files = [f for f in os.listdir(stimuli_folder) if f.endswith('.jpeg') or f.endswith('.jpg')]
subject_folders = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]

# Create Streamlit selection boxes for image and subject
selected_image = st.selectbox('Select an Image File', image_files)
selected_subject = st.selectbox('Select a Subject Folder', subject_folders)

# Extract image filename without extension
stimulus_name = os.path.splitext(selected_image)[0]
image_path = os.path.join(stimuli_folder, selected_image)
data_path = os.path.join(data_folder, selected_subject, f'{stimulus_name}.mat')

# Load Stimulus Image
img = mpimg.imread(image_path)

# Load Scanpath Data for Selected Subject and Image
subject_data = scipy.io.loadmat(data_path)
points = subject_data[stimulus_name][0][0][4][0][0][2]

# Filter out points with negative values and flip x, y coordinates
filtered_points = points[(points[:, 0] >= 0) & (points[:, 1] >= 0)]
# print(len(filtered_points))
x, y = filtered_points[:, 0], filtered_points[:, 1]
# print(filtered_points)

# Parameters for fixation detection
time_interval = 1 / 240  # 240 Hz sampling rate
velocity_threshold = 1000  # I-VT velocity threshold (pixels/second)
dispersion_threshold = 25  # I-DT spatial dispersion threshold (pixels)
min_fixation_duration = 0.1  # Minimum fixation duration (seconds)


fixation_sections, fixation_history_x, fixation_history_y = get_fixation(x, y, time_interval)

# Plot and visualize I-VT, I-DT, and Combined Fixations
st.subheader("Fixation Detection Results")

# Combined Fixations Visualization
combined_fig, combined_ax = plt.subplots()
combined_ax.imshow(img)
combined_ax.plot(fixation_history_x, fixation_history_y, marker='o', color='red', linestyle='-', linewidth=1, markersize=5)  # Combined fixations
combined_ax.set_title("Combined Fixations (Red)")
combined_ax.set_xlim(0, img.shape[1])
combined_ax.set_ylim(img.shape[0], 0)
combined_ax.set_axis_off()
st.pyplot(combined_fig)

# Load precomputed centerbias for the MIT1003 dataset
centerbias_template = np.load('../centerbias_mit1003.npy')
centerbias = zoom(centerbias_template, (img.shape[0] / centerbias_template.shape[0], 
                                        img.shape[1] / centerbias_template.shape[1]), 
                                        order=0, mode='nearest')
centerbias -= logsumexp(centerbias)

# Select Starting Index for Prediction
start_idx = st.slider('Select starting index for fixation history', min_value=0, max_value=len(fixation_history_x) - 4, step=1)
end_idx = start_idx + 4

# Subset the selected fixation points for the model
selected_fixation_x = fixation_history_x[start_idx:end_idx]
selected_fixation_y = fixation_history_y[start_idx:end_idx]

# Convert inputs to tensors
image_tensor = torch.tensor([img.transpose(2, 0, 1)]).to(DEVICE)
centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)
x_hist_tensor = torch.tensor([selected_fixation_x]).to(DEVICE)
y_hist_tensor = torch.tensor([selected_fixation_y]).to(DEVICE)

# Perform the prediction using the selected fixations
with torch.no_grad():
    log_density_prediction = model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)

# Visualization of the Final Prediction
f, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
axs[0].imshow(img)
axs[0].plot(fixation_history_x[:end_idx], fixation_history_y[:end_idx], 'o-', color='red')
axs[0].scatter(fixation_history_x[end_idx-1], fixation_history_y[end_idx-1], 100, color='yellow')
axs[0].set_title('Fixation Scanpath')
axs[0].set_axis_off()
predicted_heatmap = log_density_prediction.detach().cpu().numpy()[0, 0]
axs[1].imshow(img, alpha=0.5)
heatmap = axs[1].imshow(predicted_heatmap, cmap='jet', alpha=0.6)
axs[1].plot(fixation_history_x[:end_idx], fixation_history_y[:end_idx], 'o-', color='red')
axs[1].scatter(fixation_history_x[end_idx-1], fixation_history_y[end_idx-1], 100, color='yellow')
axs[1].set_title('Predicted Heatmap')
axs[1].set_axis_off()
f.colorbar(heatmap, ax=axs[1])
st.pyplot(f)
