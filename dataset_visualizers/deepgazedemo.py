import sys; sys.path.insert(0, "..")
import streamlit as st
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.image as mpimg
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
import deepgaze_pytorch

# Set the paths for the stimuli, data, and affiliated image folders
stimuli_folder = '../Datasets/MIT1003/ALLSTIMULI/'
data_folder = '../Datasets/MIT1003/DATA/'
fixation_folder = '../Datasets/MIT1003/ALLFIXATIONMAPS/'

# Load the pretrained DeepGazeIII model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(DEVICE)


# Parameters for fixation detection
time_interval = 1 / 240  # 240 Hz sampling rate
velocity_threshold = 1000  # I-VT velocity threshold (pixels/second)
dispersion_threshold = 25  # I-DT spatial dispersion threshold (pixels)
min_fixation_duration = 0.1  # Minimum fixation duration (seconds)



def get_fixation(x, y, time_interval, velocity_threshold=1000, dispersion_threshold=25, min_fixation_duration=0.1):
    """Run both I-DT and I-VT and return fixation sections and their average positions."""
    def calculate_velocity(x, y, time_interval):
        velocities = [np.sqrt((x[i] - x[i - 1])**2 + (y[i] - y[i - 1])**2) / time_interval for i in range(1, len(x))]
        return velocities

    def apply_ivt(x, y, time_interval, velocity_threshold, min_fixation_duration):
        velocities = calculate_velocity(x, y, time_interval)
        labels = np.zeros(len(x))  # 0 = Saccade, 1 = Fixation
        current_fixation = []

        for i in range(len(velocities)):
            if velocities[i] < velocity_threshold:
                current_fixation.append(i + 1)  # +1 due to velocities being shorter
            else:
                if len(current_fixation) * time_interval >= min_fixation_duration:
                    labels[current_fixation] = 1
                current_fixation = []
        return labels

    def apply_idt(x, y, time_interval, dispersion_threshold, min_fixation_duration):
        labels = np.zeros(len(x))
        i = 0
        while i < len(x):
            j = i
            while j < len(x) and (max(x[i:j + 1]) - min(x[i:j + 1])) < dispersion_threshold and (max(y[i:j + 1]) - min(y[i:j + 1])) < dispersion_threshold:
                j += 1
            if (j - i) * time_interval >= min_fixation_duration:
                labels[i:j] = 1
            i = j
        return labels

    # Apply both algorithms and combine using OR operation
    ivt_labels = apply_ivt(x, y, time_interval, velocity_threshold, min_fixation_duration)
    idt_labels = apply_idt(x, y, time_interval, dispersion_threshold, min_fixation_duration)
    fixation_labels = np.logical_and(ivt_labels, idt_labels).astype(int)

    # Extract fixation sections
    fixation_sections, avg_x, avg_y = [], [], []
    current_fixation = []

    for i, label in enumerate(fixation_labels):
        if label == 1:
            current_fixation.append((x[i], y[i]))
        else:
            if current_fixation:
                fixation_sections.append(current_fixation)
                avg_x.append(np.mean([p[0] for p in current_fixation]))
                avg_y.append(np.mean([p[1] for p in current_fixation]))
                current_fixation = []

    if current_fixation:
        fixation_sections.append(current_fixation)
        avg_x.append(np.mean([p[0] for p in current_fixation]))
        avg_y.append(np.mean([p[1] for p in current_fixation]))

    return fixation_sections, avg_x, avg_y

# List all the image files in the stimuli folder
image_files = [f for f in os.listdir(stimuli_folder) if f.endswith('.jpeg') or f.endswith('.jpg')]
subject_folders = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]

# Create Streamlit selection boxes for image and multiple subjects
selected_image = st.selectbox('Select an Image File', image_files)

# Add a button to select all subjects
select_all = st.button('Select All Subjects')

# Automatically select all subjects if the button is clicked
if select_all:
    selected_subjects = st.multiselect('Select Subject Folders (multiple selection)', subject_folders, default=subject_folders)
else:
    selected_subjects = st.multiselect('Select Subject Folders (multiple selection)', subject_folders)

# Additional selection box for the subject to be used in DeepGazeIII
deepgaze_subject = st.selectbox('Select a Subject for DeepGazeIII Prediction', selected_subjects)

# Extract image filename without extension
stimulus_name = os.path.splitext(selected_image)[0]

# Set up the paths for the main image and fixation maps
image_path = os.path.join(stimuli_folder, selected_image)
fixmap_path = os.path.join(fixation_folder, f'{stimulus_name}_fixMap.jpg')
fixpts_path = os.path.join(fixation_folder, f'{stimulus_name}_fixPts.jpg')
# Checkboxes to select whether to display affiliated images
display_fixmap = st.checkbox('Display Fixation Map')
display_fixpts = st.checkbox('Display Fixation Points')

if os.path.exists(image_path):
    # Load the image as background
    img = mpimg.imread(image_path)

    # Load the fixation map and points if selected
    fixmap_img = mpimg.imread(fixmap_path) if display_fixmap and os.path.exists(fixmap_path) else None
    fixpts_img = mpimg.imread(fixpts_path) if display_fixpts and os.path.exists(fixpts_path) else None

    # Create a figure and display the image as the background
    fig, ax = plt.subplots()
    ax.imshow(img)

    # Overlay the fixation map if selected
    if fixmap_img is not None:
        ax.imshow(fixmap_img, cmap='jet', alpha=0.4)  # Semi-transparent overlay

    # Overlay the fixation points if selected
    if fixpts_img is not None:
        ax.imshow(fixpts_img, cmap='gray', alpha=0.6)

    # Display data for each selected subject with a distinct color
    colors = plt.cm.get_cmap('hsv', len(selected_subjects))  # Generate distinct colors for each subject

    for idx, subject in enumerate(selected_subjects):
        # Set up the data path for each selected subject
        data_path = os.path.join(data_folder, subject, f'{stimulus_name}.mat')

        if os.path.exists(data_path):
            # Load the subject data using scipy.io.loadmat
            subject_data = scipy.io.loadmat(data_path)

            # Extract the key data points from the loaded .mat structure
            points = subject_data[stimulus_name][0][0][4][0][0][2]

            # Filter out points where x or y values are below zero
            filtered_points = points[(points[:, 0] >= 0) & (points[:, 1] >= 0)]

            # Flip x and y coordinates as per your preference
            x = filtered_points[:, 0]
            y = filtered_points[:, 1]

            fixation_sections, x, y = get_fixation(x, y, time_interval)

            # Overlay the polyline for each subject using distinct colors
            ax.plot(x, y, marker='o', color=colors(idx), linewidth=0.5, markersize=2, label=f'Subject {subject}')
                # Display the final plot using Streamlit
    st.pyplot(fig)


# Load and display the image with overlaid data if selections are made
if os.path.exists(image_path) and selected_subjects:
    # Load the image as background and ensure it has 3 channels
    img = mpimg.imread(image_path)
    if img.shape[-1] == 4:  # Drop alpha channel if present
        img = img[:, :, :3]

    # Display the image and create the necessary visualizations
    fig, ax = plt.subplots()
    ax.imshow(img)

    # DeepGazeIII prediction visualization using the selected subject
    if deepgaze_subject:
        # Use selected subject's data for DeepGazeIII prediction
        deepgaze_data_path = os.path.join(data_folder, deepgaze_subject, f'{stimulus_name}.mat')
        if os.path.exists(deepgaze_data_path):
            deepgaze_data = scipy.io.loadmat(deepgaze_data_path)
            deepgaze_points = deepgaze_data[stimulus_name][0][0][4][0][0][2]

            # Filter and extract fixation points for DeepGazeIII
            filtered_points = deepgaze_points[(deepgaze_points[:, 0] >= 0) & (deepgaze_points[:, 1] >= 0)]
            x, y = filtered_points[:, 0], filtered_points[:, 1]
            # print(filtered_points)

            fixation_sections, deepgaze_x, deepgaze_y = get_fixation(x, y, time_interval)

            # Slider for selecting the starting fixation index for DeepGazeIII prediction
            fixation_start_idx = st.slider('Select Starting Fixation Index for DeepGazeIII', min_value=0, max_value=len(deepgaze_x)-4, step=1)

    
            image_tensor = torch.tensor([img.transpose(2, 0, 1)]).to(DEVICE)

            centerbias_template = np.load('../centerbias_mit1003.npy')
            centerbias = zoom(centerbias_template, (img.shape[0] / centerbias_template.shape[0], 
                                                    img.shape[1] / centerbias_template.shape[1]), 
                                                    order=0, mode='nearest')
            centerbias -= logsumexp(centerbias)
            centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)

            # Use a range of fixations starting from the selected index
            x_hist_tensor = torch.tensor([deepgaze_x[fixation_start_idx:fixation_start_idx + 4]]).to(DEVICE)
            y_hist_tensor = torch.tensor([deepgaze_y[fixation_start_idx:fixation_start_idx + 4]]).to(DEVICE)

            # Perform the prediction using the selected fixations
            with torch.no_grad():
                log_density_prediction = model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)

            # Create a new figure for DeepGazeIII visualization
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
            # Original Image with Fixations
            axs[0].imshow(img)
            axs[0].plot(deepgaze_x[0:fixation_start_idx + 4], deepgaze_y[0:fixation_start_idx + 4], 'o-', color='red')
            axs[0].scatter(deepgaze_x[fixation_start_idx + 3], deepgaze_y[fixation_start_idx + 3], 100, color='yellow')
            axs[0].set_title('Fixation Scanpath')
            axs[0].set_axis_off()

            # Predicted Heatmap
            predicted_heatmap = log_density_prediction.detach().cpu().numpy()[0, 0]
            axs[1].imshow(img, alpha=0.5)
            heatmap = axs[1].imshow(predicted_heatmap, cmap='jet', alpha=0.6)
            axs[1].plot(deepgaze_x[0:fixation_start_idx + 4], deepgaze_y[0:fixation_start_idx + 4], 'o-', color='red')
            axs[1].scatter(deepgaze_x[fixation_start_idx + 3], deepgaze_y[fixation_start_idx + 3], 100, color='yellow')
            axs[1].set_title('DeepGazeIII Predicted Heatmap')
            axs[1].set_axis_off()
            fig.colorbar(heatmap, ax=axs[1])

            # Display the DeepGazeIII prediction using Streamlit
            st.pyplot(fig)
