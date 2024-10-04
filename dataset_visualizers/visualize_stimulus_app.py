import streamlit as st
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.image as mpimg

# Set the paths for the stimuli, data, and affiliated image folders
stimuli_folder = '../Datasets/MIT1003/ALLSTIMULI/'
data_folder = '../Datasets/MIT1003/DATA/'
fixation_folder = '../Datasets/MIT1003/ALLFIXATIONMAPS/'

# List all the image files in the stimuli folder
image_files = [f for f in os.listdir(stimuli_folder) if f.endswith('.jpeg') or f.endswith('.jpg')]

# List all subject folders in the data folder
subject_folders = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]

# Create Streamlit selection boxes for image and multiple subjects
selected_image = st.selectbox('Select an Image File', image_files)

# Add a button to select all subjects
select_all = st.button('Select All Subjects')

# Automatically select all subjects if the button is clicked
if select_all:
    selected_subjects = st.multiselect('Select Subject Folders (multiple selection)', subject_folders, default=[])
    selected_subjects = subject_folders
else:
    selected_subjects = st.multiselect('Select Subject Folders (multiple selection)', subject_folders, default=[])

# Extract image filename without extension
stimulus_name = os.path.splitext(selected_image)[0]

# Set up the paths for the main image and fixation maps
image_path = os.path.join(stimuli_folder, selected_image)
fixmap_path = os.path.join(fixation_folder, f'{stimulus_name}_fixMap.jpg')
fixpts_path = os.path.join(fixation_folder, f'{stimulus_name}_fixPts.jpg')

# Checkboxes to select whether to display affiliated images
display_fixmap = st.checkbox('Display Fixation Map')
display_fixpts = st.checkbox('Display Fixation Points')

# Display the selected image and subjects for confirmation
st.write(f'Selected Image: {selected_image}')
st.write(f'Selected Subjects: {", ".join(selected_subjects) if selected_subjects else "None"}')

# Button to run I-VT and I-DT algorithms
run_ivt = st.button('Run I-VT')
run_idt = st.button('Run I-DT')

# Load and display the image with overlaid data if selections are made
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

            # Overlay the polyline for each subject using distinct colors
            ax.plot(x, y, marker='o', color=colors(idx), linewidth=0.5, markersize=2, label=f'Subject {subject}')

            # Run I-VT or I-DT if the buttons are clicked
            if run_ivt or run_idt:
                # Parameters for fixation detection
                time_interval = 1 / 240  # Sampling rate is 240 Hz (4.167 ms interval)
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
                                current_fixation = []
                    return fixations

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
                        i = j
                    return fixations

                # Visualize fixations and saccades based on the selected algorithm
                if run_ivt:
                    ivt_fixations = apply_ivt(x, y, time_interval, velocity_threshold, min_fixation_duration)
                    for fixation in ivt_fixations:
                        ax.plot(fixation[:, 0], fixation[:, 1], marker='o', color='blue', markersize=5, label='I-VT Fixation')

                if run_idt:
                    idt_fixations = apply_idt(x, y, time_interval, dispersion_threshold, min_fixation_duration)
                    for fixation in idt_fixations:
                        ax.plot(fixation[:, 0], fixation[:, 1], marker='x', color='green', markersize=5, label='I-DT Fixation')

    # Display the final plot using Streamlit
    st.pyplot(fig)
else:
    st.write('Please select a valid image and at least one subject.')
