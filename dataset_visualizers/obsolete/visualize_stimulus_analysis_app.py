import streamlit as st
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.image as mpimg

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

# Load the image and subject data
if os.path.exists(image_path) and os.path.exists(data_path):
    img = mpimg.imread(image_path)
    subject_data = scipy.io.loadmat(data_path)
    points = subject_data[stimulus_name][0][0][4][0][0][2]

    # Filter and flip x and y coordinates
    filtered_points = points[(points[:, 0] >= 0) & (points[:, 1] >= 0)]
    x = filtered_points[:, 0]
    y = filtered_points[:, 1]

    # Sampling rate and parameters
    time_interval = 1 / 240  # 240 Hz sampling rate
    velocity_threshold = 1000  # Velocity threshold for I-VT
    dispersion_threshold = 25  # Dispersion threshold for I-DT
    min_fixation_duration = 0.1  # Minimum fixation duration in seconds

    # Helper functions for I-VT and I-DT algorithms
    def calculate_velocity(x, y, time_interval):
        velocities = [np.sqrt((x[i] - x[i - 1])**2 + (y[i] - y[i - 1])**2) / time_interval for i in range(1, len(x))]
        return velocities

    def apply_ivt(x, y, time_interval, velocity_threshold, min_fixation_duration):
        velocities = calculate_velocity(x, y, time_interval)
        fixations, saccades = [], []
        current_fixation = []
        for i in range(len(velocities)):
            if velocities[i] < velocity_threshold:
                current_fixation.append((x[i], y[i]))
            else:
                if len(current_fixation) * time_interval >= min_fixation_duration:
                    fixations.append(np.array(current_fixation))
                else:
                    saccades.extend(current_fixation)
                current_fixation = []
        return fixations, saccades

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

    # Add buttons to run I-VT and I-DT algorithms
    run_ivt = st.button('Run I-VT')
    run_idt = st.button('Run I-DT')

    # Create a figure and display the image as the background
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.plot(x, y, marker='o', color='gray', linewidth=1, markersize=2, linestyle='-', alpha=0.6, label='Raw Gaze Path')

    # Run I-VT if button is clicked
    if run_ivt:
        ivt_fixations, ivt_saccades = apply_ivt(x, y, time_interval, velocity_threshold, min_fixation_duration)
        for fixation in ivt_fixations:
            ax.plot(fixation[:, 0], fixation[:, 1], marker='o', color='blue', linewidth=0, markersize=5, label='I-VT Fixation')
        if len(ivt_saccades) > 0:
            ivt_saccades = np.array(ivt_saccades)
            ax.plot(ivt_saccades[:, 0], ivt_saccades[:, 1], marker='x', color='green', linewidth=1, markersize=6, label='I-VT Saccade')
        st.write("I-VT Analysis Complete")

    # Run I-DT if button is clicked
    if run_idt:
        idt_fixations, idt_saccades = apply_idt(x, y, time_interval, dispersion_threshold, min_fixation_duration)
        for fixation in idt_fixations:
            ax.plot(fixation[:, 0], fixation[:, 1], marker='o', color='red', linewidth=0, markersize=5, label='I-DT Fixation')
        if len(idt_saccades) > 0:
            idt_saccades = np.array(idt_saccades)
            ax.plot(idt_saccades[:, 0], idt_saccades[:, 1], marker='x', color='purple', linewidth=1, markersize=6, label='I-DT Saccade')
        st.write("I-DT Analysis Complete")

    # Set plot limits and labels
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f'Gaze Data Analysis for Subject {selected_subject} on {selected_image}')
    ax.legend()

    # Display the final plot using Streamlit
    st.pyplot(fig)

else:
    st.write("Please select a valid image and subject.")
