# Import necessary libraries
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg


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

# Flip x and y coordinates
x = filtered_points[:, 1]
y = filtered_points[:, 0]


# Create a figure and display the image as the background
fig, ax = plt.subplots()
ax.imshow(img)

# Overlay the polyline on the image using the filtered and flipped coordinates
ax.plot(x, y, marker='o', color='red', linewidth=2, markersize=5)

# Set limits to match the image dimensions
ax.set_xlim(0, img.shape[1])
ax.set_ylim(img.shape[0], 0)  # Invert y-axis to match image orientation

# Add labels and title
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Polyline Overlay on Image")

# Show the final plot
plt.show()
