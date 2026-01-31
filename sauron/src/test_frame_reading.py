import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import numpy as np
#import open3d as o3d

data = sio.loadmat(r'/media/sauron/SAURON/forDistro/3 LIDAR/lidar_profile_2024_07_31.mat')

# Print keys and values in a readable format
print("Keys and values in the .mat file:")
for key in data.keys():
    if key.startswith('__'):  # Skip metadata keys
        continue
    value = data[key]
    size_bytes = value.nbytes if hasattr(value, 'nbytes') else 0
    print(f"\nKey: {key}")
    print(f"Size: {size_bytes / (1024*1024):.2f} MB")  # Convert bytes to MB
    print(f"Type: {type(value)}")
    print(f"Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
    print(f"Sample of values: {value[:1000] if hasattr(value, '__len__') else value}")



# # Extract avg_grid_out data
avg_grid = data['z_avg_grid_out']

# # Implement downsampling (take every nth point)
downsample_factor = 10  # Adjust this value to change downsampling rate
downsampled_data = avg_grid[::downsample_factor, ::downsample_factor]

# # Print original and downsampled shapes
# print("\nDownsampling Results:")
# print(f"Original shape: {avg_grid.shape}")
# print(f"Downsampled shape: {downsampled_data.shape}")

# # Optional: Visualize the downsampled data
# plt.figure(figsize=(95, 68))

# plt.subplot(121)
# plt.imshow(avg_grid)
# plt.title('Original Data')
# plt.colorbar()

# plt.subplot(122)
# plt.imshow(downsampled_data)
# plt.title('Downsampled Data')
# plt.colorbar()


# plt.tight_layout()
# plt.show()

# ... existing code ...

# Create a new color-thresholded visualization
plt.figure(figsize=(15, 10))

# Calculate statistics for better thresholding
data_mean = np.mean(downsampled_data)
data_std = np.std(downsampled_data)
vmin = data_mean - 2 * data_std  # Lower bound: mean - 2 standard deviations
vmax = data_mean + 2 * data_std  # Upper bound: mean + 2 standard deviations

# Create custom colormap with distinct thresholds
colors = ['darkblue', 'blue', 'lightblue', 'green', 'yellow', 'orange', 'red', 'darkred']
n_bins = len(colors)
threshold_levels = np.linspace(vmin, vmax, n_bins + 1)

# Create the thresholded visualization
plt.subplot(111)
im = plt.imshow(downsampled_data, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
plt.colorbar(im, label='Elevation (units)')
plt.title('Thresholded Elevation Map')

# Add threshold annotations
threshold_text = "Elevation Thresholds:\n"
for i in range(len(threshold_levels)-1):
    threshold_text += f"{threshold_levels[i]:.2f} to {threshold_levels[i+1]:.2f}: {colors[i]}\n"
plt.figtext(1.15, 0.5, threshold_text, fontsize=8, va='center')

# Add grid for better reference
plt.grid(True, alpha=0.3)

# Add axis labels
plt.xlabel('X Distance (units)')
plt.ylabel('Y Distance (units)')

plt.tight_layout()
plt.show()

# Print statistics
print("\nData Statistics:")
print(f"Mean elevation: {data_mean:.2f}")
print(f"Standard deviation: {data_std:.2f}")
print(f"Min elevation: {np.min(downsampled_data):.2f}")
print(f"Max elevation: {np.max(downsampled_data):.2f}")
# ... existing code ...

# After the data statistics printing, add:

# Define elevation threshold for object detection
elevation_threshold = data_mean + 0.1 * data_std  # Adjust multiplier as needed

# Create binary mask for high elevation points
binary_mask = (downsampled_data > elevation_threshold).astype(np.uint8)

# Import required CV2 for connected components analysis
import cv2

# Find connected components (objects) in the binary mask
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

# Create a new figure for object detection visualization
plt.figure(figsize=(15, 10))

# Plot the original downsampled data
plt.imshow(downsampled_data, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
plt.colorbar(label='Elevation (units)')

# Draw bounding boxes around detected objects
for i in range(1, num_labels):  # Start from 1 to skip background
    # Get object properties
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    
    # Filter out small objects (adjust minimum_area as needed)
    minimum_area = 50  # Minimum pixel area to be considered an object
    if area > minimum_area:
        # Draw rectangle
        rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=2)
        plt.gca().add_patch(rect)
        
        # Add label
        plt.text(x, y-5, f'Object {i}', color='red', fontsize=8)

plt.title('Elevation Map with Detected Objects')
plt.xlabel('X Distance (units)')
plt.ylabel('Y Distance (units)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print detection results
print("\nObject Detection Results:")
print(f"Elevation threshold: {elevation_threshold:.2f}")
print(f"Number of objects detected: {sum(1 for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > minimum_area)}")
