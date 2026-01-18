import scipy.io
import matplotlib.pyplot as plt

fin = "/run/media/anna/New Volume/forDistro/2 Sidelooking/img_cuda_2024_07_31_side_looking_hanning.mat"
#fin = "/run/media/anna/New Volume/forDistro/1 Downlooking/2024_07_31_aspire_3d_sar_img.mat"

# Load the .mat file
mat_data = scipy.io.loadmat(fin)  # Replace 'your_file.mat' with your actual file path

# Access each variable and plot
# Assuming img_* variables are 2D arrays representing images
img_vars = ['img_hh', 'img_hv', 'img_vh', 'img_vv']
for img_var in img_vars:
    img_data = mat_data[img_var]
    
    plt.figure()
    plt.imshow(img_data, cmap='gray')  # Adjust colormap as needed
    plt.colorbar()
    plt.title(f'Image: {img_var}')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

# Plot x_img, x_strip, y_img, z_img if they are 1D arrays or coordinates
if 'x_img' in mat_data and 'y_img' in mat_data:
    plt.figure()
    plt.plot(mat_data['x_img'], mat_data['y_img'], label='Y vs. X')
    plt.xlabel('X Image')
    plt.ylabel('Y Image')
    plt.title('Y vs. X Image')
    plt.legend()

if 'z_img' in mat_data:
    plt.figure()
    plt.plot(mat_data['x_img'], mat_data['z_img'], label='Z vs. X')
    plt.xlabel('X Image')
    plt.ylabel('Z Image')
    plt.title('Z vs. X Image')
    plt.legend()

plt.show()
