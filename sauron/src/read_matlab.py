import scipy.io
import matplotlib.pyplot as plt

fin = "/run/media/anna/New Volume/forDistro/2 Sidelooking/img_cuda_2024_07_31_side_looking_hanning.mat"
#fin = "/run/media/anna/New Volume/forDistro/1 Downlooking/2024_07_31_aspire_3d_sar_img.mat"
# Load the .mat file
mat_data = scipy.io.loadmat(fin)  # Replace 'your_file.mat' with the path to your file

# Inspect the keys to understand the data structure
#print(mat_data.keys())

for key in mat_data.keys():
    print(key)

#helolo
# Access a specific variable from the MATLAB data
# Replace 'variable_name' with the name of the variable you want to plot

#"""
data = mat_data[input("which value to plot: ")]  # For example: mat_data['data']
print(data[0])
print("---------")
print(len(data[0]))
"""
# Assuming data is in a simple array format, plot it
plt.plot(mat_data["x_img"][0],mat_data["y_img"][0] )
plt.xlabel('X-axis label')  # Set your X-axis label
plt.ylabel('Y-axis label')  # Set your Y-axis label
plt.title('Graph Title')    # Set your graph title
plt.show()
"""
