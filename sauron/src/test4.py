import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import numpy as np
#import open3d as o3d

file_path = r"/media/sauron/SAURON/forDistro/3 LIDAR/lidar_point_cloud_2024_07_31.mat"

# Load MATLAB data
def load_matlab_data(file_path):
    data = sio.loadmat(file_path)
    return data

# Extract x, y, z coordinates
def extract_coordinates(pointCloud):
    x = pointCloud['x_lidar'].flatten()
    y = pointCloud['y_lidar'].flatten()
    z = pointCloud['z_lidar'].flatten()
    return x, y, z

# Plot 3D point cloud using Matplotlib
def plot_with_matplotlib(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=z, cmap='viridis')
    plt.show()

# Plot 3D point cloud using Plotly
def plot_with_plotly(x, y, z):
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(color=z, colorscale='Viridis'))])
    fig.show()

 # Visualize point cloud using Open3D
# def visualize_with_open3d(x, y, z):
#    pcd = o3d.geometry.PointCloud()
#    pcd.points = o3d.utility.Vector3dVector(np.column_stack((x, y, z)))
#    o3d.visualization.draw_geometries([pcd])

# Main function
def main():
    file_path2 = file_path
    pointCloud = load_matlab_data(file_path2)
    x, y, z = extract_coordinates(pointCloud)
    
    print("Plotting with Matplotlib...")
    plot_with_matplotlib(x, y, z)
    
    print("Plotting with Plotly...")
    plot_with_plotly(x, y, z)
    
    print("Visualizing with Open3D...")
    visualize_with_open3d(x, y, z)

if __name__ == "__main__":
    main()
