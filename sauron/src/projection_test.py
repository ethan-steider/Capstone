import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from tqdm import tqdm

class SensorFusionSystem:
    def __init__(self, lidar_path, image_dir, pc_shape=(9500, 6750)):
        # Initialize system parameters
        self.pc_width, self.pc_height = pc_shape
        self.image_dir = image_dir
       
        # Load and process LiDAR data
        self.load_lidar_data(lidar_path)
        self.process_lidar_data()
       
        # Initialize camera parameters
        self.focal_length = 8  # mm
        self.sensor_size = (14.0, 10.5)  # mm
        self.image_res = (4096, 3000)
       
        # Calculate spatial resolution
        self.res_x = 15 / pc_shape[0]  # meters/pixel
        self.res_y = 15 / pc_shape[1]  # meters/pixel

    def load_lidar_data(self, path):
        """Load and process LiDAR data from .mat file"""
        data = sio.loadmat(path)
        self.avg_grid = data['z_avg_grid_out']
        print(f"Loaded LiDAR data with shape: {self.avg_grid.shape}")

    def process_lidar_data(self, downsample_factor=10):
        """Process LiDAR data with downsampling and statistics"""
        self.downsampled_data = self.avg_grid[::downsample_factor, ::downsample_factor]
        self.data_mean = np.mean(self.downsampled_data)
        self.data_std = np.std(self.downsampled_data)
        print(f"Downsampled LiDAR shape: {self.downsampled_data.shape}")

    def load_rgb_images(self):
        """Load RGB images from specified directory"""
        self.image_files = sorted([f for f in os.listdir(self.image_dir)
                                 if f.lower().endswith(('png', 'jpg', 'jpeg'))])
        print(f"Found {len(self.image_files)} RGB images")
        return self.image_files

    def project_frame(self, rgb_img, frame_idx):
        """Project RGB frame onto point cloud"""
        # Calculate corresponding point cloud column
        col = frame_idx % self.pc_width

        step = max(1, self.pc_width // max(1, len(self.image_files)))

        tile_width = rgb_img.shape[1] // step
       
        # Extract relevant point cloud slice
        pc_slice = self.avg_grid[col::step, :]
       
        # Calculate projection matrix (simplified example)
        f = self.focal_length * max(rgb_img.shape)/self.sensor_size[0]
        K = np.array([[f, 0, rgb_img.shape[1]/2],
                      [0, f, rgb_img.shape[0]/2],
                      [0, 0, 1]])
       
        # Perform perspective warping
        warped = cv2.warpPerspective(rgb_img, K, (tile_width, pc_slice.shape[0]))
        return warped, pc_slice

    def fuse_data(self):
        """Main fusion pipeline"""
        self.load_rgb_images()
        fused_results = []
       
        for idx, img_file in enumerate(tqdm(self.image_files)):
            img = cv2.imread(os.path.join(self.image_dir, img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
           
            # Project and fuse
            aligned_rgb, pc_slice = self.project_frame(img, idx)
            fused = np.dstack((aligned_rgb,
                             cv2.resize(pc_slice, (aligned_rgb.shape[1], aligned_rgb.shape[0]))))
            fused_results.append(fused)
           
            # Visualization and analysis
            if idx % 1000 == 0:
                self.visualize_fusion(fused, idx)
                self.detect_objects(pc_slice, idx)
       
        return fused_results

    def visualize_fusion(self, fused_data, idx):
        """Visualize fused data"""
        plt.figure(figsize=(18, 10))
       
        plt.subplot(121)
        plt.imshow(fused_data[..., :3])
        plt.title(f'RGB Projection - Frame {idx}')
       
        plt.subplot(122)
        plt.imshow(fused_data[..., 3], cmap='jet')
        plt.title('Elevation Data')
        plt.colorbar()
       
        plt.tight_layout()
        plt.show()

    def detect_objects(self, pc_slice, idx):
        """Elevation-based object detection"""
        threshold = self.data_mean + 0.1 * self.data_std
        binary_mask = (pc_slice > threshold).astype(np.uint8)
       
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, 8)
       
        print(f"\nFrame {idx} - Objects detected: {num_labels-1}")
        print(f"Max elevation: {np.max(pc_slice):.2f}, Mean: {np.mean(pc_slice):.2f}")

# Usage ------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize system with paths
    fusion_system = SensorFusionSystem(
        lidar_path=r'/media/sauron/SAURON/forDistro/3 LIDAR/lidar_profile_2024_07_31.mat',
        image_dir=r'/media/sauron/SAURON/forDistro/1 Downlooking/RGB',
        pc_shape=(9500, 6750)
    )
   
    # Run fusion pipeline
    fused_results = fusion_system.fuse_data()
