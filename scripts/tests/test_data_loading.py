# Create test script: scripts/test_data_loading.py
import open3d as o3d
import numpy as np
import os

def test_8iVFB_loading():
    data_path = "datasets/8iVFB_small"
    print(f"Testing data loading from: {os.getcwd()}")
    if os.path.exists(data_path):
        # Find first .ply file
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith('.ply'):
                    file_path = os.path.join(root, file)
                    print(f"Testing with: {file_path}")
                    
                    # Load with Open3D
                    pcd = o3d.io.read_point_cloud(file_path)
                    print(f"Points: {len(pcd.points)}")
                    print(f"Has colors: {len(pcd.colors) > 0}")
                    print(f"Has normals: {len(pcd.normals) > 0}")
                    
                    # Convert to numpy
                    points = np.asarray(pcd.points)
                    print(f"Point cloud shape: {points.shape}")
                    print(f"Point range: [{points.min():.3f}, {points.max():.3f}]")
                    return True
    return False

if __name__ == "__main__":
    success = test_8iVFB_loading()
    print(f"Data loading test: {'PASSED' if success else 'FAILED'}")