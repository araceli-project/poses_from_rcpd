import numpy as np
import cv2

def visualize_poses(poses_file="poses.npz", image_paths=None, output_dir="output"):
    
    poses_file = np.load(poses_file, allow_pickle=True)
    poses = poses_file['poses']
    print(poses.item().keys())
    print(poses.item()["poses_xy"])
    
    if image_paths is None:
        print("Please provide image_paths to visualize")
        return

    for idx, image_path in enumerate(image_paths):
        if idx >= len(poses.item()["poses_xy"]):
            break
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            continue
        
        pose_data = poses.item()["poses_xy"][idx]
        if pose_data.size > 0:
            for person_keypoints in pose_data:
                for keypoint in person_keypoints:
                    x, y = int(keypoint[0]), int(keypoint[1])
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        
        cv2.imwrite(f"pose_{image_path}", image)

if __name__ == "__main__":
    image_paths = ['image1.jpg', 'image2.jpg']  # Replace with actual image paths
    visualize_poses(image_paths=image_paths)
