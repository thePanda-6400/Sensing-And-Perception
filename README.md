3D Reconstruction and Grasping using Computer Vision and Robotics
Overview

This project explores 3D object reconstruction using computer vision and the implementation of a grasping algorithm for robotic manipulation. The project involves capturing object images, generating a 3D model, and enabling a robot to pick and place the reconstructed object using ROS, MoveIt!, and Gazebo.
Author

Dhruv Pandit
K23118144
Key Objectives

    3D Reconstruction of an object using image processing techniques.

    Camera Calibration for accurate depth estimation.

    Feature Detection & Matching for generating a point cloud.

    Implementation of a Grasping Algorithm to manipulate objects using a robotic arm.

Technologies & Tools Used

    Computer Vision: OpenCV, Open3D

    Feature Detection: SIFT (Scale-Invariant Feature Transform)

    3D Reconstruction: Point Cloud Processing, STL Model Export

    Robot Simulation: ROS, MoveIt!, Gazebo, RViz

    Hardware: Panda Robot

Methodology
1. Image Capture

    Images were captured using a smartphone camera.

    A checkerboard pattern was used for camera calibration.

    Object images were taken under different angles to ensure better reconstruction.

2. Camera Calibration

    Intrinsic parameters such as focal length and principal points were obtained.

    Extrinsic parameters helped determine the camera’s position in the 3D space.

    Used OpenCV’s findChessboardCorners() for calibration.

3. 3D Reconstruction

    Feature Extraction: SIFT algorithm detects keypoints from captured images.

    Feature Matching:

        Brute Force Matcher & K-Nearest Neighbors (KNN) were used for matching keypoints.

        RANSAC filtering was applied to remove incorrect matches.

    Point Cloud & Dense Reconstruction:

        Open3D was used to generate a point cloud and dense 3D model.

        The final mesh was exported in STL format for simulation in Gazebo.

4. Grasping Algorithm

    The Panda Robot was used for object manipulation.

    MoveIt!’s pick-and-place function was integrated for grasping tasks.

    The STL model was imported into RViz for visualization.

    Due to simulation constraints, a random object was used instead of the actual STL model.

Challenges & Improvements

    Smartphone camera filters distorted images → Future work could use RAW images for better accuracy.

    Cluttered background caused incorrect feature detection → Used plain backgrounds to improve object recognition.

    Importing STL into RViz was problematic → Used a placeholder object for grasping tests.

Future Work

    Implement real-world grasping instead of simulation.

    Improve image preprocessing to remove noise and enhance feature extraction.

    Optimize path planning for more precise robotic movement.3D Reconstruction and Grasping using Computer Vision and Robotics
Overview

This project explores 3D object reconstruction using computer vision and the implementation of a grasping algorithm for robotic manipulation. The project involves capturing object images, generating a 3D model, and enabling a robot to pick and place the reconstructed object using ROS, MoveIt!, and Gazebo.
Author

Dhruv Pandit
K23118144
Key Objectives

    3D Reconstruction of an object using image processing techniques.

    Camera Calibration for accurate depth estimation.

    Feature Detection & Matching for generating a point cloud.

    Implementation of a Grasping Algorithm to manipulate objects using a robotic arm.

Technologies & Tools Used

    Computer Vision: OpenCV, Open3D

    Feature Detection: SIFT (Scale-Invariant Feature Transform)

    3D Reconstruction: Point Cloud Processing, STL Model Export

    Robot Simulation: ROS, MoveIt!, Gazebo, RViz

    Hardware: Panda Robot

Methodology
1. Image Capture

    Images were captured using a smartphone camera.

    A checkerboard pattern was used for camera calibration.

    Object images were taken under different angles to ensure better reconstruction.

2. Camera Calibration

    Intrinsic parameters such as focal length and principal points were obtained.

    Extrinsic parameters helped determine the camera’s position in the 3D space.

    Used OpenCV’s findChessboardCorners() for calibration.

3. 3D Reconstruction

    Feature Extraction: SIFT algorithm detects keypoints from captured images.

    Feature Matching:

        Brute Force Matcher & K-Nearest Neighbors (KNN) were used for matching keypoints.

        RANSAC filtering was applied to remove incorrect matches.

    Point Cloud & Dense Reconstruction:

        Open3D was used to generate a point cloud and dense 3D model.

        The final mesh was exported in STL format for simulation in Gazebo.

4. Grasping Algorithm

    The Panda Robot was used for object manipulation.

    MoveIt!’s pick-and-place function was integrated for grasping tasks.

    The STL model was imported into RViz for visualization.

    Due to simulation constraints, a random object was used instead of the actual STL model.

Challenges & Improvements

    Smartphone camera filters distorted images → Future work could use RAW images for better accuracy.

    Cluttered background caused incorrect feature detection → Used plain backgrounds to improve object recognition.

    Importing STL into RViz was problematic → Used a placeholder object for grasping tests.

Future Work

    Implement real-world grasping instead of simulation.

    Improve image preprocessing to remove noise and enhance feature extraction.

    Optimize path planning for more precise robotic movement.
