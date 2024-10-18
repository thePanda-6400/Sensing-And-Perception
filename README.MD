THIS IS A README FILE FOR THE SENSING AND PERCEPTION COURSEWORK 

EXECUTING 3D RECONSTRUCTION CODE

The code is saved as final.py and a jupyter notebook with the same code has also been provided depending on the user's preference
The images used in the 3d reconstruction have been provided with the camera matrix provided in the code but if new images are to be used follow the steps given below 

First save the images whose 3d reconstruction is to be done in the same folder as the python script as a jpeg with the labels imgxx where xx ranges from 01 to the final number of images taken in the double digits. 

Add the path of the images to the code in line 6 of the script
The camera matrix used is defined on line 22. While images used for the reconstruction have been provided, if the code is being run on new set of images, the camera matrix for the camera matrix should be used in this location

EXECUTING GRASPING
The simulation was carried out using ROS1 Noetic on an Ubuntu 20.04 system
Follow the instructions on  http://wiki.ros.org/noetic/Installation/Ubuntu to install the system 
install MoveIt using the instructions in https://moveit.github.io/moveit_tutorials/doc/getting_started/getting_started.html

Open the two terminals, 

in the first one enter the following to launch the simulator with the Panda Robot
roslaunch panda_moveit_config demo.launch

In the second terminal enter the following command to launch the pick and place programme
rosrun moveit_tutorials pick_place_tutorial