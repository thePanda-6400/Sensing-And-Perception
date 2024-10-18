import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import open3d as o3d
images = sorted(glob.glob('C:/Users/DHRUV/Downloads/Sensing and perception/img??.jpeg')) ##CHANGE TO PATH WHERE IMAGES ARE STORED LOCALLY


img1 = cv2.imread('img02.jpeg')  
img2 = cv2.imread('img03.jpeg') 



# Convert to grayscale
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Print the shape of the images
print(f'Image1 Shape: {img1_gray.shape}')
print(f'Image2 Shape: {img2_gray.shape}')
#Define Camera Matrix
K = np.array([[1.66006937e+03, 0.00000000e+00, 1.02769927e+03],
 [0.00000000e+00, 1.65302995e+03, 8.28393603e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])



bf = cv2.BFMatcher()

sift = cv2.SIFT_create(nfeatures=10000)

def get_matches(kp1, d1, kp2, d2):
  matches = bf.knnMatch(d1, d2, k=2)

  good_matches = [m for m,n in matches if m.distance < 0.7*n.distance] # Lowe's ratio test
  print('GOOD MATCHES: ', len(good_matches), ' out of total: ', len(matches))

  #Determine 2D coords of the same feature in each of the two images
  #(One pair of coords for each match)
  pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
  pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

  #Estimate the relative pose of the cameras
  E, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 500)
  _, R, t, _ = cv2.recoverPose(E, pts1, pts2, cameraMatrix=K, mask=mask)
  return pts1, pts2, mask, R, t, good_matches


#Display matches in two images
def show_matches(mask, img1, kp1, img2, kp2, good_matches):
  matchesMask = mask.ravel().tolist()
  draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = None,
                    matchesMask = matchesMask,  # Draw only inliers
                    flags = 2)
  matched_img = cv2.drawMatches(img1,kp1,img2,kp2,good_matches,None,**draw_params)
  plt.imshow(matched_img, cmap='gray')
  plt.show()
  return matched_img


def feature_extraction(img1, img2, displaying_matches=False):
  kp1,d1 = sift.detectAndCompute(img1, None)
  kp2,d2 = sift.detectAndCompute(img2, None)
  pts1, pts2, mask, R, t, good_matches = get_matches(kp1, d1, kp2, d2)
  if displaying_matches:
    show_matches(mask, img1, kp1, img2, kp2, good_matches)
  return pts1, pts2, mask, R, t, good_matches

#Triangulate the features from a pair of images and generate a point cloud,
#Using the Camera's parameters and relative pose

def to_pointcloud(points_3D):
  #Convert numpy array to a open3d point cloud format
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points_3D)
  return pcd

def triangulate(K, R, t, pts1, pts2):
  P1 = np.dot(K, np.hstack((np.identity(3), np.zeros((3, 1)))))
  P2 = np.dot(K, np.hstack((R, t)))

  points_3D = cv2.triangulatePoints(P1, P2, np.transpose(pts1), np.transpose(pts2))
  points_3D /= points_3D[3]
  points_3D = np.transpose(points_3D[:3,:])

  return to_pointcloud(points_3D)

#Remove outliers from the point cloud based on distance from nearest neighbours (measured in standard deviations of the distribution)
def filter_outliers(pcd):
  _, indices_of_inliers = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1)
  inlier_cloud = pcd.select_by_index(indices_of_inliers)
  return inlier_cloud
#Triangulate the features from a pair of images and generate a point cloud,
#Using the Camera's parameters and relative pose


points_3D = np.empty((0, 3))


for current, next in zip(images,images[1:]):
  img1 = cv2.imread(current)
  img2 = cv2.imread(next)
  pts1, pts2, mask, R, t, good_matches = feature_extraction(img1, img2, displaying_matches=True)
  pcd = triangulate(K, R, t, pts1, pts2)
  inlier_cloud = filter_outliers(pcd)
  points_3D = np.concatenate((points_3D, inlier_cloud.points), axis=0)
  o3d.visualization.draw_plotly([inlier_cloud])

pcd = to_pointcloud(points_3D)
o3d.visualization.draw_plotly([pcd])
mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 1, mesh, pt_map)
mesh.compute_vertex_normals()
o3d.visualization.draw_plotly([mesh])
o3d.io.write_triangle_mesh("output_mesh.stl", mesh)
