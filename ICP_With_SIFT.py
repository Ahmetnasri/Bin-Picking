import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from Reading_Writing_Reference_Points import *
import random

#Description of the this script:
# This script encludes all needed functions to apply Iterative Closest Point - ICP algorithm to estimate the 3D Pose of an object
# from its 2D-image. It takes the 3D-Coordinates of specific points and there pixel-coordinates in the image matrix to do this estimation.
# This scripts needs 3D-coordinates of specific points, the pixel coordinates of these points, camera parameters in form of intrinsic camera
# matrix


#>>>>> This function creates a transformation matrix using the six pose parameters: x, y, z, alpha, beta, gamma. These parameter should be
#      given to the function in form of a row vector    <<<<<
def vector_to_matrix(vector): 

    rotation_x = np.zeros((3,3))
    rotation_x[0][0] = 1.0
    rotation_x[1][1] = math.cos(vector[3])
    rotation_x[1][2] = -math.sin(vector[3])
    rotation_x[2][1] = math.sin(vector[3])
    rotation_x[2][2] = math.cos(vector[3])

    rotation_y = np.zeros((3,3))
    rotation_y[1][1] = 1.0
    rotation_y[0][0] = math.cos(vector[4])
    rotation_y[0][2] = math.sin(vector[4])
    rotation_y[2][0] = -math.sin(vector[4])
    rotation_y[2][2] = math.cos(vector[4])

    rotation_z = np.zeros((3,3))
    rotation_z[2][2] = 1.0
    rotation_z[0][0] = math.cos(vector[5])
    rotation_z[0][1] = -math.sin(vector[5])
    rotation_z[1][0] = math.sin(vector[5])
    rotation_z[1][1] = math.cos(vector[5])

    # Transformation matrix in form Rz.Ry.Rx (applies the rotation on x then y the z)
    rotation_matrix = rotation_z.dot(rotation_y.dot(rotation_x))

    trans_matrix = np.zeros((3,4))
    for i in range(3):
        for j in range(3):
            trans_matrix[i][j] = rotation_matrix[i][j]
    
    trans_matrix[0][3] = vector[0]
    trans_matrix[1][3] = vector[1]
    trans_matrix[2][3] = vector[2]

    return trans_matrix


#>>>>> This function projects on the image matrix to find there supposed pixel-coordinates on the image.
#      It uses the 3D-coordinates of the points, camera parameter in form of an intrinsic camera matrix, and the pose of the camera
#      in respect to the worlds origin.
def project_points(pose_vector, points, intrinsic_camera):
    
    #creating a transformation matrix from the world to camera pose
    extrinsic_matrix = vector_to_matrix(pose_vector)
    
    #the full transformation matrix is the transformation from worlds origin to the camera and then from camera to object
    trans_matrix = intrinsic_camera.dot(extrinsic_matrix)

    #tranformed_points = extrinsic_matrix.dot(points)
    projection = trans_matrix.dot(points)

    for i in range(len(projection)):
        for j in range(len(projection[0])):
            projection[i][j] = projection[i][j]/projection[2][j] #Eliminating the z component of the projected points by normalizing 
                                                                 #x and y in respect to z
    
    #Projection vector is given in form of a column vector. x and y coordinates alternate so that every two rows give the x and y of one point
    projection_vector = np.zeros((2*len(points[0]), 1))
    for i in range(len(points[0])):
        projection_vector[2*i] = projection[0][i]
        projection_vector[2*i + 1] = projection[1][i]
    
    return projection_vector#, tranformed_points

#>>>>> This function separate the coordinates of group of points into 3 arrays, one for each axis
def separate_xyz(points):
    x = []
    y = []
    z = []

    for point in points:
        x.append(point[0])
        y.append(point[1])
        z.append(point[2])
    
    return x, y, z


#>>>>> This function takes the 3D-coordinates of a point and add another element to them to make the multiplication with the transformation
#      matrix mathematically possible. This element would have the value 1
#      multiple points can be given to the function at once, all x values should be given in form of a row vector. The same for y and z
def prep_points(x, y, z):

    complete_points = np.zeros((4, len(x))).T
    for i in range(len(complete_points)):
        complete_points[i][0] = x[i]
        complete_points[i][1] = y[i]
        complete_points[i][2] = z[i]
        complete_points[i][3] = 1
   
    return complete_points.T


def prep_matrix(matrix):
    new_matrix = np.zeros((4, 4))
    new_matrix[0:3][0:4] = matrix
    new_matrix[3][3] = 1.0

    return new_matrix

    
#>>>>> This function calculate the error in the estimated pose using mean square distance between the supposed points pixel-coordinates, according
#      to the estimated pose, and the pixel-coordinates of these points in the given image
def calc_error(guessed_points, image_points):

    sum_error = 0

    for i in range(len(guessed_points)):
        sum_error += (guessed_points[i] - image_points[i])**2

    return sum_error/len(guessed_points)

def calc_error_new(guessed_points, image_points):
    print(guessed_points.shape)
    print(image_points.shape)

    return 10


#>>>>> This function calculate the derivative (jacobian matrix) of the pose vector at a given point in respect to all six parameters: x,y,z,Rx,Ry,Rz
def calc_jacobian(step_length, guessed_points, transformation_vector, points, intrinsic_camera):

    J = (np.zeros((2*len(points), 6))).T

    for i in range(6):
        vector = (np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).T).reshape(6,1)
        vector[i] = step_length
        rotation_vector, translation_vector = split_rotation_translation(transformation_vector + vector)
        
        b, j = cv.projectPoints(np.array(points), rotation_vector, translation_vector, intrinsic_camera, np.zeros((4,1)))
        b = order_points(b)

        a = guessed_points
                
        J[i] = ((b - a))[0] / step_length
        
    return J.T

#>>>>> This function update the estimated pose vector using the calculated derivatives
def update_transformation_vector(y, y0, x, J):
    
    #print("\n\tupdate_transformation_vector:")
    pseudo_inv = np.linalg.pinv(J)

    dy = y0 - y

    dx = pseudo_inv.dot(dy)

    x += dx
    
    return x


#>>>>> This function applies Scale Invariant Feature Transformation - SIFT to find corresponding points in the image
def apply_sift(img, nFeatures = 100, nOctaves = 5):
    sift = cv.xfeatures2d.SIFT_create(nfeatures=nFeatures, nOctaveLayers=nOctaves)
    kp, des = cv.xfeatures2d_SIFT.detectAndCompute(sift, img, None)
    return kp, des


#>>>>> This function applies sobel filter for edge detection
def apply_sobel(img, scale = 1, delta = 0, ddepth = 3, ksize=3, weight=1, gamma=0):

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=ksize, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=ksize, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(abs_grad_x, weight, abs_grad_y, weight, gamma)

    return grad


#>>>>> This function creates the intrinsic camera matrix using camera parameters
def get_camera_intrinsic_matrix_old(camera_width_pixels = 800, camera_hight_pixels = 600, optical_center_x = 400, optical_center_y = 300, s = 0, px = 0, py = 0):

    #matrix1 = np.zeros((3,3))
    #matrix1[0][0] = camera_width_pixels
    #matrix1[1][1] = camera_hight_pixels
    #matrix1[2][2] = 1.0

    matrix2 = np.zeros((3,3))
    matrix2[0][0] = optical_center_x
    matrix2[1][1] = optical_center_y
    matrix2[0][1] = s/camera_width_pixels
    matrix2[0][2] = px
    matrix2[1][2] = py
    matrix2[2][2] = 1.0

    #intrinsic_camera = matrix1.dot(matrix2)

    return matrix2

#>>>>> This function creates the points vector (with x and y alternating) using the keypoints that where found by SIFT
def get_points_from_kp(keypoints):

    image_points_vector = np.zeros((2*len(keypoints), 1))
    i = 0
    for kp in keypoints:
        image_points_vector[2*i] = kp.pt[0]
        image_points_vector[2*i + 1] = kp.pt[1]
        i += 1

    return image_points_vector

#>>>>> This function finds matches between a query image and a train image. It takes the description of the found points in both images
def get_good_matches(des_query, des_train, threshold = 0.75, nMatches = 2):

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des_query, des_train, k= nMatches)
    good = []
    for m,n in matches:
        if m.distance < threshold*n.distance:
            good.append([m])
    q_index = []
    t_index = []
    for dmatch in good:
        q_index.append(dmatch[0].queryIdx)
        t_index.append(dmatch[0].trainIdx)


    return good, q_index, t_index


#>>>>> This function takes all known points and give only the 3D-coordinates and the keypoints corresponding to the found points in the image
def get_matches_with_coordinates(reference_points_coordinates, keypoints, reference_index_list, keypoint_index_list):
    points_coordinates = []

    for i in reference_index_list:
        points_coordinates.append(reference_points_coordinates[i])
    
    image_coordinates = []

    kp = []

    for j in keypoint_index_list:
        kp.append(keypoints[j])
        image_coordinates.append(keypoints[j].pt)

    return points_coordinates, image_coordinates, kp

#>>>>> This function checks if the found point has been already found
def point_added(list_of_points, to_add_point):
    for point in list_of_points:
        if to_add_point[0] == point[0] and to_add_point[1] == point[1] and to_add_point[2] == point[2]:
            return True

    return False

#>>>>> This function delete duplicates from found points
def get_unique_points(points_coordinates, image_points_coordinates, keypoints):
    unique_points = []
    unique_image_points = []
    unique_keypoints = []

    for i in range(len(keypoints)):
        if not (keypoints[i] in unique_keypoints):

            unique_points.append(points_coordinates[i])

            unique_image_points.append(image_points_coordinates[i])

            unique_keypoints.append(keypoints[i])

    return unique_points, unique_image_points, unique_keypoints


def visualize_points(points, world_to_object = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):

    transformation_vector = (np.array(world_to_object).T).reshape(6,1)

    transformation_matrix = vector_to_matrix(transformation_vector)
    #print('i am herew#')
    #print(type(points))
    #print(points.shape)

    trans_points = transformation_matrix.dot(points)
    
    x = trans_points[0][:]
    y = trans_points[1][:]
    z = trans_points[2][:]

    fig = plt.figure()
    ax = Axes3D(fig)
    

    ax.scatter(x, y, z, c="#000000", s=50)
    """t = "Visualization of Points after the Transformation " + str(transformation_vector)
    ax.set_label("Something")"""
    #plt.show()


def split_rotation_translation(transformation_vector):
    rotation_vector = np.zeros((3,1))
    translation_vector = np.zeros((3,1))

    for i in range(3):
        translation_vector[i][0] = transformation_vector[i][0]

    for i in range(3):
        rotation_vector[i][0] = transformation_vector[i+3][0]

    return rotation_vector, translation_vector

def order_points(image_points):

    image_points_vector = np.zeros((2*len(image_points), 1))

    i = 0
    for p in image_points:
        image_points_vector[2*i] = p[0][0]
        image_points_vector[2*i + 1] = p[0][1]
        i += 1
    
    return image_points_vector

def order_points_2(image_points):

    image_points_vector = np.zeros((2*len(image_points), 1))

    i = 0
    for p in image_points:
        image_points_vector[2*i] = p[0]
        image_points_vector[2*i + 1] = p[1]
        i += 1
    
    return image_points_vector

def get_sub_group(points, image_points, keypoints, size=10):

    temp_points = []
    temp_image_points = []
    temp_key_points = []

    indeces = []

    for i in range(size):
        index = -1
        while True:
            index = random.randint(0, len(points)-1)
            if not(index in indeces):
                break
        
        temp_points.append(points[index])
        temp_image_points.append(image_points[(index*2)])
        temp_image_points.append(image_points[(index*2)+1])
        temp_key_points.append(keypoints[index])
    
    temp_image_points = np.array(temp_image_points)

    return temp_points, temp_image_points, temp_key_points





#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>       TEST        <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#