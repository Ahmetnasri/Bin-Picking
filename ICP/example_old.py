import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ICP_With_SIFT import *
from Reading_Writing_Reference_Points import *
import time
import math
import random
import os
import faulthandler
faulthandler.enable()
#import draw_coordinate
def get3d(img):
    print(img.shape)
    FLOOR_DEPTH = None
    #dirname =
    dirname = os.path.dirname(__file__) #Path to the this file
    #dirname = 'C:/Users/ahmet/Desktop/Bin Picking Project/Baraa/'
    #Using the old function because I am not using a real camera. These images are saved directly from the CAD-Program
    intrinsic_camera_matrix = get_camera_intrinsic_matrix_old(camera_width_pixels = 640, camera_hight_pixels = 480, optical_center_x = 604.591, optical_center_y = 604.591, s=0, px= 326.605, py = 235.312)
    4
    #>>>>>>>>  Please use this function for calculating eh intrinsic camera matrix if you are using a real camera  <<<<<<<<<<<<
    #intrinsic_camera_matrix = get_camera_intrinsic_matrix(path_to_calibration_images)
    print('intrinsic')
    print(intrinsic_camera_matrix)
    dist = np.array([[0.13896223, -0.18744982,  0.00194218,  0.00593407, -0.25117682]])
    #_______________________________________________________________________________________________________________________________________
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>       Saving Points Example       <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    #Importing images

    #________________________________________________________________________________________________________________________________________
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>       Comparing New Image With Saved Example      <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    #Path to description file
    des_file = os.path.join(dirname, "Des.dat")

    #Path to test image
    img_path = os.path.join(dirname, "test4.png")
    #img_path = os.path.join(dirname, "newer_data/0.png")

    #img = cv.imread(img_path)

    #Showing test image
    cv.imshow("Test Foto", img)
    print(img.shape)
    cv.waitKey(0)

    cv.destroyAllWindows()

    #Extracting matching keypoints and there 3D-coordinates
    points, image_points, keypoints = compare_image_with_reference_data(des_file, img_path, matching_threshold = 0.9)


    #Showing test image with the found matching keypoints
    img_with_importantPoints = cv.drawKeypoints(img, keypoints, img, color=(0, 200, 0))

    #plt.imshow(img_with_importantPoints)
    #plt.show()

    print("Number of unique matching points found is: " + str(len(points)))



    #______________________________________________________________________________________________________________________________________
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>       Applying ICP To Estimate Pose       <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    #Guessing the pose

    _, rotation_vector, translation_vector, _ = cv.solvePnPRansac(objectPoints=np.array(points), imagePoints=np.array(image_points), cameraMatrix=intrinsic_camera_matrix, distCoeffs=np.zeros((4,1)), flags=cv.SOLVEPNP_ITERATIVE, confidence=0.9999999999 ,reprojectionError=0.0001)

    guess_world_to_object_vector = np.concatenate([translation_vector, rotation_vector])

    # Importing all points to apply the estimated pose to them (Just for displaying the result)
    des_file = os.path.join(dirname, "Des.dat")
    color_points_des, color_points_coordinates, sobel_points_des, sobel_points_coordinates = read_data_from_description_file(des_file)
    all_points = np.concatenate([color_points_coordinates, sobel_points_coordinates])

    # Visualizing all points in their original pose x=0, y=0, z=0, Rx=0, Ry=0, Rz=0
    xx, yy, zz = separate_xyz(all_points)
    all_points = prep_points(xx, yy, zz)
    visualize_points(all_points)

    # Extracting the pixel coordinates from the found keypoints
    image_points = get_points_from_kp(keypoints)

    # Projecting the points using the estimated pose
    guessed_points, jacobian = cv.projectPoints(np.array(points), rotation_vector, translation_vector, intrinsic_camera_matrix, np.zeros((4,1)))

    # Calculating the error
    guessed_points = order_points(guessed_points)
    error = calc_error(guessed_points, image_points)
    print("\nThe relative Error is ---  " + str(error))


    img_copy = img.copy()
    kp_copy = keypoints.copy()
    i = 0
    for keyP in kp_copy:
        keyP.pt = (guessed_points[i], guessed_points[i+1])
        i += 2

    print("\n\n-- The Transformation Vector is:\n")
    print(guess_world_to_object_vector)

    # Visualizing the 3D-points after applying the estimated pose
    #visualize_points(all_points, guess_world_to_object_vector)


    # Showing the image with the projected points and the pose
    i = cv.drawKeypoints(img_copy,kp_copy,img_copy,color=(255,0,0), flags=0)
    current_pose = "x=%.2f, y=%.2f, z=%.2f, Rx=%.2f, Ry=%.2f, Rz=%.2f"%(guess_world_to_object_vector[0][0], guess_world_to_object_vector[1][0], guess_world_to_object_vector[2][0], guess_world_to_object_vector[3][0], guess_world_to_object_vector[4][0], guess_world_to_object_vector[5][0])
    i = cv.putText(i, current_pose , (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv.LINE_AA) 

    #plt.imshow(i)
    #plt.show()


    #r = (guess_world_to_object_vector[3][0], guess_world_to_object_vector[4][0], guess_world_to_object_vector[5][0])
    #t = (guess_world_to_object_vector[0][0], guess_world_to_object_vector[1][0], guess_world_to_object_vector[2][0])
    #t = (0.0, 0.0 , 0.0)
    #point = (320,240)
    #draw_coordinate.draw_System(intrinsic_camera_matrix, dist, img ,point,r, t)