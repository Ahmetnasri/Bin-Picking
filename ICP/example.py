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
#import draw


faulthandler.enable()


def draw_coordinate(img, z,xx,yy):

    point = (350,280)

    img = cv.circle(img, point,1, (255,255,0), 5)

    #cv2.imshow('fas',img)
    #cv2.waitKey(0)
    h,w,r = img.shape
    #print(h)
    #print(w)
    axis = (int(h/10) , int(w/10))
    #print(axis[0])
    #print(point+axis)
    point2 = (int(xx)+axis[0], int(yy))
    point3 = (int(xx), int(yy)+axis[1])
    #print(point2)


    cos = math.cos(z)
    #print(cos)

    x = cos * 40
    #print(x)

    sin = math.sin(z)
    #print(sin)

    y = sin * 40
    #print(y)


    cos1 = math.cos(math.pi/2-z)
    #print(cos1)

    x1 = cos1 * 40
    #print(x1)

    sin1 = math.sin(math.pi/2-z)
    #print(sin1)

    y1 = sin1 * 40
    #print(y1)


    #print(math.pi/2)

    point4 = (int(xx)+int(x), int(yy)+int(y))
    point5 = (int(xx)-int(y1), int(yy)+int(x1))
    point6 = (int(xx)-int(y), int(yy)+int(x))
    point1 = (int(xx),int(yy))
    #print('point1')
    #print(point1)
    img = cv.line(img, point1, point2, (0,0,255), 5)
    img = cv.line(img, point1, point3, (0,0,255), 5)

    img = cv.line(img, point1, point4, (255,0,0), 5)
    img = cv.line(img, point1, point6, (255,0,0), 5)
    #cv.imshow('img',img)
    #cv.waitKey(0)
    return img


def get3d(img,x,y):
    FLOOR_DEPTH = None

    dirname = os.path.dirname(__file__) #Path to the this file
    #dirname = 'C:/Users/ahmet/Desktop/Bin Picking Project/Baraa/'
    #Using the old function because I am not using a real camera. These images are saved directly from the CAD-Program
    intrinsic_camera_matrix = get_camera_intrinsic_matrix_old(camera_width_pixels = 640, camera_hight_pixels = 480, optical_center_x = 604.591, optical_center_y = 604.591, s=0, px= 326.605, py = 235.312)
    4
    #>>>>>>>>  Please use this function for calculating eh intrinsic camera matrix if you are using a real camera  <<<<<<<<<<<<
    #intrinsic_camera_matrix = get_camera_intrinsic_matrix(path_to_calibration_images)

   
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>       Comparing New Image With Saved Example      <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    #Path to description file
    des_file = os.path.join(dirname, "Des.dat")
    #print('ss')
    #print(des_file)
    #print(dirname)
    #print('aa')

    #Path to test image
    #img_path = os.path.join(dirname, "Dice/Test.png")
    #img_path = os.path.join(dirname, "Test_X_Rotation.png")
    #img_path = os.path.join(dirname, "newer_data/0.png")

    #img = cv.imread(img_path)

    #Showing test image
    #cv.imshow("Test Foto", img)

    #cv.waitKey(0)

    #cv.destroyAllWindows()

    #Extracting matching keypoints and there 3D-coordinates with the best matching ratio
    points, image_points, keypoints = compare_image_with_reference_data(des_file, img, matching_threshold = 0.9)

    if len(points) > 4:
        for i in range(85, 10, -10):
            t_points, t_image_points, t_keypoints = compare_image_with_reference_data(des_file, img, matching_threshold = i/100.0)

            if len(t_points) > 6:
                points = t_points
                image_points = t_image_points
                keypoints = t_keypoints
                #print("With matching threshold " + str(i/100.0) + ": " + str(len(t_points)) + " were found")

            else:
                break


    #Showing test image with the found matching keypoints
    img_with_importantPoints = cv.drawKeypoints(img, keypoints, img, color=(0, 200, 0))

    #plt.imshow(img_with_importantPoints)
    #plt.show()

    #print("Number of unique matching points found is: " + str(len(points)))



    #______________________________________________________________________________________________________________________________________
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>       Applying ICP To Estimate Pose       <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    #Guessing the pose
    #success, rotation_vector, translation_vector = cv2.solvePnP(np.array(points), np.array(image_points), intrinsic_camera_matrix, np.zeros((4,1)), flags=cv.SOLVEPNP_ITERATIVE)


    point_np = np.array(points)


    _, rotation_vector, translation_vector, _ = cv.solvePnPRansac(objectPoints=point_np, imagePoints=np.array(image_points), cameraMatrix=intrinsic_camera_matrix, distCoeffs=np.zeros((4,1)), flags=cv.SOLVEPNP_ITERATIVE, confidence=0.9999999 ,reprojectionError=5)



    guess_world_to_object_vector = np.concatenate([translation_vector, rotation_vector])
    #print(guess_world_to_object_vector.shape)


    des_file = os.path.join(dirname, "Des.dat")
    color_points_des, color_points_coordinates, sobel_points_des, sobel_points_coordinates = read_data_from_description_file(des_file)
    all_points = np.concatenate([color_points_coordinates, sobel_points_coordinates])

    xx, yy, zz = separate_xyz(all_points)
    all_points = prep_points(xx, yy, zz)

    visualize_points(all_points)

    image_points = get_points_from_kp(keypoints)

    guessed_points, jacobian = cv.projectPoints(point_np, rotation_vector, translation_vector, intrinsic_camera_matrix, np.zeros((4,1)))
    guessed_points = order_points(guessed_points)
    error = calc_error(guessed_points, image_points)
    #print("\nThe relative Error is ---  " + str(error))
    img_copy = img.copy()
    kp_copy = keypoints.copy()
    i = 0
    for keyP in kp_copy:
        keyP.pt = (guessed_points[i], guessed_points[i+1])
        i += 2

    #print("\n\n-- The Transformation Vector is:\n")
    #print(guess_world_to_object_vector)

    visualize_points(all_points, guess_world_to_object_vector)

    i = cv.drawKeypoints(img_copy,kp_copy,img_copy,color=(255,0,0), flags=0)
    current_pose = "x=%.2f, y=%.2f, z=%.2f, Rx=%.2f, Ry=%.2f, Rz=%.2f"%(guess_world_to_object_vector[0][0], guess_world_to_object_vector[1][0], guess_world_to_object_vector[2][0], guess_world_to_object_vector[3][0], guess_world_to_object_vector[4][0], guess_world_to_object_vector[5][0])
    i = cv.putText(i, current_pose , (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv.LINE_AA) 

    #plt.imshow(i)
    #plt.show()


    xx = x
    yy = y

    a = draw_coordinate(i,guess_world_to_object_vector[5][0],xx,yy)
    return a

