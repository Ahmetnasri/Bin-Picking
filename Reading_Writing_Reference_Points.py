import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
#from ICP_With_SIFT import *
import ICP_With_SIFT
import time
import math
import random
import pyrealsense2 as rs
#>>>>> Descrition of this script
#      This script is for exporting and importing description data for a specific object. It can be used for exporting keypoints 
#      that were calculated using SIFT to create a description String that can be saved in form of a text file.
#      And to extract these descriptions from the text file again

#>>>>> This function creates a rotation matrix from given angles for the rotation around the 3-axis
def get_rotation_matrix(angles):

    alpha = (angles[0]/180) * math.pi
    beta = (angles[1]/180) * math.pi
    gamma = (angles[2]/180) * math.pi

    rotation_x = np.zeros((3,3))
    rotation_x[0][0] = 1.0
    rotation_x[1][1] = math.cos(alpha)
    rotation_x[1][2] = -math.sin(alpha)
    rotation_x[2][1] = math.sin(alpha)
    rotation_x[2][2] = math.cos(alpha)

    rotation_y = np.zeros((3,3))
    rotation_y[1][1] = 1.0
    rotation_y[0][0] = math.cos(beta)
    rotation_y[0][2] = math.sin(beta)
    rotation_y[2][0] = -math.sin(beta)
    rotation_y[2][2] = math.cos(beta)

    rotation_z = np.zeros((3,3))
    rotation_z[2][2] = 1.0
    rotation_z[0][0] = math.cos(gamma)
    rotation_z[0][1] = -math.sin(gamma)
    rotation_z[1][0] = math.sin(gamma)
    rotation_z[1][1] = math.cos(gamma)

    rotation_matrix = rotation_z.dot(rotation_y.dot(rotation_x))

    return rotation_matrix

#>>>>> To keep a detailed description of the object we use 6 images from the six different faces of the object. Here are the supposed
#      rotation angles for all faces in respect to the top view. These should be modified in case the user wanted to define his/her 
#      own rotation angles for a specific image
"""rotation_angles_Top = (0, 0, 0)

rotation_angles_Bottom = (180, 0, 0)

rotation_angles_Left_Side = (-90, 0, 0)

rotation_angles_Right_Side = (90, 0, 0)

rotation_angles_Back_Side = (0, 90, 0)

rotation_angles_Front_Side = (0, -90, 0)"""


rotation_angles_Top = (0, 0, 0)

rotation_angles_Bottom = (0, -180, 0)

rotation_angles_Left_Side = (0, -90, 0)

rotation_angles_Right_Side = (0, 90, 0)

rotation_angles_Back_Side = (-90, 0, 0)

rotation_angles_Front_Side = (90, 0, 0)

rotation_matrises = [
        get_rotation_matrix(rotation_angles_Top), 
        get_rotation_matrix(rotation_angles_Bottom), 
        get_rotation_matrix(rotation_angles_Left_Side), 
        get_rotation_matrix(rotation_angles_Right_Side), 
        get_rotation_matrix(rotation_angles_Back_Side),
        get_rotation_matrix(rotation_angles_Front_Side)
    ]


def get_object_centers(images, bag_paths):
    
    centers = []

    for i in range(len(images)):
        img = images[i]
        depth_frame = 0
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            rs.config.enable_device_from_file(config, bag_paths[i])

            profile = pipeline.start(config)

            align_to = rs.stream.color
            align = rs.align(align_to)

            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            depth_frame = frames.get_depth_frame()

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            object_part = np.array([gray > 0])
            object_part = object_part.reshape((object_part.shape[1], object_part.shape[2]))
            object_part = object_part*255
            object_part = object_part.astype(np.uint8)

            temp_object_part = np.argwhere(object_part != 0)
            temp_object_part = np.hsplit(temp_object_part, [1, 1])

            minY = min(temp_object_part[0])[0]

            maxY = max(temp_object_part[0])[0]

            minX = min(temp_object_part[2])[0]

            maxX = max(temp_object_part[2])[0]

            centerY = int((maxY-minY)/2)
            centerX = int((maxX-minX)/2)

            object_center = [centerX, centerY]

            object_part = object_part[minY:maxY, minX:maxX]

            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

            depth_value = depth_frame.get_distance(object_center[0], object_center[1])
            center_point = rs.rs2_deproject_pixel_to_point(depth_intrin, object_center, depth_value)

            pipeline.stop()

            centers.append(center_point)
        
        except:
            print("An Error occured while trying to find object center")

    return centers



def get_depth(bag_path, point = None):
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    rs.config.enable_device_from_file(config, bag_path)

    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    frames = pipeline.wait_for_frames()
    frames = align.process(frames)

    depth_frame = frames.get_depth_frame()
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

    depth_value = 0
    
    if not point:
        i = 0
        j = 0

        x_changed = False
        p = [int(depth_intrin.ppx), int(depth_intrin.ppy)]
        depth_value = 0.0
        while True:

            depth_value = depth_frame.get_distance(p[0]+i, p[1]+j)

            if i > 10 and j > 10:
                depth_value = None
                break

            if depth_value != 0.0:
                FLOOR_DEPTH = depth_value
                break
            
            elif x_changed:
                j += 1
                x_changed = False

            else:
                i += 1
                x_changed = True

    pipeline.stop()
    return depth_value


def get_full_description (images, object_centers, bag_paths, floor_depth, octaves = [8],NFeatures = 100):

    descrition_data = ""

    for octave in octaves:

        for k in range(len(images)):
            try:
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.depth, rs.format.z16, 30)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
                rs.config.enable_device_from_file(config, bag_paths[k])
                pipeline.start(config)

                align_to = rs.stream.color
                align = rs.align(align_to)

                frames = pipeline.wait_for_frames()
                frames = align.process(frames)
                
                # Get depth frame
                depth_frame = frames.get_depth_frame()

                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

                

                kp, des = ICP_With_SIFT.apply_sift(images[k], nFeatures= NFeatures, nOctaves=octave)

                for i in range(len(des)):
                    name = str(k)
                    x = int(kp[i].pt[0])
                    y = int(kp[i].pt[1])

                    depth_pixel = [x, y]
                    depth_value = depth_frame.get_distance(x, y)
                    point = None

                    if depth_value != 0:
                        point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_value)

                    if point:

                        if point[2] < (floor_depth + 0.001):
                            for j in range(len(des[i])):
                                descrition_data += str(des[i][j])
                                if j != len(des[i]) - 1:
                                    descrition_data += ","
                            
                            descrition_data += "//"

                            point[2] = -(floor_depth - point[2])/2.0

                            transformed_point = rotation_matrises[k].dot(point)

                            #transformed_point[0] = transformed_point[0] - object_centers[k][0]
                            #transformed_point[1] = transformed_point[1] - object_centers[k][1]

                            print("K: " + str(k) + "\tPoint: " + str(point) + "\t Transformed: " + str(transformed_point))

                            descrition_data += str(transformed_point[0]) + "x" + str(transformed_point[1]) + "x" + str(transformed_point[2])
                        else:
                            point = None

                    if i != len(des) - 1 and point:

                        descrition_data += "Point_Description_End"

                if descrition_data[-1] == "d":
                    descrition_data = descrition_data[0:(len(descrition_data) - (len("Point_Description_End") + 1))]

                if k != len(images) - 1:
                    descrition_data += "Image_Descrition_End"

                else:

                    descrition_data += "*****"
            finally:
                pipeline.stop()

        for img in images:
            img = ICP_With_SIFT.apply_sobel(img)

        for k in range(len(images)):
            try:
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.depth, rs.format.z16, 30)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
                rs.config.enable_device_from_file(config, bag_paths[k])
                pipeline.start(config)

                align_to = rs.stream.color
                align = rs.align(align_to)

                frames = pipeline.wait_for_frames()
                frames = align.process(frames)
                
                # Get depth frame
                depth_frame = frames.get_depth_frame()

                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

                kp, des = ICP_With_SIFT.apply_sift(images[k], nFeatures= NFeatures, nOctaves=octave)

                for i in range(len(des)):
                    name = str(k)
                    x = int(kp[i].pt[0])
                    y = int(kp[i].pt[1])

                    depth_pixel = [x, y]
                    depth_value = depth_frame.get_distance(x, y)
                    point = None
                    if depth_value != 0:
                        point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_value)

                    if point:

                        if point[2] < (floor_depth + 0.001):
                            for j in range(len(des[i])):
                                descrition_data += str(des[i][j])
                                if j != len(des[i]) - 1:
                                    descrition_data += ","

                            descrition_data += "//"

                            point[2] = -(floor_depth - point[2])/2.0
                            
                            transformed_point = rotation_matrises[k].dot(point)

                            #transformed_point[0] = transformed_point[0] - object_centers[k][0]
                            #transformed_point[1] = transformed_point[1] - object_centers[k][1]

                            print("K: " + str(k) + "\tPoint: " + str(point) + "\t Transformed: " + str(transformed_point))
                            
                            descrition_data += str(transformed_point[0]) + "x" + str(transformed_point[1]) + "x" + str(transformed_point[2])
                        
                        else:
                            point = None
                    
                    if i != len(des) - 1 and point:

                        descrition_data += "Point_Description_End"

                if descrition_data[-1] == "d":
                    descrition_data = descrition_data[0:(len(descrition_data) - (len("Point_Description_End") + 1))]

                if k != len(images) - 1:

                    descrition_data += "Image_Descrition_End"

            finally:
                pipeline.stop()

    return descrition_data


#>>>>> This function extracts points description and points coordinates from a string
def split_data(images_des_string):
    images_des = images_des_string.split("Image_Descrition_End")

    points = []

    for image_des in images_des:

        temp_des = image_des.split("Point_Description_End")

        points += temp_des

    points_des = []

    points_coordinates = []
    ii = 0
    for point in points:
        #print(point)
        split_des_coordinates = point.split("//")
        
        temp_des = np.array(split_des_coordinates[0].split(',')).astype(np.float32)
    
        points_des.append(temp_des)

        split_xyz = split_des_coordinates[1].split('x')

        temp_coordinates = (float(split_xyz[0]), float(split_xyz[1]), float(split_xyz[2]))

        points_coordinates.append(temp_coordinates)

        ii += 1

    points_des = np.array(points_des)

    return points_des, points_coordinates

#>>>>> This function processes the description file and returns description of all points and there 3D-coordinates
def read_data_from_description_file(des_file):

    des_file = open(des_file,"r")

    points_des = des_file.readline()

    split_color_sobel = points_des.split("*****")

    color_points_des, color_points_coordinates = split_data(split_color_sobel[0])

    sobel_points_des = np.array([])
    sobel_points_coordinates = []

    if len(split_color_sobel) > 1:

        sobel_points_des, sobel_points_coordinates = split_data(split_color_sobel[1])

    return color_points_des, color_points_coordinates, sobel_points_des, sobel_points_coordinates

#>>>>> This function do the actual comparision between a given image and the saved reference points and returns keypoints, 3D-coordinates 
#      and pixel-coordinates of the matching points.
def compare_image_with_reference_data(path_to_reference_data, img, matching_threshold = 0.9):

    color_points_des, color_points_coordinates, sobel_points_des, sobel_points_coordinates = read_data_from_description_file(path_to_reference_data)

    #img = cv.imread(path_to_image)
    #cv.imshow('fsad',img)
    #cv.waitKey(0)
    color_kp, color_des = ICP_With_SIFT.apply_sift(img, nFeatures = 100, nOctaves = 8)

    color_good_matches, color_reference_index, color_des_index = ICP_With_SIFT.get_good_matches(color_points_des, color_des, threshold = matching_threshold, nMatches = 2)

    sobel = ICP_With_SIFT.apply_sobel(img)

    sobel_kp, sobel_des = ICP_With_SIFT.apply_sift(sobel, nFeatures = 100, nOctaves = 8)

    sobel_good_matches, sobel_reference_index, sobel_des_index = ICP_With_SIFT.get_good_matches(sobel_points_des, sobel_des, threshold = matching_threshold, nMatches = 2)

    color_points, color_image_points, color_keypoints = ICP_With_SIFT.get_matches_with_coordinates(color_points_coordinates, color_kp, color_reference_index, color_des_index)
    
    sobel_points, sobel_image_points, sobel_keypoints = ICP_With_SIFT.get_matches_with_coordinates(sobel_points_coordinates, sobel_kp, sobel_reference_index, sobel_des_index)

    points = color_points + sobel_points

    image_points = color_image_points + sobel_image_points

    keypoints = color_keypoints + sobel_keypoints

    points, image_points, keypoints = ICP_With_SIFT.get_unique_points(points, image_points, keypoints)

    return points, image_points, keypoints