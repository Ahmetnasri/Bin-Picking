import cv2 
import numpy as np 
import math


def draw_coordinate(img, z):
    #img = cv2.imread('test.png')
    #cv2.imshow('fas',img)
    #cv2.waitKey(0)

    point = (350,280)

    img = cv2.circle(img, point,1, (255,255,0), 5)

    #cv2.imshow('fas',img)
    #cv2.waitKey(0)
    h,w,r = img.shape
    #print(h)
    #print(w)
    axis = (int(h/10) , int(w/10))
    #print(axis[0])
    #print(point+axis)
    point2 = (350+axis[0], 280)
    point3 = (350, 280+axis[1])
    #print(point2)


    cos = math.cos(z)
    #print(cos)

    x = cos * 40
    print(x)

    sin = math.sin(z)
    #print(sin)

    y = sin * 40
    print(y)


    cos1 = math.cos(math.pi/2-z)
    #print(cos1)

    x1 = cos1 * 40
    print(x1)

    sin1 = math.sin(math.pi/2-z)
    #print(sin1)

    y1 = sin1 * 40
    print(y1)


    print(math.pi/2)

    point4 = (350+int(x), 280+int(y))
    point5 = (350-int(y1), 280+int(x1))
    point6 = (350-int(y), 280+int(x))

    img = cv2.line(img, point, point2, (0,0,255), 5)
    img = cv2.line(img, point, point3, (0,0,255), 5)

    img = cv2.line(img, point, point4, (255,0,0), 5)
    img = cv2.line(img, point, point6, (255,0,0), 5)



    #cv2.imshow('fas',img)
    #cv2.waitKey(0)