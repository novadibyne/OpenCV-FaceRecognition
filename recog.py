#!/usr/bin/env python3

import cv2
import numpy as np
import face_recognition as fr

def loadandextract():
    #Load image and convert to RGB
    img_han1 = fr.load_image_file('opencv\Face Recog - Match Face\Han_Hyo_Joo_in_2017.jpg')
    img_han1_rgb = cv2.cvtColor(img_han1, cv2.COLOR_BGR2RGB)

    img_test = fr.load_image_file('opencv\Face Recog - Match Face\Han_Hyo_Joo_in_2013.jpg')
    img_test_rgb = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

    #Extract location of the images
    han1_loc = fr.face_locations(img_han1_rgb)[0]
    test_loc = fr.face_locations(img_test_rgb)[0]
    
    #Extract features of the images
    han_encode = fr.face_encodings(img_han1_rgb)[0]
    #test_encode = fr.face_encodings(img_han2_rgb)[0]
    test_encode = fr.face_encodings(img_test_rgb)[0]

    #Draw a rectange around the face
    cv2.rectangle(img_han1_rgb, (han1_loc[3],han1_loc[0]) ,(han1_loc[1],han1_loc[2]), (255,255,255), 3)
    #cv2.rectangle(img_han2_rgb, (face2_loc[3],face2_loc[0]) ,(face2_loc[1],face2_loc[2]), (255,255,255), 3)
    cv2.rectangle(img_test_rgb, (test_loc[3],test_loc[0]) ,(test_loc[1],test_loc[2]), (255,255,255), 3)

    #COmparison to see if the modelled and test image matches
    comparison_results = fr.compare_faces([han_encode], test_encode)
    #Calculate distance of face , the more the distance , the more the chance it wont maatch
    face_distance = fr.face_distance([han_encode], test_encode)
    print(comparison_results,face_distance)
    
    cv2.putText(img_test_rgb, f'{comparison_results}, {round(face_distance[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),3)
   
    #Display both images
    cv2.imshow('hanhyo1', img_han1_rgb)
    cv2.imshow('test', img_test_rgb)

    #Wait until keystroke
    cv2.waitKey(0)

    #return han_encode,test_encode,img_test_rgb


def compareimg(func):
    ##getting values for the loadandextract function
    img_lal_rgb, han_encode,test_encode = func()
    
    #COmparison to see if the modelled and test image matches
    comparison_results = fr.compare_faces([han_encode], test_encode)
    #Calculate distance of face , the more the distance , the more the chance it wont maatch
    face_distance = fr.face_distance([han_encode], test_encode)
    print(comparison_results,face_distance)
    
    cv2.putText(img_lal_rgb, f'{comparison_results}, round({face_distance[0]},2)',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),3)
   

#compareimg(loadandextract)    

loadandextract()