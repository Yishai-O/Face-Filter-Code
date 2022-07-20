import cv2
import numpy as np 

path = './'

#get facial classifiers
face_cascade = cv2.CascadeClassifier(path +'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(path +'haarcascade_eye.xml')

#read images
def use_cat():
    img = cv2.imread('color.jpg')
    cat = cv2.imread('cat.png')

    #get shape of cat
    original_cat_h,original_cat_w,cat_channels = cat.shape

    #get shape of img
    img_h,img_w,img_channels = img.shape

    #convert to gray
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cat_gray = cv2.cvtColor(cat, cv2.COLOR_BGR2GRAY)

    #create mask and inverse mask of cat
    #Note: I used THRESH_BINARY_INV because my image was already on 
    #transparent background, try cv2.THRESH_BINARY if you are using a white background
    ret, original_mask = cv2.threshold(cat_gray, 60, 255, cv2.THRESH_BINARY_INV)
    original_mask_inv = cv2.bitwise_not(original_mask)
    #find faces in image using classifier
    faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
    for (x,y,w,h) in faces:
        #retangle for testing purposes
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        #coordinates of face region
        face_w = w
        face_h = h
        face_x1 = x
        face_x2 = face_x1 + face_w
        face_y1 = y
        face_y2 = face_y1 + face_h

        #cat size in relation to face by scaling
        cat_width = int(1.1 * face_w)
        cat_height = int(cat_width * original_cat_h / (original_cat_w * 1.33))
        
        #setting location of coordinates of cat
        cat_x1 = face_x2 - int(face_w/2.1) - int(cat_width/2)
        cat_x2 = cat_x1 + cat_width
        cat_y1 = face_y1 - int(face_h*0.2)
        cat_y2 = cat_y1 + cat_height 

        #check to see if out of frame
        if cat_x1 < 0:
            cat_x1 = 0
        if cat_y1 < 0:
            cat_y1 = 0
        if cat_x2 > img_w:
            cat_x2 = img_w
        if cat_y2 > img_h:
            cat_y2 = img_h

        #Account for any out of frame changes
        cat_width = cat_x2 - cat_x1
        cat_height = cat_y2 - cat_y1

        #resize cat to fit on face
        cat = cv2.resize(cat, (cat_width,cat_height), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(original_mask, (cat_width,cat_height), interpolation = cv2.INTER_AREA)
        mask_inv = cv2.resize(original_mask_inv, (cat_width,cat_height), interpolation = cv2.INTER_AREA)

        #take ROI for cat from background that is equal to size of cat image
        roi = img[cat_y1:cat_y2, cat_x1:cat_x2]

        #original image in background (bg) where cat is not present
        roi_bg = cv2.bitwise_and(roi,roi,mask = mask)
        roi_fg = cv2.bitwise_and(cat,cat,mask=mask_inv)
        dst = cv2.add(roi_bg,roi_fg)

        #put back in original image
        img[cat_y1:cat_y2, cat_x1:cat_x2] = dst


    cv2.imwrite('cat.jpg',img) #display image
    cv2.waitKey(0) #wait until key is pressed to proceed
    cv2.destroyAllWindows() #close all windows
def use_bunny():
    img = cv2.imread('color.jpg')
    bunny = cv2.imread('bunny.png')

    #get shape of bunny
    original_bunny_h,original_bunny_w,bunny_channels = bunny.shape

    #get shape of img
    img_h,img_w,img_channels = img.shape

    #convert to gray
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bunny_gray = cv2.cvtColor(bunny, cv2.COLOR_BGR2GRAY)

    #create mask and inverse mask of bunny
    #Note: I used THRESH_BINARY_INV because my image was already on 
    #transparent background, try cv2.THRESH_BINARY if you are using a white background
    ret, original_mask = cv2.threshold(bunny_gray, 237, 255, cv2.THRESH_BINARY)
    original_mask_inv = cv2.bitwise_not(original_mask)
    #find faces in image using classifier
    faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
    for (x,y,w,h) in faces:
        #retangle for testing purposes
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        #coordinates of face region
        face_w = w
        face_h = h
        face_x1 = x
        face_x2 = face_x1 + face_w
        face_y1 = y
        face_y2 = face_y1 + face_h

        #bunny size in relation to face by scaling
        bunny_width = int(2.3 * face_w)
        bunny_height = int((bunny_width * original_bunny_h)/ (original_bunny_w * 0.5))
        
        #setting location of coordinates of bunny
        bunny_x1 = face_x2 - int(face_w/2) - int(bunny_width/2)
        bunny_x2 = bunny_x1 + bunny_width
        bunny_y1 = (face_y1) - int(face_h)
        bunny_y2 = bunny_y1 + bunny_height 

        #check to see if out of frame
        if bunny_x1 < 0:
            bunny_x1 = 0
        if bunny_y1 < 0:
            bunny_y1 = 0
        if bunny_x2 > img_w:
            bunny_x2 = img_w
        if bunny_y2 > img_h:
            bunny_y2 = img_h

        #Account for any out of frame changes
        bunny_width = bunny_x2 - bunny_x1
        bunny_height = bunny_y2 - bunny_y1

        #resize bunny to fit on face
        bunny = cv2.resize(bunny, (bunny_width,bunny_height), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(original_mask, (bunny_width,bunny_height), interpolation = cv2.INTER_AREA)
        mask_inv = cv2.resize(original_mask_inv, (bunny_width,bunny_height), interpolation = cv2.INTER_AREA)

        #take ROI for bunny from background that is equal to size of bunny image
        roi = img[bunny_y1:bunny_y2, bunny_x1:bunny_x2]

        #original image in background (bg) where bunny is not present
        roi_bg = cv2.bitwise_and(roi,roi,mask = mask)
        roi_fg = cv2.bitwise_and(bunny,bunny,mask=mask_inv)
        dst = cv2.add(roi_bg,roi_fg)

        #put back in original image
        img[bunny_y1:bunny_y2, bunny_x1:bunny_x2] = dst


    cv2.imwrite('bunny.jpg',img) #display image
    cv2.waitKey(0) #wait until key is pressed to proceed
    cv2.destroyAllWindows() #close all windows
def use_dog():
    img = cv2.imread('color.jpg')
    dog = cv2.imread('dog.png')

    #get shape of dog
    original_dog_h,original_dog_w,dog_channels = dog.shape

    #get shape of img
    img_h,img_w,img_channels = img.shape

    #convert to gray
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dog_gray = cv2.cvtColor(dog, cv2.COLOR_BGR2GRAY)

    #create mask and inverse mask of dog
    #Note: I used THRESH_BINARY_INV because my image was already on 
    #transparent background, try cv2.THRESH_BINARY if you are using a white background
    ret, original_mask = cv2.threshold(dog_gray, 200, 255, cv2.THRESH_BINARY)
    original_mask_inv = cv2.bitwise_not(original_mask)
    #find faces in image using classifier
    faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
    for (x,y,w,h) in faces:
        #retangle for testing purposes
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        #coordinates of face region
        face_w = w
        face_h = h
        face_x1 = x
        face_x2 = face_x1 + face_w
        face_y1 = y
        face_y2 = face_y1 + face_h

        #dog size in relation to face by scaling
        dog_width = int(2.75 * face_w)
        dog_height = int(dog_width * original_dog_h / (original_dog_w * 1.33))
        
        #setting location of coordinates of dog
        dog_x1 = face_x2 - int(face_w/1.75) - int(dog_width/2)
        dog_x2 = dog_x1 + dog_width
        dog_y1 = face_y1 - int(face_h*0.25)
        dog_y2 = dog_y1 + dog_height 

        #check to see if out of frame
        if dog_x1 < 0:
            dog_x1 = 0
        if dog_y1 < 0:
            dog_y1 = 0
        if dog_x2 > img_w:
            dog_x2 = img_w
        if dog_y2 > img_h:
            dog_y2 = img_h

        #Account for any out of frame changes
        dog_width = dog_x2 - dog_x1
        dog_height = dog_y2 - dog_y1

        #resize dog to fit on face
        dog = cv2.resize(dog, (dog_width,dog_height), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(original_mask, (dog_width,dog_height), interpolation = cv2.INTER_AREA)
        mask_inv = cv2.resize(original_mask_inv, (dog_width,dog_height), interpolation = cv2.INTER_AREA)

        #take ROI for dog from background that is equal to size of dog image
        roi = img[dog_y1:dog_y2, dog_x1:dog_x2]

        #original image in background (bg) where dog is not present
        roi_bg = cv2.bitwise_and(roi,roi,mask = mask)
        roi_fg = cv2.bitwise_and(dog,dog,mask=mask_inv)
        dst = cv2.add(roi_bg,roi_fg)

        #put back in original image
        img[dog_y1:dog_y2, dog_x1:dog_x2] = dst


    cv2.imwrite('dog.jpg',img) #display image
    cv2.waitKey(0) #wait until key is pressed to proceed
    cv2.destroyAllWindows() #close all windows