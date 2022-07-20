import cv2
import numpy as np 

path = './'

#get facial classifiers
face_cascade = cv2.CascadeClassifier(path +'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(path +'haarcascade_eye.xml')

#read images
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
if __name__ == "__main__":
    use_bunny()