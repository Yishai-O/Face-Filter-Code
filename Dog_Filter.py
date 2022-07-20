import cv2
import numpy as np 

path = './'

#get facial classifiers
face_cascade = cv2.CascadeClassifier(path +'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(path +'haarcascade_eye.xml')

#read images
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
if __name__ == "__main__":
    use_dog()