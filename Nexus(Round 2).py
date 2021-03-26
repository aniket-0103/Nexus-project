import cv2
import numpy as np
import time

#Function to detect the nearest shape to the bot 
def detect_nearest_shape(contours):

    l = []                                              # Array of area of shapes detected of a suitable range
    conts = []                                          # Array of contours for the respective area array
    y_val = []                                          # Array of y value of center pixel of the contour array
    x_val = []                                          # Array of x value of center pixel of the contour array

# Finding the centers and area of all the contours detected in a suitable area range
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if (area > 2000 and area < 15000):

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                y_val = y_val + [cY]
                x_val = x_val + [cX]

                conts = conts + [cnt]
                if (grayimg[cY, cX] > 120):
                    l = l + [area]
            else:
                cX = 0
                cY = 0

    ind = y_val.index(max(y_val))        # Y value of the center pixel of the nearest shape

    return ind,max(y_val),l,conts,x_val

# Function to find the shape with a particular shape according to the barcode value.
def find_color(dec,img):

    imageFrame = img
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

# Masking the image and finding contours
    if dec%3 == 1:

        red_lower = np.array([156, 97, 121], np.uint8)
        red_upper = np.array([180, 255, 255], np.uint8)
        red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

        contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):

            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            cv2.putText(img, "Red", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

    elif dec%3 == 0:

        green_lower = np.array([25, 52, 72], np.uint8)
        green_upper = np.array([102, 255, 255], np.uint8)
        green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

        contours, hierarchy = cv2.findContours(green_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):

            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),(x + w, y + h),(0, 255, 0), 2)

            cv2.putText(img, "Green", (x, y),cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 255, 0))

    elif dec%3==2:

        blue_lower = np.array([94, 80, 2], np.uint8)
        blue_upper = np.array([120, 255, 255], np.uint8)
        blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

        contours, hierarchy = cv2.findContours(blue_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):

            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y),(x + w, y + h),(255, 0, 0), 2)

            cv2.putText(img, "Blue", (x, y),cv2.FONT_HERSHEY_SIMPLEX,1.0, (255, 0, 0))

# Function to read the barcode value
def read_barcode(l,cnts):

    bin = []                            # Array to represent the barcode value in binary form.
    # l.remove(max(l))
    avg = (max(l) + min(l)) / 2

    for i in range(len(l)):

        if (l[i] < avg):
            bin.append(0)

        else:
            bin.append(1)

    dec = 8 * bin[0] + 4 * bin[1] + 2 * bin[2] + bin[3]
    approx = cv2.approxPolyDP(cnts[0], 0.01 * cv2.arcLength(cnts[0], True), True)
    x = approx.ravel()[0]
    y = approx.ravel()[1]

    cv2.putText(img,"barcode" , (x - 10, y - 10), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 0, 0), 2)
    return dec

# Function to arrange the contours in the order of left to right in the screen
def get_contour_precedence(contour, cols):

    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

# Function to find the color of the detected shape
def detect_the_color(img, contour,cX,cY):

    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    x = approx.ravel()[0]
    y = approx.ravel()[1]
    c = img[cY,cX]

    if c[0]>c[1] and c[0]>c[2]:
        cv2.putText(img, "Blue circle", (x-10,y-10), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 0, 0),2)

    elif c[0] < c[1] and c[1] > c[2]:
        cv2.putText(img, "Green circle", (x-5, y-10), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 0, 0),2)

    elif c[0] < c[1] and c[1] > c[2]:
        cv2.putText(img, "Red circle", (x-10, y-10), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 0, 0),2)

# Reading the inpput video
cap = cv2.VideoCapture('D:\Mnit, Jaipur\Zine robotics\exus.mp4')
dec=None

#while (True):
while (cap.isOpened()):

    rat, img = cap.read()

    if img is None:
        break

    else:
        hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        #Canny edge detection
        imgc = img
        imgc = cv2.GaussianBlur(imgc, (7, 7), 0)
        imgc = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)
        imgc = cv2.Canny(imgc, 40, 100)

        kernel = np.ones((5, 5))
        imgc = cv2.dilate(imgc, kernel, 0)

        # Finding contours in the image
        contours, _ = cv2.findContours(imgc, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours.sort(key=lambda x: get_contour_precedence(x, imgc.shape[1]))

        y_ind, max_y, l, conts, x_val = detect_nearest_shape(contours)
        cv2.drawContours(img, conts, y_ind, (0, 0, 255), 5)

        # Finding the region of screen in which shape is detected and ordering the bot to take action accordingly
        if x_val[y_ind]>150 and x_val[y_ind]<200:
            print("Forward")

        elif x_val[y_ind]>200:
            print("Right")

        elif x_val[y_ind]<150:
            print("Left")

        #detect_the_color(img, conts[y_ind], x_val[y_ind], max_y)
        if len(l) == 4:
            dec = read_barcode(l,conts)

    if dec is not None:
        find_color(dec, img)

    cv2.imshow('image', img)
    time.sleep(0.04)
    key = cv2.waitKey(1)

    if key ==27 :
        break

print("Barcode value :", dec)

cv2.destroyAllWindows()
cap.release()

