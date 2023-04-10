# from typing import Counter
import cv2
import numpy as np
import os

# web camera
cap = cv2.VideoCapture('videos/video.mp4')
min_width_rect = 80
min_height_rect = 80
count_line_position = 550
# Initialize Substractor
algo = cv2.bgsegm.createBackgroundSubtractorMOG()
iter = 1


def center_handle(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1

    return cx, cy


detect = []
ofset = 6  # allowable error between pixels
counter = 0

while True:
    ret, frame = cap.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    # applying on each frame
    img_sub = algo.apply(blur)#applying gaussian blur
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))#image dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))#kernel passing for contours
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    countershape, h = cv2.findContours(
        dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#finding contours

    cv2.line(frame, (25, count_line_position),
             (1200, count_line_position), (255, 127, 0), 3)

    for (i, c) in enumerate(countershape):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width_rect) and (h >= min_height_rect)
        if not validate_counter:
            continue

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "VEHICLE :"+str(counter), (x, y-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 244, 0), 2)

        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(frame, center, 4, (0, 0, 255), -1)

        for(x, y) in detect:
            if y < (count_line_position+ofset) and y > (count_line_position-ofset):
                counter += 1
                cv2.line(frame, (25, count_line_position),
                         (1200, count_line_position), (0, 127, 255), 3)
                detect.remove((x, y))
                print("VEHICLE COUNTER: "+str(counter))
                directory = r'F:\ANPR\ANPR\ANPR\image'#change this directory to your image folders path else it wont work
                os.chdir(directory)
                iterstring = str(iter)
                filename = 'savedImage'+iterstring+'.jpg'
                iter = iter+1
                # Using cv2.imwrite() method
                # Saving the image
                cv2.imwrite(filename, frame)
                # cv2.imshow('DETECTOR',dilatada)
                cv2.putText(frame, "VEHICLE COUNTER :"+str(counter), (450, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    cv2.imshow('Video Original', frame)

    if cv2.waitKey(1) == 13:
        break
cv2.destroyAllWindows()
cv2.release()
