##############################
#ALL THE MODULES THAT WE ARE GOING TO USE.
import cv2 
import matplotlib.pyplot as plt 
import pytesseract 

################################
# DOWNLOADS REQUIRED:

# 1] https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_russian_plate_number.xml

# 2]https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.0.0-alpha.20210811.exe

################################
#ALL THE SET PARAMETERES AND CONSTANT VALUE.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
fw = 640
fh = 480
cascade= cv2.CascadeClassifier("F:\cvTej\haarcascade_russian_plate_number.xml")#this is our trained model most commonly used in image processing for object detection and tracking, primarily facial detection and recognition.
minarea = 500
color = (255,0,255)
#C:\Program Files\Tesseract-OCR
################################

# CODE STARTS FROM HERE.

cap = cv2.VideoCapture(0)# TELL WHICH CAMERA TO USE
# ADJUSTING THE SCREEN AND BRIGHTNESS
cap.set(3,fw)
cap.set(4,fh)
cap.set(10,150)

# SATRT TO READ THE VIDEO.
while True:
    success, img = cap.read()

    imggry = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#CONVERTING THE ORIGINAL BGR VIDEO TO GRAY SCALE FOR DETECTION

    numberplate = cascade.detectMultiScale(imggry,1.1,4)#cascade function to detect images of different size

    for(x,y,w,h) in numberplate:

        area = w*h

        if area>minarea:#FILTER TO DETECT THE NUMBER PLATE

            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)#IF NUMBER PALTE IS DETECTED THEN PUT A RECTANGLE AROUND IT.

            cv2.putText(img,"number plate",(x,y-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color,2)

            imgroi = img[y:y+h,x:x+w]#CORPING THE IMAGE AND GETING THE IMAGE OF REGION OF INTREST.

            # cv2.imshow("roi",img)
            width = int(imgroi.shape[1] * 150 / 100)#SCALING THE WIDTH OF IMAGE

            height = int(imgroi.shape[0] * 150 / 100)#SCALING THE HEIGHT OF IMAGE

            dim = (width, height)

            resized_image = cv2.resize(imgroi, dim, interpolation = cv2.INTER_AREA)#RESIZING THE IMGROI IMAGE

            cv2.imshow("reimg",resized_image)#DISPLAYING THE RESIZED IMAGE

            resized_image_gray = cv2.cvtColor(resized_image,cv2.COLOR_BGR2GRAY)#AGAIN CONVERTING THE RESIZED IMAGE TO GRAY SCALE FOR TEXT DTECTION.

            blurimg = cv2.medianBlur(resized_image_gray,3)#CLEARING AND DE NOISING THE RESIZED GRAY IMAGE USING BLUR

            cv2.imshow("blurimage",resized_image_gray)#DISPLAYING THE GRAY IMAGE

            print(pytesseract.image_to_string(blurimg, config = f'--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))

    
    cv2.imshow("result",img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
    
   
