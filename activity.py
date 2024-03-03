import cv2

boy = cv2.imread('boy.jpg')
grey = cv2.cvtColor(boy,cv2.COLOR_BGR2GRAY)
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detect = classifier.detectMultiScale(grey)
for x,y,w,h in detect:
    cv2.rectangle(boy,(x,y),(x+w,y+h),(255,0,0),2)
#cv2.imshow('Image',boy)
    
video = cv2.VideoCapture(0)
loadclassifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret,frame = video.read()
    greyImage = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    scale = loadclassifier.detectMultiScale(greyImage)
    for x,y,w,h in scale:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow('Image',frame)
    cv2.waitKey(0)
