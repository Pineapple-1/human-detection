import cv2 as cv

cap = cv.VideoCapture('data\humans.mp4')
cascade = cv.CascadeClassifier('haarfile.xml')

while (True):
    istrue,image = cap.read()
    # we have to make it gray 
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    # cascade classifier
    
    human = cascade.detectMultiScale(gray , 1.5,minNeighbors = 1)
    # if we select less neighbours we can have  more detections
    for (x,y,z,w) in human:
        cv.rectangle(image,(x,y),(x+z,y+w),(200,220,140),thickness=2)
        cv.imshow('human', image)
        
    if cv.waitKey(2) & 0xFF == 27:
      break
    
cap.release()
cv.destroyAllWindows()