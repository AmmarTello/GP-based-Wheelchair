import cv2

videoNumber = 3

videoPath = "Videos/"
videoName = ["1.avi", "2.avi", "3.avi", "4.avi"]

imagePath = "Videos/"
folderName = ["1", "2", "3", "4"]
extension = ".png"

videoFullName = videoPath + videoName[videoNumber]
cap = cv2.VideoCapture(videoFullName)

imageNumber = 5672
counter = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret != -1:
        imageName = "{:04d}.png".format(imageNumber)
        imageFullName = imagePath + folderName[videoNumber] + "/" + imageName
        cv2.imwrite(imageFullName, frame)
        counter = counter + 1
        print(counter)

        imageNumber = imageNumber + 1

cap.release()
cv2.destroyAllWindows()
