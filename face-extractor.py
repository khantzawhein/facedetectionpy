import cv2 as cv

cap = cv.VideoCapture(0)

serial = 0

while True:
    _, img = cap.read()
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_classifier = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
        # extract the face from the image
        face = gray_image[y:y + h, x:x + w]
        # resize the face to 224x224 pixels
        face = cv.resize(face, (224, 224))
        # save the face to a file with serial
        cv.imwrite(f"./data/face-{serial}.jpg", face)
        serial += 1

    cv.imshow(
        "My Face Detection Project", img
    )  # display the processed frame in a window named "My Face Detection Project"
    k = cv.waitKey(500)
    if k == 27:  # Press ESC to close the camera view
        break

cap.release()
cv.destroyAllWindows()