import teachable_machine
import cv2 as cv

model = teachable_machine.TeachableMachine(model_path="keras_model.h5",
                                           labels_file_path="labels.txt")
cap = cv.VideoCapture(1)


def classify_face(img, x, y, w, h):
    face = img[y:y + h, x:x + w]
    face = cv.resize(face, (224, 224))
    face = cv.cvtColor(face, cv.COLOR_RGB2GRAY)
    cv.imwrite("face.jpg", face)
    result = model.classify_image("face.jpg")
    # print(result)
    cv.putText(img, f"{result['class_name']} - {result['class_confidence'] * 100} %", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)


def detect_bounding_box(vid):
    gray_image = cv.cvtColor(vid, cv.COLOR_BGR2GRAY)
    face_classifier = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
        classify_face(vid, x, y, w, h)
        # cv.putText(vid, "test", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return faces


while True:
    _, img = cap.read()
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = detect_bounding_box(img)
    cv.imshow(
        "My Face Detection Project", img
    )  # display the processed frame in a window named "My Face Detection Project"
    k = cv.waitKey(1)
    if k == 27:  # Press ESC to close the camera view
        break

cap.release()
cv.destroyAllWindows()
