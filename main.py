import cv2
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import time  # Import the time module
import datetime

# Initialize Firebase Admin SDK
cred = credentials.Certificate("serviceAccountKey.json")  # Replace with your service account key file
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://age-and-gender-detection-e4171-default-rtdb.firebaseio.com'  # Replace with your Firebase Realtime Database URL
})

def faceBox(faceNet, frame, db_ref):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []

    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bboxs.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    for bbox in bboxs:
        face = frame[max(0, bbox[1] - 20):min(bbox[3] + 20, frame.shape[0] - 1),
                     max(0, bbox[0] - 20):min(bbox[2] + 20, frame.shape[1] - 1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), [78.4263377603, 87.7689143744, 114.895847746], swapRB=False)
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]
        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = ageList[agePred[0].argmax()]
 
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        label = "{},{}".format(gender, age)
        cv2.rectangle(frame, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), (0, 255, 0), -1)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Send data to Firebase with a timestamp
        data = {
            'gender': gender,
            'age': age,
            'timestamp': current_time  # Generate a timestamp (milliseconds since epoch)
        }
        db_ref.push(data)  # Push data to the database

    return frame

faceProto = "opencv_face_detector.pbtxt" 
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel" #.caffemodel files containing pre trained age model
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel" #.caffemodel files containing pre trained gender model
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

video = cv2.VideoCapture(0)
while True:
    ret, frame = video.read()
    frame = faceBox(faceNet, frame, db.reference('people'))  # Pass the database reference
    cv2.imshow("Age-Gender", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
