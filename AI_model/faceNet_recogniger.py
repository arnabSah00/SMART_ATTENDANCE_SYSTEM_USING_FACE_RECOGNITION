import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
import joblib
import datetime
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load classifier and label encoder
classifier = joblib.load('.\model\svm_classifier.joblib')
label_encoder = joblib.load('.\model\label_encoder.joblib')

attendance = {}

def mark_attendance(student_id):
    now = datetime.datetime.now()
    if student_id not in attendance:
        attendance[student_id] = now.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Attendance marked for Student ID {student_id} at {attendance[student_id]}")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    face = mtcnn(img)
    if face is not None:
        face_cropped = face.unsqueeze(0).to(device)
        embedding = facenet(face_cropped).detach().cpu().numpy()

        probs = classifier.predict_proba(embedding)
        prob = np.max(probs)
        pred_int = classifier.predict(embedding)[0]
        pred_id = label_encoder.inverse_transform([pred_int])[0]

        if prob > 0.7:  # Confidence threshold
            mark_attendance(pred_id)
            label_text = f"ID: {pred_id} ({prob*100:.2f}%)"
            color = (0, 255, 0)
        else:
            label_text = "Unknown"
            color = (0, 0, 255)
    
        cv2.putText(frame, label_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('FaceNet Attendance', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save attendance to CSV file
with open("attendance.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["StudentID", "Timestamp"])
    for student_id, timestamp in attendance.items():
        writer.writerow([student_id, timestamp])

print("Attendance saved to attendance.csv")
