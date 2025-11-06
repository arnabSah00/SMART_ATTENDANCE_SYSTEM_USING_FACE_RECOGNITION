import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
import joblib
import datetime
from collections import deque, Counter
import csv

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize MTCNN face detector
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=device)

# Load pretrained FaceNet model for embeddings
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load classifier and label encoder (trained on your dataset embeddings)
classifier = joblib.load('.\model\svm_classifier.joblib')
label_encoder = joblib.load('.\model\label_encoder.joblib')

# Attendance dictionary to store marked student IDs and timestamps
attendance = {}

# Buffer for temporal smoothing (store recent recognized IDs)
prediction_buffer = deque(maxlen=7)  # 7-frame buffer

# Threshold confidence to accept recognition
CONFIDENCE_THRESHOLD = 0.8

def mark_attendance(student_id):
    now = datetime.datetime.now()
    if student_id not in attendance:
        day = now.strftime("%A")               # E.g., Thursday
        date_part = now.strftime("%Y-%m-%d")   # E.g., 2025-11-06
        time_part = now.strftime("%H:%M:%S")   # E.g., 18:33:22
        attendance[student_id] = (day, date_part, time_part)
        print(f"Attendance marked for Student ID {student_id} on {day} at {date_part} {time_part}")

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face_tensor = mtcnn(img)

        if face_tensor is not None:
            face_tensor = face_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                embedding = facenet(face_tensor).cpu().numpy()

            # Predict probabilities and class
            probs = classifier.predict_proba(embedding)
            max_prob = np.max(probs)
            pred_idx = classifier.predict(embedding)[0]
            pred_id = label_encoder.inverse_transform([pred_idx])[0]

            if max_prob > CONFIDENCE_THRESHOLD:
                prediction_buffer.append(pred_id)
            else:
                prediction_buffer.append("Unknown")

            # Majority voting in buffer to smooth predictions
            most_common_pred = Counter(prediction_buffer).most_common(1)[0][0]

            if most_common_pred != "Unknown":
                mark_attendance(most_common_pred)
                label_text = f"ID: {most_common_pred} ({max_prob*100:.2f}%)"
                color = (0, 255, 0)
            else:
                label_text = "Unknown"
                color = (0, 0, 255)
        else:
            # No face detected
            prediction_buffer.append("Unknown")
            label_text = "No Face"
            color = (0, 0, 255)

        # Display label on frame
        cv2.putText(frame, label_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Show live video
        cv2.imshow("FaceNet Attendance (Smoothed)", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save attendance to CSV
    with open("attendance.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["StudentID", "Day", "Date", "Time"])
        for student_id, (day, date_part, time_part) in attendance.items():
            writer.writerow([student_id, day, date_part, time_part])

    print("Attendance saved to attendance.csv")

    ## create a object of attendance record for save in data base

if __name__ == "__main__":
    main()
