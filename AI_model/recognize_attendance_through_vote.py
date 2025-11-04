import cv2
import numpy as np
import pickle
import datetime
from collections import deque, Counter
import time


def recognize_attendance(haar_cascade_path, model_path, labels_path):
    """
    Parameters:
    - haar_cascade_path: Path to Haar Cascade XML face detector.
    - model_path: Path to the trained LBPH face recognition model.
    - labels_path: Path to the label map for ID <-> student reg_no.
    """

    # Initialize the face recognizer and load the trained model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)

    # Load label map to convert numeric label back to student registration number (string)
    with open(labels_path, 'rb') as f:
        label_map_data = pickle.load(f)
        reverse_label_map = label_map_data["reverse_label_map"]  # int label -> student reg_no

    # Track attendance: student_id -> timestamp
    attendance_marked = {}

    # Label history per detected face; key = face bounding box tuple (x,y,w,h)
    face_label_history = {}
    # Timestamp of last seen per face_key
    face_last_seen = {}

    max_history_len = 10
    min_votes_to_confirm = 2
    confidence_vote_threshold = 50
    max_recognize_wait_secs = 2

    # Start video capture
    cap = cv2.VideoCapture(0)

    print("Starting attendance recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        current_time = time.time()

        # Process faces
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]

            label, confidence = recognizer.predict(face_roi)

            face_key = (x, y, w, h)
            face_last_seen[face_key] = current_time

            if confidence < confidence_vote_threshold:
                if face_key not in face_label_history:
                    face_label_history[face_key] = deque(maxlen=max_history_len)

                face_label_history[face_key].append(label)

                common_label, count = Counter(face_label_history[face_key]).most_common(1)[0]

                if count >= min_votes_to_confirm:
                    student_id = reverse_label_map[common_label]

                    if student_id not in attendance_marked:
                        attendance_marked[student_id] = datetime.datetime.now()
                        print(f"Attendance marked for {student_id} at {attendance_marked[student_id]}")

                    # Draw green rectangle and student ID
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{student_id}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                else:
                    first_seen_time = face_last_seen.get(face_key, current_time)
                    time_waited = current_time - first_seen_time
                    if time_waited > max_recognize_wait_secs:
                        # Too long to recognize, mark unknown
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, "Unknown", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        # Still stabilizing recognition
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                        cv2.putText(frame, "Recognizing...", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            else:
                # Low confidence — clear history and mark unknown
                if face_key in face_label_history:
                    del face_label_history[face_key]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Remove face histories for faces not seen in last 3 seconds
        to_remove = [key for key, last_seen in face_last_seen.items() if current_time - last_seen > 3]
        for key in to_remove:
            face_label_history.pop(key, None)
            face_last_seen.pop(key, None)

        cv2.imshow('Attendance Recognition', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("Quitting attendance recognition.")
            break

    cap.release()
    cv2.destroyAllWindows()

    return attendance_marked


if __name__ == "__main__":
    HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'  # Face detector XML path
    MODEL_PATH = './model/trained_lbph_model.yml'              # Trained LBPH model path
    LABELS_PATH = './model/label_map.pkl'                       # Label map path

    attendance = recognize_attendance(HAAR_CASCADE_PATH, MODEL_PATH, LABELS_PATH)

    print("\nFinal Attendance Records:")
    for student_id, timestamp in attendance.items():
        print(f"{student_id}: {timestamp}")
