import cv2
import numpy as np
import pickle
import datetime

# Recognize student attendance live from webcam using trained LBPH model
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

    # Dictionary to keep track of attendance to avoid duplicates
    attendance_marked = {}

    # Start webcam capture
    cap = cv2.VideoCapture(0)

    print("Starting attendance recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for detection

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        # Process each detected face
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]

            # Predict the label and confidence score
            label, confidence = recognizer.predict(face_roi)

            # Lower confidence means better match; tune threshold accordingly (e.g., 50)
            if confidence < 55:
                student_id = reverse_label_map[label]

                # Mark attendance if not already marked
                if student_id not in attendance_marked:
                    attendance_marked[student_id] = datetime.datetime.now()
                    print(f"Attendance marked for {student_id} at {attendance_marked[student_id]}")

                # Draw a rectangle and label around recognized face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{student_id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                # Unknown face - draw red rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Show live video frame with annotations
        cv2.imshow('Attendance Recognition', frame)

        # Quit recognition loop on pressing 'q'
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            print("Quitting attendance recognition.")
            break

    cap.release()
    cv2.destroyAllWindows()

    # Return attendance dictionary with student IDs and timestamps
    return attendance_marked


if __name__ == "__main__":
    HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'  # Face detector XML
    MODEL_PATH = './model/trained_lbph_model.yml'               # Trained LBPH model
    LABELS_PATH = './model/label_map.pkl'                        # Saved label mapping

    attendance = recognize_attendance(HAAR_CASCADE_PATH, MODEL_PATH, LABELS_PATH)
    
    print("\nFinal Attendance Records:")
    for student_id, timestamp in attendance.items():
        print(f"{student_id}: {timestamp}")
