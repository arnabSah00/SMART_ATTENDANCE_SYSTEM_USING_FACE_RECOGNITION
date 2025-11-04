##does not work don't believe on this code

import cv2
import numpy as np
import torch
from torchvision import transforms
import pickle      # For saving/loading label mappings and other data
import datetime    # For recording attendance timestamps

# Load your pre-trained liveness detection PyTorch model
class LivenessModel(torch.nn.Module):
    # Define your model architecture or load an existing one
    # For example purposes, we'll assume model is loaded externally
    def __init__(self):
        super(LivenessModel, self).__init__()
        # model definition
    
    def forward(self, x):
        # forward pass
        pass

def load_liveness_model(model_path):
    model = torch.load(model_path)
    model.eval()   # Set model to evaluation mode
    return model

def preprocess_face_for_liveness(face_image):
    """
    Prepares the face image for liveness model input.
    Resize, normalize and convert to tensor.
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),  # Assuming model expects 32x32 input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Example normalization
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(face_image).unsqueeze(0)  # Add batch dimension

def is_live_face(liveness_model, face_image_np, threshold=0.5):
    """
    Returns True if face is live, else False.
    """
    input_tensor = preprocess_face_for_liveness(face_image_np)
    with torch.no_grad():
        output = liveness_model(input_tensor)
        prob = torch.sigmoid(output).item()
        return prob > threshold

# Example Integration in attendance recognition workflow:
def recognize_with_liveness(haar_cascade_path, model_path, labels_path, liveness_model_path):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)

    # Load label map
    with open(labels_path, 'rb') as f:
        label_data = pickle.load(f)
        reverse_label_map = label_data["reverse_label_map"]

    # Load liveness model
    liveness_model = load_liveness_model(liveness_model_path)

    attendance_marked = {}
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_color = frame[y:y+h, x:x+w]  # face in BGR for liveness
            face_gray = gray[y:y+h, x:x+w]

            # Liveness check
            if is_live_face(liveness_model, face_color):
                label, confidence = recognizer.predict(face_gray)
                if confidence < 50:
                    student_id = reverse_label_map[label]
                    if student_id not in attendance_marked:
                        attendance_marked[student_id] = datetime.datetime.now()
                    color = (0, 255, 0)
                    text = student_id
                else:
                    color = (0, 0, 255)
                    text = "Unknown"
            else:
                color = (0, 255, 255)
                text = "Spoof Detected"
            
            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.imshow("Attendance with Liveness", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return attendance_marked
