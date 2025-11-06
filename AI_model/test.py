import os
import cv2
import pickle
import numpy as np
from scipy.spatial.distance import cosine
import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

# Paths and configs
DATASET_PATH = './dataset'  # Folder for images
EMBEDDINGS_PATH = './model/embeddings.pkl'
HAAR_PATH = 'haarcascade_frontalface_default.xml'

# Initialize FaceNet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Preprocessing transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Utility functions
def load_embeddings():
    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH, 'rb') as f:
            return pickle.load(f)
    else:
        return {}

def save_embeddings(data):
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(data, f)

def get_face_embedding(face_img):
    face_tensor = transform(face_img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = facenet(face_tensor).cpu().numpy()[0]
    return emb

def check_duplicate(embedding, embeddings_db, threshold=0.4):
    for reg_no, emb_list in embeddings_db.items():
        for db_emb in emb_list:
            dist = cosine(embedding, db_emb)
            if dist < threshold:
                return True, reg_no, dist
    return False, None, None

def create_student_folder(reg_no):
    folder_path = os.path.join(DATASET_PATH, reg_no)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def get_next_image_num(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.jpg')]
    nums = []
    for f in files:
        parts = f.split('_')
        if len(parts) > 1:
            try:
                num = int(parts[1].split('.')[0])
                nums.append(num)
            except:
                continue
    return max(nums)+1 if nums else 1

def register_face(reg_no):
    embeddings_db = load_embeddings()
    folder = create_student_folder(reg_no)

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(HAAR_PATH)

    print(f"Registering for: {reg_no}")
    captured = 0
    while captured < 50:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            cv2.imshow('Register', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Use largest face
        (x, y, w, h) = max(faces, key=lambda r: r[2]*r[3])
        face_img = frame[y:y+h, x:x+w]

        # Check for duplicate
        emb = get_face_embedding(face_img)
        duplicate, dup_reg_no, dist = check_duplicate(emb, embeddings_db)
        if duplicate:
            msg = f" {dup_reg_no} (sim={1-dist:.2f})"
            color = (0, 0, 255)
        else:
            msg = "New face"
            color = (0, 255, 0)

        # Draw
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, msg, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow('Register', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            if duplicate:
                print("Already registered, skipping.")
                continue
            # Save face image
            img_num = get_next_image_num(folder)
            img_path = os.path.join(folder, f"img_{img_num}.jpg")
            cv2.imwrite(img_path, face_img)
            print(f"Saved: {img_path}")
            # Save embedding for next time
            embeddings_db.setdefault(reg_no, []).append(emb)
            save_embeddings(embeddings_db)
            captured += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Register face with duplicate check")
    parser.add_argument('--reg_no', required=True, help='Student registration number')
    args = parser.parse_args()
    register_face(args.reg_no)
