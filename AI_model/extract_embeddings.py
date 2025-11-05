import os
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def register_students(dataset_path='dataset'):
    embeddings = []
    labels = []

    for student_id in os.listdir(dataset_path):
        student_folder = os.path.join(dataset_path, student_id)
        if not os.path.isdir(student_folder):
            continue

        for img_name in os.listdir(student_folder):
            img_path = os.path.join(student_folder, img_name)
            img = Image.open(img_path).convert('RGB')

            face = mtcnn(img)
            if face is not None:
                face = face.unsqueeze(0).to(device)
                embedding = facenet(face)
                embeddings.append(embedding.detach().cpu().numpy()[0])
                labels.append(student_id)  # Keep alphanumeric student ID

    embeddings = np.array(embeddings)
    labels = np.array(labels)
    return embeddings, labels

if __name__ == '__main__':
    embeddings, labels = register_students()
    np.save('.\model\embeddings.npy', embeddings)
    np.save('.\model\labels.npy', labels)
    print("Registration complete. Embeddings and labels saved.")
