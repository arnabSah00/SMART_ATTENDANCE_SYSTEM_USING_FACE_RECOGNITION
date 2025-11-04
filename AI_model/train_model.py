import os
import cv2
import numpy as np
from PIL import Image
import pickle  # To save the label map for later reference

# Train LBPH Face Recognizer model using student images for attendance system
def train_model(haar_cascade_path, dataset_path, model_save_path, labels_save_path,
                message_callback=None, tts_callback=None):
    """
    Parameters:
    - haar_cascade_path: Path to Haar Cascade XML face detector.
    - dataset_path: Root dataset folder; each subfolder is a student ID with images.
    - model_save_path: Path where trained LBPH model will be saved.
    - labels_save_path: Path to save label map (student ID <-> int ID).
    - message_callback: Optional function to update UI messages.
    - tts_callback: Optional function for text-to-speech announcements.
    """
    
    # Create LBPH recognizer from OpenCV
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Load Haar cascade for face detection
    detector = cv2.CascadeClassifier(haar_cascade_path)
    
    # Get faces and numeric labels, and generate mapping from student IDs (strings) to ints
    faces, ids, label_map = get_faces_and_labels(dataset_path, detector)
    
    # Train the recognizer model on the gathered faces and corresponding numeric IDs
    recognizer.train(faces, np.array(ids))
    
    # Save the trained LBPH model for later face recognition
    recognizer.save(model_save_path)
    
    # Save the mapping from numeric labels back to student registration numbers (strings)
    with open(labels_save_path, 'wb') as f:
        pickle.dump(label_map, f)
    
    result_message = f"Training completed successfully. Model saved to {model_save_path}"
    
    if message_callback:
        message_callback(result_message)
    if tts_callback:
        tts_callback(result_message)

    print(result_message)


#level with neumaric value 
def get_faces_and_labels(dataset_path, detector):
    """
    Traverse dataset directory, detect faces in images,
    and return face samples with numeric IDs and label map.
    """
    faces = []
    ids = []
    label_map = {}                   # Map: student reg_no (string) => int label
    reverse_label_map = {}           # Reverse map: int label => student reg_no
    current_label = 0
    
    # List student folders inside dataset path
    student_folders = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    for folder_path in student_folders:
        reg_no = os.path.basename(folder_path)  # Student registration number folder name
        
        # Create new label if this reg_no not seen before
        if reg_no not in label_map:
            label_map[reg_no] = current_label
            reverse_label_map[current_label] = reg_no
            current_label += 1
        
        numeric_label = label_map[reg_no]
        
        # List images for this student
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        
        for image_path in image_files:
            # Open image and convert to grayscale
            pil_image = Image.open(image_path).convert('L')
            image_np = np.array(pil_image, 'uint8')

            # Detect faces in image
            detected_faces = detector.detectMultiScale(image_np)
            
            # Extract face portion for each detected face and assign label
            for (x, y, w, h) in detected_faces:
                face_region = image_np[y:y+h, x:x+w]  # Crop face ROI
                faces.append(face_region)
                ids.append(numeric_label)
    
    # Return all extracted faces and label IDs alongside the label mapping dict
    return faces, ids, {"label_map": label_map, "reverse_label_map": reverse_label_map}


if __name__ == "__main__":
    # Define paths - update as per your environment
    HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'  # Provided in AI_model folder
    DATASET_PATH = './dataset'                                  # Folder with student face images
    MODEL_SAVE_PATH = './model/trained_lbph_model.yml'          # Where to save trained model
    LABELS_SAVE_PATH = './model/label_map.pkl'                   # Where to save label mappings

    # Train model on dataset, no UI callbacks here - just console output
    train_model(HAAR_CASCADE_PATH, DATASET_PATH, MODEL_SAVE_PATH, LABELS_SAVE_PATH)
