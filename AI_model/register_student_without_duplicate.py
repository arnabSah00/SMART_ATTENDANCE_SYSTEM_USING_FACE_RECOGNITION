import os
import cv2
import pickle

# Paths for your trained model and labels
MODEL_SAVE_PATH = './model/trained_lbph_model.yml'
LABELS_SAVE_PATH = './model/label_map.pkl'
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'

# Create student folder if not exists
def create_student_folder(reg_no):
    folder_path = f"./dataset/{reg_no}"
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

# Get next sequential image number in folder
def get_next_image_number(folder):
    existing_files = [f for f in os.listdir(folder) if f.endswith('.jpg')]
    numbers = []
    for f in existing_files:
        parts = f.split('_')
        if len(parts) < 2:
            continue
        num_str = parts[1].split('.')[0]
        if num_str.isdigit():
            numbers.append(int(num_str))
    return max(numbers) + 1 if numbers else 1

# Load LBPH recognizer and labels if available
def load_recognizer_and_labels():
    if not os.path.exists(MODEL_SAVE_PATH) or not os.path.exists(LABELS_SAVE_PATH):
        print("No trained model or labels found; skipping duplicate detection.")
        return None, None

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_SAVE_PATH)

    with open(LABELS_SAVE_PATH, 'rb') as f:
        label_data = pickle.load(f)

    return recognizer, label_data['reverse_label_map']

# Check if face already registered using existing LBPH model
def is_face_registered(frame, detector, recognizer, reverse_label_map, threshold=70):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray)
    if len(faces) == 0:
        return False, None, None

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face_roi)
        if confidence < threshold:
            reg_no = reverse_label_map.get(label)
            return True, reg_no, confidence
    return False, None, None

# Main function to capture student images with duplicate check
def capture_images(reg_no, image_count=50):
    folder = create_student_folder(reg_no)
    cap = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

    recognizer, reverse_label_map = load_recognizer_and_labels()

    count = 0
    print(f"Starting registration for student: {reg_no}")
    while count < image_count:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera.")
            continue

        # Draw UI rectangle
        height, width, _ = frame.shape
        cv2.rectangle(frame, (width//4, height//4), (3*width//4, 3*height//4), (0, 255, 0), 2)
        cv2.putText(frame, f"Place face in box - Image {count+1}/{image_count}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Student Registration", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            # If model exists, check for duplicate
            if recognizer:
                is_dup, existing_reg_no, conf = is_face_registered(frame, detector, recognizer, reverse_label_map)
                if is_dup:
                    print(f"Duplicate face detected! Already registered as {existing_reg_no} (confidence: {conf:.2f}). Image not saved.")
                    continue  # skip saving this image

            # Save image with sequential name
            next_num = get_next_image_number(folder)
            img_name = os.path.join(folder, f"image_{next_num}.jpg")
            cv2.imwrite(img_name, frame)
            print(f"Saved image: {img_name}")

            count += 1

        elif key == ord('q'):
            print("Registration aborted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Register student face images.")
    parser.add_argument('--reg_no', required=True, help='Student registration number')
    args = parser.parse_args()

    capture_images(args.reg_no)
