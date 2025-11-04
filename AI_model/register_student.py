import cv2
import os
import argparse


# Create a folder for each student's images based on registration number
def create_student_folder(reg_no):
    folder_path = f"./dataset/{reg_no}"
    os.makedirs(folder_path, exist_ok=True)  # Create folder if not exists
    return folder_path


def get_next_image_number(folder):
    existing_files = [f for f in os.listdir(folder) if f.endswith('.jpg')]
    numbers = [int(f.split('_')[1].split('.')[0]) for f in existing_files if f.startswith('image_')]
    return max(numbers) + 1 if numbers else 1


# Capture multiple face images interactively with UI
def capture_images(reg_no, image_count=50):
    folder = create_student_folder(reg_no)
    cap = cv2.VideoCapture(0)  # Start webcam
    count = 0
    capturing = False  # Flag to start continuous capture after space pressed

    while count < image_count:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from camera.")
            continue

        # UI overlay with rectangle to guide face placement
        ui_frame = frame.copy()
        height, width, _ = frame.shape
        rect_x1, rect_y1 = width // 4, height // 4
        rect_x2, rect_y2 = 3 * width // 4, 3 * height // 4  # Fixed height calculation to inside frame
        cv2.rectangle(ui_frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 255, 0), 2)
        cv2.putText(ui_frame, f"Place face in the box - Image {count + 1}/{image_count}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Student Registration", ui_frame)
        key = cv2.waitKey(1) & 0xFF

        if not capturing and key == ord(' '):  # Start capturing on first space press
            capturing = True
            print("Started capturing images automatically...")

        if capturing:
            next_num = get_next_image_number(folder)
            img_name = os.path.join(folder, f"image_{next_num}.jpg")
            cv2.imwrite(img_name, frame)
            print(f"Image saved: {img_name}")
            count += 1

        if key == ord('q') or key == 27:
            print("Registration aborted by admin.")
            break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register new student face images")
    parser.add_argument('--reg_no', required=True, help="Student registration number")
    args = parser.parse_args()

    capture_images(args.reg_no)
