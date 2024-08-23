import cv2
import numpy as np
from PIL import Image
import os
import glob


def process_image_folder(folder_path, save_dir):
    # Determine the save directory from the folder name
    folder_name = '_'.join(os.path.basename(folder_path).split('_')[1:])
    final_save_dir = os.path.join(save_dir, folder_name)

    # Check if the directory exists and contains files
    if os.path.exists(final_save_dir) and os.listdir(final_save_dir):
        print(f"Skipping image processing for {final_save_dir}, files already exist.")
        return

    # If no files, continue processing
    if not os.path.exists(final_save_dir):
        os.makedirs(final_save_dir)

    # Process each image in the folder
    for image_path in glob.glob(os.path.join(folder_path, "*.jpg")):
        img = Image.open(image_path)
        width, height = img.size
        left = width * 0.1
        top = height * 0.1
        right = width * 0.9
        bottom = height * 0.9
        cropped_img = img.crop((left, top, right, bottom))

        # Resize the image to 512x512->224*224
        resized_img = cropped_img.resize((224, 224))
        cropped_image_array = np.array(resized_img)

        file_index = len(glob.glob(os.path.join(final_save_dir, "*.npy"))) + 1
        save_path = os.path.join(final_save_dir, f"{file_index:02d}.npy")
        np.save(save_path, cropped_image_array)
        print(f"Saved image {file_index:02d}.npy in {final_save_dir}")


def process_video(video_path, save_dir):
    # Determine the save directory from the file name
    folder_name = '_'.join(os.path.basename(video_path).split('_')[1:]).split('.')[0]
    final_save_dir = os.path.join(save_dir, folder_name)

    # Check if the directory exists and contains files
    if os.path.exists(final_save_dir) and os.listdir(final_save_dir):
        print(f"Skipping video processing for {final_save_dir}, files already exist.")
        retcurn

    # If no files, continue processing
    if not os.path.exists(final_save_dir):
        os.makedirs(final_save_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        if count % 10 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(cnt)
                frame = frame[y:y + h, x:x + w]

            # Resize the frame to 512x512
            resized_frame = cv2.resize(frame, (224, 224))
            frame_array = np.array(resized_frame)

            file_index = count // 10 + 1
            save_path = os.path.join(final_save_dir, f"{file_index:02d}.npy")
            np.save(save_path, frame_array)
            print(f"Saved video frame {file_index:02d}.npy in {final_save_dir}")

        count += 1

    cap.release()
    print(f"Completed processing video: {video_path}")


# Paths
base_dir = "Please use your own path"
save_dir = "Please use your own path"

# Process all image folders and videos in the base directory
for folder in glob.glob(os.path.join(base_dir, "Photos_*")):
    print(f"Processing image folder: {folder}")
    process_image_folder(folder, save_dir)

for video_file in glob.glob(os.path.join(base_dir, "*.mp4")):
    print(f"Processing video: {video_file}")
    process_video(video_file, save_dir)
