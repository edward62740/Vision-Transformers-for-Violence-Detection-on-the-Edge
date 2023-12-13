import os
from PIL import Image

import cv2

PROJ_PATH = r"C:\Users\Workstation\Documents\GitHub\URECA-Project"

train_video_folder = PROJ_PATH + '/ucf_dataset'
train_frame_folder = PROJ_PATH + '/ucf_dataset_proc5'

fps_target = 5  # Target frames per second

for category_folder in os.listdir(train_video_folder):
    category_path = os.path.join(train_video_folder, category_folder)
    if os.path.isdir(category_path):  # Check if it's a directory
        for video_file in os.listdir(category_path):
            if video_file.endswith('.mp4'):
                video_path = os.path.join(category_path, video_file)
                frame_subdir = os.path.splitext(video_file)[0]
                frame_dir = os.path.join(train_frame_folder, category_folder, frame_subdir)
                os.makedirs(frame_dir, exist_ok=True)

                video_capture = cv2.VideoCapture(video_path)
                frame_count = 0
                fps = video_capture.get(cv2.CAP_PROP_FPS)  # Get the original video's frames per second
                frame_interval = int(round(fps / fps_target))  # Calculate the frame interval to achieve the target fps

                while True:
                    ret, frame = video_capture.read()
                    if not ret:
                        break
                    if frame_count % frame_interval == 0:  # Select frames at the desired interval
                        frame = Image.fromarray(frame)  # Convert frame to PIL Image
                        #frame = frame.resize(image_size)  # Resize the image
                        frame_path = os.path.join(frame_dir, f'{frame_count}.jpg')
                        frame.save(frame_path)  # Save the resized image
                    frame_count += 1

                video_capture.release()