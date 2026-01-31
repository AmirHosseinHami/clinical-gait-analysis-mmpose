# This script reads an input video frame by frame, runs whole-body pose estimation on each frame
# using a pretrained RTMPose model from MMPose, draws the detected skeleton/keypoints on the frame,
# and saves the visualized frames as sequentially numbered images inside the "output_frames" folder.

import cv2
import os
from mmpose.apis import MMPoseInferencer
import shutil

# Initialize the MMPose inferencer with a pretrained model
inferencer = MMPoseInferencer("rtmpose-m_8xb64-270e_coco-wholebody-256x192")

# Path to the input video
video_path = "temp_video.mp4"

# Directory to save visualized pose frames (created locally when running the script)
final_output_dir = "output_frames"
os.makedirs(final_output_dir, exist_ok=True)

# Video capture
cap = cv2.VideoCapture(video_path)
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Make a temporary folder for this frame
    temp_vis_dir = f"temp_vis_{frame_idx}"
    os.makedirs(temp_vis_dir, exist_ok=True)

    # Run inference and save visualized output in the temp folder
    _ = next(inferencer(frame, vis_out_dir=temp_vis_dir))

    # Look for any image file inside the temp folder
    saved_images = [f for f in os.listdir(temp_vis_dir)]

    if saved_images:
        temp_img_path = os.path.join(temp_vis_dir, saved_images[0])
        final_img_path = os.path.join(final_output_dir, f"frame_{frame_idx:04d}.jpg")
        shutil.move(temp_img_path, final_img_path)
    else:
        print(f"Warning: No image saved for frame {frame_idx}")

    # Delete the temporary folder
    shutil.rmtree(temp_vis_dir)

    frame_idx += 1


cap.release()
print("All frames processed and saved.")
