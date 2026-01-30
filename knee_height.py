import cv2
import numpy as np
from skimage.segmentation import flood_fill
from tqdm import tqdm

from mmengine.registry import OPTIMIZERS

OPTIMIZERS.module_dict.clear()

from mmpose.apis import MMPoseInferencer
import torch

import matplotlib.pyplot as plt

# Generate a binary mask of the patient's walking path


def extract_mask(start_frame, last_frame):
    [H, W] = start_frame.shape[:2]
    start_frame = np.flip(start_frame[H // 2 : H, :], 2)
    last_frame = np.flip(last_frame[H // 2 : H, :], 2)

    image = start_frame[:, :, 2] - (
        start_frame[:, :, 0] // 2 + start_frame[:, :, 1] // 2
    )
    image = np.array(
        255 * ((image - image.min()) / (image.max() - image.min())), dtype=np.uint8
    )

    image[(image < 15) | (image > 60)] = 0
    image[image > 0] = 255

    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    y, x = np.where(opening > 0)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xc = (xmin + xmax) // 2
    yc = (ymin + ymax) // 2

    scanny = cv2.Canny(start_frame, 40, 100)
    lcanny = cv2.Canny(last_frame, 40, 100)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    final1 = np.zeros(scanny.shape)
    final1[:, :xc] = lcanny[:, :xc]
    final1[:, xc:] = scanny[:, xc:]
    final1 = final1.astype(np.uint8)
    final1 = cv2.dilate(final1, kernel)

    final2 = np.zeros(scanny.shape)
    final2[:, :xc] = scanny[:, :xc]
    final2[:, xc:] = lcanny[:, xc:]
    final2 = final2.astype(np.uint8)
    final2 = cv2.dilate(final2, kernel)

    final = final2 if np.sum(final1) > np.sum(final2) else final1

    filled_checkers = flood_fill(final, (yc, xmax + 30), 127)
    filled_checkers = flood_fill(filled_checkers, (yc, xmin - 30), 127)
    filled_checkers = flood_fill(filled_checkers, (yc, xmin - 100), 127)
    filled_checkers = flood_fill(filled_checkers, (ymin - 10, xc), 127)
    filled_checkers = flood_fill(filled_checkers, (ymax + 20, xc), 127)
    filled_checkers = flood_fill(filled_checkers, (yc + 30, xc + 30), 127)
    filled_checkers[y, x] = 127

    path = 255 * np.array(filled_checkers == 127, dtype=np.uint8)
    kernel = np.ones((11, 11))
    final_path = cv2.dilate(path, kernel)
    final_path = cv2.dilate(final_path, kernel)
    final_path = cv2.dilate(final_path, kernel)
    final_path[yc, xc] = 0

    mask = np.zeros((H, W))
    mask[-final_path.shape[0] - 1 : -1, :] = final_path
    return mask.astype(np.uint8)


def select_patient_by_mask(keypoints_list, mask):
    """
    Select the person whose keypoints overlap the most with the given mask.
    """
    max_overlap = -1
    selected_kp = None

    for kp in keypoints_list:
        kp = np.array(kp)
        x = np.clip(kp[:, 0].astype(int), 0, mask.shape[1] - 1)
        y = np.clip(kp[:, 1].astype(int), 0, mask.shape[0] - 1)

        inside_mask = mask[y, x]
        overlap_count = np.count_nonzero(inside_mask)

        if overlap_count > max_overlap:
            max_overlap = overlap_count
            selected_kp = kp

    return selected_kp

# Note: The original video used in this analysis is confidential and cannot be shared.
# Please replace 'video_path' with the path to your own video file for testing.
video_path = (
    r"C:\Users\mrham\Desktop\internship_project\ph2\VID04.2103582.20190907174345.avi"
)
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError("Video could not be opened! Check the path or file.")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)  # for converting frames to seconds

ret, first_frame = cap.read()
if not ret:
    raise ValueError("First frame of the video could not be read!")
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
ret, last_frame = cap.read()
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

mask = extract_mask(first_frame, last_frame).astype(bool)

inferencer = MMPoseInferencer("rtmpose-m_8xb64-270e_coco-wholebody-256x192")

selected_keypoints_per_frame = []
right_knee_height_per_frame = []  # <<<------------ calculate left knee height

for _ in tqdm(range(frame_count)):
    ret, frame = cap.read()
    if not ret:
        break

    res_generator = inferencer(frame)
    result = next(res_generator)
    keypoints_list = [inst["keypoints"] for inst in result["predictions"][0]]

    selected_keypoints = select_patient_by_mask(keypoints_list, mask)
    selected_keypoints_per_frame.append(selected_keypoints)

    # <<<------------ calculate left knee height
    if selected_keypoints is not None:
        H = frame.shape[0]  # image height
        right_knee_idx = 15  # index of left knee in COCO-wholebody model
        # Distance from bottom of the frame = H - y
        right_knee_height = H - selected_keypoints[right_knee_idx, 1]
    else:
        right_knee_height = np.nan  # if no keypoints detected in this frame
    right_knee_height_per_frame.append(right_knee_height)
    # ---------------------------------------------------------

cap.release()


# Plot left–knee height over time (frames and seconds)
frames = np.arange(len(right_knee_height_per_frame))
times = frames / fps

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(frames, right_knee_height_per_frame, "b.-")
plt.xlabel("Frame")
plt.ylabel("Left Knee Height (pixels)")
plt.title("Left Knee Height vs Frame")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(times, right_knee_height_per_frame, "r.-")
plt.xlabel("Time (seconds)")
plt.ylabel("Left Knee Height (pixels)")
plt.title("Left Knee Height vs Time")
plt.grid(True)

plt.tight_layout()
plt.show()
