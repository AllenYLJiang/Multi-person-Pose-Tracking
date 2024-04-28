import cv2
from SimpleHRNet import SimpleHRNet
import torch

# "keypoints": {
#     0: "nose",
#     1: "left_eye",
#     2: "right_eye",
#     3: "left_ear",
#     4: "right_ear",
#     5: "left_shoulder",
#     6: "right_shoulder",
#     7: "left_elbow",
#     8: "right_elbow",
#     9: "left_wrist",
#     10: "right_wrist",
#     11: "left_hip",
#     12: "right_hip",
#     13: "left_knee",
#     14: "right_knee",
#     15: "left_ankle",
#     16: "right_ankle"
# },
# "skeleton": [
#     # # [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
#     # # [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
#     # [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
#     # [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
#     [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
#     [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],  # [3, 5], [4, 6]
#     [0, 5], [0, 6]
# ]
mask_out_outside_region = 0
model = SimpleHRNet(48, 17, "./weights/pose_hrnet_w48_384x288.pth", device=torch.device('cuda'))

center_coord = [403, 463] # [442, 400] # [402, 468] # [432, 412] # [405, 462] # [904, 300] # [902, 304] # [444, 465] # [902, 304]
size_single_person = [127, 156] # [189, 120] # [132, 182] # [170, 116] # [126, 157] # [153, 98] # [160, 118] # [182, 180] # [160, 118]

scale_list = [1.0] # [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.75, 1.8, 1.9, 2.0]

for scale_value in scale_list:
    width, height = size_single_person[0] * scale_value, size_single_person[1] * scale_value
    left, top, right, bottom = int(center_coord[0] - width / 2), int(center_coord[1] - height / 2), int(center_coord[0] + width / 2), int(center_coord[1] + height / 2)
    # curr_crop = image[top:bottom, left:right, :]
    image = cv2.imread("D:\\Materials\\MyPaper\\Second_submit\\Fig10b\\simple_demo\\000008.png", cv2.IMREAD_COLOR)
    # mask out
    if mask_out_outside_region:
        image[0:top, :, :] = 0
        image[bottom:, :, :] = 0
        image[:, 0:left, :] = 0
        image[:, right:, :] = 0

    heatmaps, joints = model.predict(image, left, right, top, bottom)
    for joint_idx in range(17):
        radius, thickness, color = 2, 2, [0, 255, 0]
        image = cv2.circle(image, (int(joints[0, joint_idx, 1]), int(joints[0, joint_idx, 0])), radius, color, thickness)
    cv2.imwrite('D:\\Materials\\MyPaper\\Second_submit\\Fig10b\\simple_demo\\' + '%06d' % (scale_list.index(scale_value)) + '.png', image)
