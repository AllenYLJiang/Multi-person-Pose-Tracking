import cv2
import os
import json
import random
import numpy as np
import ot
import math
from math import exp
# from SimpleHRNet import SimpleHRNet
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import skimage
from skimage.morphology import convex_hull_image
from skimage import data, img_as_float
from skimage.util import invert
import copy
import torch
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import colorsys
from skimage.draw import polygon
from skimage import measure
from skimage.morphology import convex_hull_image
from base_pose_model_demo import self_inference_pose_model

posetrack_submission_semantics = {
    'head top': 15-1,
    'nose': 14-1,
    'neck': 13-1,
    'left shoulder': 10-1,
    'right shoulder': 9-1,
    'left elbow': 11-1,
    'right elbow': 8-1,
    'left wrist': 12-1,
    'right wrist': 7-1,
    'left hip': 4-1,
    'right hip': 3-1,
    'left knee': 5-1,
    'right knee': 2-1,
    'left ankle': 6-1,
    'right ankle': 1-1
}

# hrnet format
hrnet_semantics = {
    0: "nose",
    1: "left eye",
    2: "right eye",
    3: "left ear",
    4: "right ear",
    5: "left shoulder",
    6: "right shoulder",
    7: "left elbow",
    8: "right elbow",
    9: "left wrist",
    10: "right wrist",
    11: "left hip",
    12: "right hip",
    13: "left knee",
    14: "right knee",
    15: "left ankle",
    16: "right ankle"
}

#
base_pose_model_semantics = {
    0: "nose",
    1: "neck",
    2: "head top",
    3: "invalid 1",
    4: "invalid 2",
    5: "left shoulder",
    6: "right shoulder",
    7: "left elbow",
    8: "right elbow",
    9: "left wrist",
    10: "right wrist",
    11: "left hip",
    12: "right hip",
    13: "left knee",
    14: "right knee",
    15: "left ankle",
    16: "right ankle"
}

posetrack_submission_semantic_to_index = {
    'head top': 14,
    'nose': 13,
    'neck': 12,
    'left shoulder': 9,
    'right shoulder': 8,
    'left elbow': 10,
    'right elbow': 7,
    'left wrist': 11,
    'right wrist': 6,
    'left hip': 3,
    'right hip': 2,
    'left knee': 4,
    'right knee': 1,
    'left ankle': 5,
    'right ankle': 0
}

manually_labelled_semantics_16_points = {
    0: "nose",
    # 1: "left eye",
    # 2: "right eye",
    # 3: "left ear",
    # 4: "right ear",
    5: "left shoulder",
    6: "right shoulder",
    7: "left elbow",
    8: "right elbow",
    9: "left wrist",
    10: "right wrist",
    11: "left hip",
    12: "right hip",
    13: "left knee",
    14: "right knee",
    15: "left ankle",
    16: "right ankle"
}

def preprocess_manual_json(manual_json):
    all_in_one_list = manual_json['shapes']
    multi_human_keypoint_list = [x for x in all_in_one_list if int(x['label']) < 18]
    multi_human_replace_or_correct_list = [int(x['label']) for x in all_in_one_list if int(x['label']) > 18]
    multi_human_keypoint_list_segment_point = [x for x in range(len(multi_human_keypoint_list) - 1) if int(multi_human_keypoint_list[x]['label']) >= int(multi_human_keypoint_list[x + 1]['label'])]
    assert(len(multi_human_keypoint_list_segment_point) == len(multi_human_replace_or_correct_list) - 1)
    keypoint_list, replace_or_correct_list = [], []
    if len(multi_human_keypoint_list_segment_point) == 0:
        keypoint_list.append(multi_human_keypoint_list)
    else:
        for segment_point in multi_human_keypoint_list_segment_point:
            if multi_human_keypoint_list_segment_point.index(segment_point) == 0:
                keypoint_list.append(multi_human_keypoint_list[:segment_point+1])
            else:
                keypoint_list.append(multi_human_keypoint_list[multi_human_keypoint_list_segment_point[multi_human_keypoint_list_segment_point.index(segment_point) - 1]+1:segment_point+1])
        keypoint_list.append(multi_human_keypoint_list[multi_human_keypoint_list_segment_point[-1] + 1:])
    replace_or_correct_list = multi_human_replace_or_correct_list
    return keypoint_list, replace_or_correct_list

def determine_skeleton_bboxes(manual_keypoint_list, manual_replace_or_correct_list):
    result_bbox_list = []
    for manual_keypoint_list_ele in manual_keypoint_list:
        # 20 denotes corrected skeleton, 21 denotes added skeleton
        if manual_replace_or_correct_list[manual_keypoint_list.index(manual_keypoint_list_ele)] == 21:
            continue
        left, right = min([x['points'][0][0] for x in manual_keypoint_list_ele]), max([x['points'][0][0] for x in manual_keypoint_list_ele])
        top, bottom = min([x['points'][0][1] for x in manual_keypoint_list_ele]), max([x['points'][0][1] for x in manual_keypoint_list_ele])
        result_bbox_list.append([top, bottom, left, right])
    return result_bbox_list

keypoints_colors = {
    0: [int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)],
    1: [int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)],
    2: [int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)],
    3: [int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)],
    4: [int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)],
    5: [int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)],
    6: [int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)],
    7: [int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)],
    8: [int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)],
    9: [int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)],
    10: [int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)],
    11: [int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)],
    12: [int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)],
    13: [int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)],
    14: [int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)],
    15: [int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)],
    16: [int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)]
}

# sample 6,12,18 points from circles with radii being 1,2,3,...
def sampling_around_joint_return_samples(center_coord, num_samples, outer_radius, min_unit_circle = 2.0):
    # the number of samples in each circle is 6,12,18,24,...
    arithmetic_progression, arithmetic_progression_basis, arithmetic_progression_delta = 0, 6, 6
    num_circles, num_samples_each_circle = 0, []
    while arithmetic_progression < num_samples:
        if arithmetic_progression == 0:
            num_circles += 1
            arithmetic_progression += arithmetic_progression_basis
            num_samples_each_circle.append(min([arithmetic_progression_basis, num_samples]))
        else:
            num_circles += 1
            tmp_num = arithmetic_progression
            arithmetic_progression += (arithmetic_progression_basis + (num_circles - 1) * arithmetic_progression_delta)
            num_samples_each_circle.append(min([arithmetic_progression - tmp_num, num_samples - tmp_num]))
    unit_circle = max([outer_radius / num_circles, min_unit_circle])
    samples = []
    # traverse circles
    for circle_idx in range(num_circles):
        curr_circle_radius = (circle_idx + 1) * unit_circle
        curr_circle_num_samples = num_samples_each_circle[circle_idx]
        # traverse points in current circle
        for sample_idx in range(curr_circle_num_samples):
            hori_curr_sample, vert_curr_sample = curr_circle_radius*math.cos(sample_idx * (math.pi * 2 / curr_circle_num_samples)), curr_circle_radius*math.sin(sample_idx * (math.pi * 2 / curr_circle_num_samples))
            samples.append([center_coord[0] + hori_curr_sample, center_coord[1] + vert_curr_sample])
    samples.append(center_coord)
    return samples

def compute_iou_single_box(curr_img_boxes, next_img_boxes):# Order: top, bottom, left, right
    intersect_vert = min([curr_img_boxes[1], next_img_boxes[1]]) - max([curr_img_boxes[0], next_img_boxes[0]])
    intersect_hori = min([curr_img_boxes[3], next_img_boxes[3]]) - max([curr_img_boxes[2], next_img_boxes[2]])
    union_vert = next_img_boxes[1] - next_img_boxes[0] # max([curr_img_boxes[1], next_img_boxes[1]]) - min([curr_img_boxes[0], next_img_boxes[0]])
    union_hori = next_img_boxes[3] - next_img_boxes[2] # max([curr_img_boxes[3], next_img_boxes[3]]) - min([curr_img_boxes[2], next_img_boxes[2]])
    if intersect_vert > 0 and intersect_hori > 0 and union_vert > 0 and union_hori > 0:
        corresponding_coefficient = float(intersect_vert) * float(intersect_hori) / (union_hori * union_vert) # (float(curr_img_boxes[1] - curr_img_boxes[0]) * float(curr_img_boxes[3] - curr_img_boxes[2]) + float(next_img_boxes[1] - next_img_boxes[0]) * float(next_img_boxes[3] - next_img_boxes[2]) - float(intersect_vert) * float(intersect_hori))
    else:
        corresponding_coefficient = 0.0
    return corresponding_coefficient

def compute_iou_single_box_normal(curr_img_boxes, next_img_boxes):# Order: top, bottom, left, right
    intersect_vert = min([curr_img_boxes[1], next_img_boxes[1]]) - max([curr_img_boxes[0], next_img_boxes[0]])
    intersect_hori = min([curr_img_boxes[3], next_img_boxes[3]]) - max([curr_img_boxes[2], next_img_boxes[2]])
    union_vert = max([curr_img_boxes[1], next_img_boxes[1]]) - min([curr_img_boxes[0], next_img_boxes[0]])
    union_hori = max([curr_img_boxes[3], next_img_boxes[3]]) - min([curr_img_boxes[2], next_img_boxes[2]])
    if intersect_vert > 0 and intersect_hori > 0 and union_vert > 0 and union_hori > 0:
        corresponding_coefficient = float(intersect_vert) * float(intersect_hori) / (float(curr_img_boxes[1] - curr_img_boxes[0]) * float(curr_img_boxes[3] - curr_img_boxes[2]) + float(next_img_boxes[1] - next_img_boxes[0]) * float(next_img_boxes[3] - next_img_boxes[2]) - float(intersect_vert) * float(intersect_hori))
    else:
        corresponding_coefficient = 0.0
    return corresponding_coefficient

# def skeleton_nms_format2(next_img_propagated_and_detected_bboxes_skeleton, next_img_propagated_and_detected_bboxes):
#     all_people_skeleton_boundaries_list = []
#     for single_person_skeleton in next_img_propagated_and_detected_bboxes_skeleton:
#         # top, bottom, left, right
#         all_people_skeleton_boundaries_list.append([min(single_person_skeleton[:, 0]), max(single_person_skeleton[:, 0]), \
#                                                     min(single_person_skeleton[:, 1]), max(single_person_skeleton[:, 1])])
#     iou_matrix = np.zeros((len(all_people_skeleton_boundaries_list), len(all_people_skeleton_boundaries_list)))
#     curr_img_all_skeletons_valid_masks = np.ones((len(all_people_skeleton_boundaries_list)))
#     # not symmetric because iou is head-body
#     for iou_matrix_row in range(iou_matrix.shape[0]):
#         for iou_matrix_col in range(iou_matrix.shape[1]):
#             if iou_matrix_row != iou_matrix_col:
#                 iou_matrix[iou_matrix_row, iou_matrix_col] = compute_iou_single_box(all_people_skeleton_boundaries_list[iou_matrix_row], \
#                     all_people_skeleton_boundaries_list[iou_matrix_col])
#                 if iou_matrix[iou_matrix_row, iou_matrix_col] >= iou_thresh:
#                     if np.mean(next_img_propagated_and_detected_bboxes_skeleton[iou_matrix_row][:, 2]) > np.mean(next_img_propagated_and_detected_bboxes_skeleton[iou_matrix_col][:, 2]):
#                         curr_img_all_skeletons_valid_masks[iou_matrix_col] = 0
#                     else:
#                         curr_img_all_skeletons_valid_masks[iou_matrix_row] = 0
#
#     curr_video_bboxes_curr_frame_corrected = []
#     for curr_video_bboxes_curr_frame_idx in range(len(next_img_propagated_and_detected_bboxes)):
#         if curr_img_all_skeletons_valid_masks[curr_video_bboxes_curr_frame_idx] == 1:
#             curr_video_bboxes_curr_frame_corrected.append(next_img_propagated_and_detected_bboxes[curr_video_bboxes_curr_frame_idx])
#
#     curr_img_all_skeletons_annorect = []
#     for curr_img_all_skeletons_valid_masks_idx in range(len(curr_img_all_skeletons_valid_masks)):
#         if curr_img_all_skeletons_valid_masks[curr_img_all_skeletons_valid_masks_idx] == 1:
#             curr_img_all_skeletons_annorect.append(next_img_propagated_and_detected_bboxes_skeleton[curr_img_all_skeletons_valid_masks_idx])
#     next_img_propagated_and_detected_bboxes_skeleton = curr_img_all_skeletons_annorect
#
#     return next_img_propagated_and_detected_bboxes_skeleton, curr_video_bboxes_curr_frame_corrected

def skeleton_nms(curr_img_all_skeletons, curr_video_bboxes, iou_thresh, curr_img_name):
    all_people_skeleton_boundaries_list = []
    # for single_person_skeleton in curr_img_all_skeletons['annorect']:
    #     # top, bottom, left, right
    #     all_people_skeleton_boundaries_list.append([min([x['y'][0] for x in single_person_skeleton['annopoints'][0]['point']]), \
    #                                                 max([x['y'][0] for x in single_person_skeleton['annopoints'][0]['point']]), \
    #                                                 min([x['x'][0] for x in single_person_skeleton['annopoints'][0]['point']]), \
    #                                                 max([x['x'][0] for x in single_person_skeleton['annopoints'][0]['point']])])

    for single_person_bbox in [x for x in curr_video_bboxes if (curr_img_name in x['image_name'])]:
        left, top, width, height = single_person_bbox['bbox'][0], single_person_bbox['bbox'][1], single_person_bbox['bbox'][2], single_person_bbox['bbox'][3]
        all_people_skeleton_boundaries_list.append([top, top + height, left, left + width])

    iou_matrix = np.zeros((len(all_people_skeleton_boundaries_list), len(all_people_skeleton_boundaries_list)))
    curr_img_all_skeletons_valid_masks = np.ones((len(all_people_skeleton_boundaries_list)))
    # not symmetric because iou is head-body
    for iou_matrix_row in range(iou_matrix.shape[0]):
        for iou_matrix_col in range(iou_matrix.shape[1]):
            if iou_matrix_row != iou_matrix_col:
                iou_matrix[iou_matrix_row, iou_matrix_col] = compute_iou_single_box(all_people_skeleton_boundaries_list[iou_matrix_row], \
                                                                                    all_people_skeleton_boundaries_list[iou_matrix_col])
                if iou_matrix[iou_matrix_row, iou_matrix_col] >= iou_thresh:
                    if curr_img_all_skeletons['annorect'][iou_matrix_row]['score'][0] > curr_img_all_skeletons['annorect'][iou_matrix_col]['score'][0]:
                        curr_img_all_skeletons_valid_masks[iou_matrix_col] = 0
                    else:
                        curr_img_all_skeletons_valid_masks[iou_matrix_row] = 0

    curr_video_bboxes_curr_frame = [x for x in curr_video_bboxes if (curr_img_name in x['image_name'])]
    curr_video_bboxes_other_frames = [x for x in curr_video_bboxes if (curr_img_name not in x['image_name'])]
    curr_video_bboxes_curr_frame_corrected = []
    for curr_video_bboxes_curr_frame_idx in range(len(curr_video_bboxes_curr_frame)):
        if curr_img_all_skeletons_valid_masks[curr_video_bboxes_curr_frame_idx] == 1:
            curr_video_bboxes_curr_frame_corrected.append(curr_video_bboxes_curr_frame[curr_video_bboxes_curr_frame_idx])

    curr_img_all_skeletons_annorect = []
    for curr_img_all_skeletons_valid_masks_idx in range(len(curr_img_all_skeletons_valid_masks)):
        if curr_img_all_skeletons_valid_masks[curr_img_all_skeletons_valid_masks_idx] == 1:
            curr_img_all_skeletons_annorect.append(curr_img_all_skeletons['annorect'][curr_img_all_skeletons_valid_masks_idx])
    curr_img_all_skeletons['annorect'] = curr_img_all_skeletons_annorect

    return curr_img_all_skeletons, curr_video_bboxes_curr_frame_corrected + curr_video_bboxes_other_frames

# def vis_curr_img_joints_keypoints(curr_img_all_skeletons, curr_video_bboxes, dst_vis_dir, video_name, curr_img_name, curr_img_keypoints = []):
#     for curr_person_skeleton_idx in range(len(curr_img_all_skeletons['annorect'])):
#         curr_img = cv2.imread(os.path.join(src_video_dir, video_name.split('_backward')[0].split('_forward')[0], curr_img_name.split('/')[-1]))
#         curr_person_skeleton = curr_img_all_skeletons['annorect'][curr_person_skeleton_idx]
#         curr_person_bbox = [x for x in curr_video_bboxes if (curr_img_name in x['image_name'])][curr_person_skeleton_idx]['bbox']
#         joint_list_coord = curr_person_skeleton['annopoints'][0]['point']
#         left, top, width, height = curr_person_bbox[0], curr_person_bbox[1], curr_person_bbox[2], curr_person_bbox[3]
#         cv2.rectangle(curr_img, pt1=(int(left), int(top)), pt2=(int(left + width), int(top + height)), color=(0, 255, 0), thickness=1)
#         for curr_joint in joint_list_coord:
#             radius, thickness, color = 2, 2, (0, 255, 0) # keypoints_colors[joint_list_coord.index(curr_joint)]  # (255, 0, 0)
#             curr_img = cv2.circle(curr_img, (int(curr_joint['x'][0]), int(curr_joint['y'][0])), radius, color, thickness)
#         if not os.path.exists(os.path.join(dst_vis_dir, video_name)):
#             os.mkdir(os.path.join(dst_vis_dir, video_name))
#         cv2.imwrite(os.path.join(dst_vis_dir, video_name, curr_img_name.split('/')[-1].split('.')[0] + '_' + str(curr_person_skeleton_idx) + '.jpg'), curr_img)
#     curr_img = cv2.imread(os.path.join(src_video_dir, video_name.split('_backward')[0].split('_forward')[0], curr_img_name.split('/')[-1]))
#     if len(curr_img_keypoints) != 0:
#         curr_img_double = cv2.resize(curr_img, (0, 0), fx=3, fy=3)
#         for curr_img_keypoint in curr_img_keypoints:
#             curr_img_double = cv2.circle(curr_img_double, (int(curr_img_keypoint[0] * 3), int(curr_img_keypoint[1] * 3)), 1, (0, 255, 0), 2)
#         cv2.imwrite(os.path.join(dst_vis_dir, video_name, next_img_name.split('/')[-1].split('.')[0] + '_samples' + '.jpg'), curr_img_double)
#     else:
#         cv2.imwrite(os.path.join(dst_vis_dir, video_name, curr_img_name.split('/')[-1].split('.')[0] + '_samples' + '.jpg'), curr_img)
#
# def vis_curr_img_joints_keypoints_format2(curr_img_expanded_bboxes_propagated_to_latter_frame_skeletons, curr_img_expanded_bboxes_propagated_to_latter_frame, dst_vis_dir, video_name, next_img_name, curr_img_expanded_samples_propagated_to_next_img=[]):
#     for curr_person_skeleton_idx in range(len(curr_img_expanded_bboxes_propagated_to_latter_frame_skeletons)):
#         curr_img = cv2.imread(os.path.join(src_video_dir, video_name.split('_backward')[0].split('_forward')[0], next_img_name.split('/')[-1]))
#         curr_person_skeleton = curr_img_expanded_bboxes_propagated_to_latter_frame_skeletons[curr_person_skeleton_idx]
#         curr_person_bbox = curr_img_expanded_bboxes_propagated_to_latter_frame[curr_person_skeleton_idx]
#         joint_list_coord = curr_person_skeleton
#         left, top, width, height = curr_person_bbox[0], curr_person_bbox[1], curr_person_bbox[2], curr_person_bbox[3]
#         cv2.rectangle(curr_img, pt1=(int(left), int(top)), pt2=(int(left + width), int(top + height)), color=(255, 0, 0), thickness=1)
#         for curr_joint_idx in range(joint_list_coord.shape[0]):
#             curr_joint = joint_list_coord[curr_joint_idx]
#             radius, thickness, color = 2, 2, keypoints_colors[curr_joint_idx]  # (255, 0, 0)
#             curr_img = cv2.circle(curr_img, (int(curr_joint[1]), int(curr_joint[0])), radius, color, thickness)
#         if not os.path.exists(os.path.join(dst_vis_dir, video_name)):
#             os.mkdir(os.path.join(dst_vis_dir, video_name))
#         cv2.imwrite(os.path.join(dst_vis_dir, video_name, next_img_name.split('/')[-1].split('.')[0] + '_' + str(curr_person_skeleton_idx) + '.jpg'), curr_img)
#     curr_img = cv2.imread(os.path.join(src_video_dir, video_name.split('_backward')[0].split('_forward')[0], next_img_name.split('/')[-1]))
#     if len(curr_img_expanded_samples_propagated_to_next_img) != 0:
#         for curr_img_keypoint in curr_img_expanded_samples_propagated_to_next_img:
#             curr_img_double = cv2.circle(cv2.resize(curr_img, (0, 0), fx=2, fy=2), (int(curr_img_keypoint[0]), int(curr_img_keypoint[1])), 1, (0, 0, 255), 2)
#         cv2.imwrite(os.path.join(dst_vis_dir, video_name, next_img_name.split('/')[-1].split('.')[0] + '_samples' + '.jpg'), curr_img_double)
#     else:
#         cv2.imwrite(os.path.join(dst_vis_dir, video_name, next_img_name.split('/')[-1].split('.')[0] + '_samples' + '.jpg'), curr_img)

def generate_distinct_colors(n):
    """
    Generate n distinct colors in RGB format with maximized differences.
    Parameters:
    - n: Number of distinct colors to generate.
    Returns:
    - List of colors in RGB format.
    """
    colors = []
    # Create colors with evenly spaced hues and random saturation and brightness
    for i in range(n):
        hue = i / n  # Evenly space the hue
        saturation = random.uniform(0.5, 1.0)  # Keep saturation high for vivid colors
        value = random.uniform(0.5, 1.0)  # Vary brightness to add variety
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        # Convert RGB from 0-1 range to 0-255 range
        rgb = tuple(int(x * 255) for x in rgb)
        colors.append(rgb)
    return colors

# curr_img_joints: a list whose sublists are joints of humans
# curr_img_keypoints: a list whose sublists are keypoints of humans
# curr_img_keypoints_mask: the id masks of samples around all humans
def locate_joints_keypoints_allhumans_former_img(former_img_example_bbox_list, former_img, Num_samples_former_img, former_img_example_skeleton_list, curr_img_joints, curr_img_keypoints, curr_img_keypoints_mask, max_motion_vector, joint_color, keypoint_color, draw_rectangle_or_not=1):
    assert(len(former_img_example_bbox_list) == len(former_img_example_skeleton_list))
    vis_img_joints = copy.deepcopy(former_img)
    vis_img_keypoints = copy.deepcopy(former_img)
    for human_idx in range(len(former_img_example_bbox_list)):
        former_img_example_skeleton = former_img_example_skeleton_list[human_idx]
        # curr_img_keypoints_curr_human: the samples around curr humans
        # curr_img_keypoints_mask_curr_human: the joint semantic id masks of samples around curr humans
        # curr_img_joints_curr_human: the joints of curr human
        curr_img_keypoints_mask_curr_human, curr_img_joints_curr_human, curr_img_keypoints_curr_human = [], [], []

        radius, thickness = 1, 1
        # traverse body joints
        for former_img_example_skeleton_ele in former_img_example_skeleton:
            vis_img_joints = cv2.circle(vis_img_joints, (int(former_img_example_skeleton_ele['x'][0]), int(former_img_example_skeleton_ele['y'][0])), radius, joint_color[human_idx], thickness)
            curr_img_joints_curr_human.append([former_img_example_skeleton_ele['x'][0], former_img_example_skeleton_ele['y'][0], former_img_example_skeleton_ele['score'][0], former_img_example_skeleton_ele['id'][0]])

        # after traversing all joints of all humans, organize curr_img_joints_curr_human to curr_img_keypoints_curr_human
        # the distance between each joint and its nearest neighbor
        nearest_neighbor_distance_list = []
        # the number of key-points around each joint
        num_samples_around_curr_joint_list = []
        for curr_img_joints_ele in curr_img_joints_curr_human:
            nearest_neighbor_distance_list.append(min([np.linalg.norm(np.array(y[:2]) - np.array(curr_img_joints_ele[:2])) for y in [x for x in curr_img_joints_curr_human if (x != curr_img_joints_ele)]]))
            # the numbers of samples around joints in curr human 
            num_samples_around_curr_joint_list.append(math.ceil(Num_samples_former_img / len(curr_img_joints_curr_human) * 1))
        # sample around joints, the joint with max number of surrounding keypoints has a sample radius of max_motion_vector, the radii of other joints are proportional to their number of surrounding keypoints
        for curr_img_joints_ele in curr_img_joints_curr_human:
            num_samples_around_curr_joint = math.ceil(Num_samples_former_img / len(curr_img_joints_curr_human) * 1)
            sampling_around_joint_return_samples_result = sampling_around_joint_return_samples([curr_img_joints_ele[0], curr_img_joints_ele[1]],
                                                                                               num_samples_around_curr_joint,
                                                                                               max_motion_vector / max(num_samples_around_curr_joint_list) * num_samples_around_curr_joint) # nearest_neighbor_distance / 2)
            curr_img_keypoints_curr_human += sampling_around_joint_return_samples_result
            curr_img_keypoints_mask_curr_human += [curr_img_joints_ele[3]] * len(sampling_around_joint_return_samples_result)
        for curr_img_keypoint in curr_img_keypoints_curr_human:
            vis_img_keypoints = cv2.circle(vis_img_keypoints, (int(curr_img_keypoint[0]), int(curr_img_keypoint[1])), 1, keypoint_color[human_idx], 1)
        curr_img_keypoints_mask.append(curr_img_keypoints_mask_curr_human)
        curr_img_joints.append(curr_img_joints_curr_human)
        curr_img_keypoints.append(curr_img_keypoints_curr_human)
        if draw_rectangle_or_not:
            vis_img_joints = cv2.rectangle(vis_img_joints, (int(former_img_example_bbox_list[human_idx]['bbox'][0]), int(former_img_example_bbox_list[human_idx]['bbox'][1])),
                                           (int(former_img_example_bbox_list[human_idx]['bbox'][0] + former_img_example_bbox_list[human_idx]['bbox'][2]), int(former_img_example_bbox_list[human_idx]['bbox'][1] + former_img_example_bbox_list[human_idx]['bbox'][3])), joint_color[human_idx], 1)
        if draw_rectangle_or_not:
            vis_img_keypoints = cv2.rectangle(vis_img_keypoints, (int(former_img_example_bbox_list[human_idx]['bbox'][0]), int(former_img_example_bbox_list[human_idx]['bbox'][1])),
                                              (int(former_img_example_bbox_list[human_idx]['bbox'][0] + former_img_example_bbox_list[human_idx]['bbox'][2]), int(former_img_example_bbox_list[human_idx]['bbox'][1] + former_img_example_bbox_list[human_idx]['bbox'][3])), joint_color[human_idx], 1)
    return curr_img_joints, curr_img_keypoints, curr_img_keypoints_mask, vis_img_joints, vis_img_keypoints

def reformat_skeletons_boxes(next_img_propagated_and_detected_bboxes_skeleton, next_img_propagated_and_detected_bboxes):
    return next_img_propagated_and_detected_bboxes_skeleton, next_img_propagated_and_detected_bboxes

def ot_matching_matrix(matching_matrix):
    if matching_matrix.shape[0] > matching_matrix.shape[1]:
        add_width = matching_matrix.shape[0] - matching_matrix.shape[1]
        matching_matrix = np.concatenate((matching_matrix, np.ones((matching_matrix.shape[0], add_width)) * np.max(matching_matrix) * 2), axis=1)
        ot_src = np.array([1.0] * matching_matrix.shape[0])
        ot_dst = np.array([1.0] * matching_matrix.shape[1])
        prev_curr_transportation_array = ot.emd(ot_src, ot_dst, matching_matrix)
        prev_curr_transportation_array = prev_curr_transportation_array[:, :prev_curr_transportation_array.shape[1] - add_width]
    elif matching_matrix.shape[0] < matching_matrix.shape[1]:
        add_height = matching_matrix.shape[1] - matching_matrix.shape[0]
        matching_matrix = np.concatenate((matching_matrix, np.ones((add_height, matching_matrix.shape[1])) * np.max(matching_matrix) * 2), axis=0)
        ot_src = np.array([1.0] * matching_matrix.shape[0])
        ot_dst = np.array([1.0] * matching_matrix.shape[1])
        prev_curr_transportation_array = ot.emd(ot_src, ot_dst, matching_matrix)
        prev_curr_transportation_array = prev_curr_transportation_array[:prev_curr_transportation_array.shape[0] - add_height, :]
    else:
        ot_src = np.array([1.0] * matching_matrix.shape[0])
        ot_dst = np.array([1.0] * matching_matrix.shape[1])
        prev_curr_transportation_array = ot.emd(ot_src, ot_dst, matching_matrix)
    return prev_curr_transportation_array

def merge_box(curr_img_boxes, next_img_boxes): # left, top, width, height
    curr_left = curr_img_boxes[0]
    curr_top = curr_img_boxes[1]
    curr_right = curr_img_boxes[0] + curr_img_boxes[2]
    curr_bottom = curr_img_boxes[1] + curr_img_boxes[3]

    next_left = next_img_boxes[0]
    next_top = next_img_boxes[1]
    next_right = next_img_boxes[0] + next_img_boxes[2]
    next_bottom = next_img_boxes[1] + next_img_boxes[3]

    return [min([curr_left, next_left]), min([curr_top, next_top]), max([curr_right, next_right]) - min([curr_left, next_left]), max([curr_bottom, next_bottom]) - min([curr_top, next_top])]

def convert_from_17_to_15(curr_human_anno, img_height, img_width):
    curr_human_anno = curr_human_anno[0, :, :]
    curr_human_anno_converted = {}
    for curr_human_anno_item in curr_human_anno:
        if curr_human_anno.tolist().index(curr_human_anno_item.tolist()) in base_pose_model_semantics.keys() and \
                base_pose_model_semantics[curr_human_anno.tolist().index(curr_human_anno_item.tolist())] in posetrack_submission_semantic_to_index:
            curr_human_anno_converted[posetrack_submission_semantic_to_index[base_pose_model_semantics[curr_human_anno.tolist().index(curr_human_anno_item.tolist())]]] = curr_human_anno_item[:3]

    # for curr_human_anno_converted_key in curr_human_anno_converted:
    #     curr_human_anno_converted[curr_human_anno_converted_key] = np.array([curr_human_anno_converted[curr_human_anno_converted_key][1], \
    #                                                                          curr_human_anno_converted[curr_human_anno_converted_key][0], \
    #                                                                          curr_human_anno_converted[curr_human_anno_converted_key][2]])
    return curr_human_anno_converted

def convert_from_17_to_15_v1(curr_human_anno, img_height, img_width):
    curr_human_anno = curr_human_anno[0, :, :]
    curr_human_anno_converted = {}
    for curr_human_anno_item in curr_human_anno:
        if curr_human_anno.tolist().index(curr_human_anno_item.tolist()) in hrnet_semantics.keys() and \
                hrnet_semantics[curr_human_anno.tolist().index(curr_human_anno_item.tolist())] in posetrack_submission_semantic_to_index:
            curr_human_anno_converted[posetrack_submission_semantic_to_index[hrnet_semantics[curr_human_anno.tolist().index(curr_human_anno_item.tolist())]]] = curr_human_anno_item[:3]
    if (9 in curr_human_anno_converted) and (8 in curr_human_anno_converted):
        left_shoulder_coord = curr_human_anno_converted[9]
        right_shoulder_coord = curr_human_anno_converted[8]
        neck_coord = (left_shoulder_coord + right_shoulder_coord) / 2
        curr_human_anno_converted[12] = neck_coord
    if (13 in curr_human_anno_converted) and (12 in curr_human_anno_converted):
        curr_human_anno_converted[14] = np.array([0.0] * 3)
        curr_human_anno_converted[14][0] = (curr_human_anno[1][0] + curr_human_anno[2][0]) / 2 # 1.5 * curr_human_anno_converted[13][0] - 0.5 * curr_human_anno_converted[12][0]
        curr_human_anno_converted[14][1] = (curr_human_anno[1][1] + curr_human_anno[2][1]) / 2 # 1.5 * curr_human_anno_converted[13][1] - 0.5 * curr_human_anno_converted[12][1]
        curr_human_anno_converted[14][2] = (curr_human_anno_converted[13][2] + curr_human_anno_converted[12][2]) / 2
        curr_human_anno_converted[14][0] = min([max([curr_human_anno_converted[14][0], 0]), img_height - 1])
        curr_human_anno_converted[14][1] = min([max([curr_human_anno_converted[14][1], 0]), img_width - 1])
    for curr_human_anno_converted_key in curr_human_anno_converted:
        curr_human_anno_converted[curr_human_anno_converted_key] = np.array([curr_human_anno_converted[curr_human_anno_converted_key][1], \
                                                                             curr_human_anno_converted[curr_human_anno_converted_key][0], \
                                                                             curr_human_anno_converted[curr_human_anno_converted_key][2]])
    return curr_human_anno_converted

def generate_gaussian_kernel(width, height, gaussian_center_hori, gaussian_center_vert, isotropicGrayscaleImage):
    step8_pred_heatmap = np.zeros((height, width)).astype('uint8')
    curr_joint_coord, curr_joint_conf = [int(round(gaussian_center_hori)), int(round(gaussian_center_vert))], 0.8 + 0.2 * random.random()
    curr_joint_kernel = copy.deepcopy(isotropicGrayscaleImage)  # heatmap_hills_lib[kernel_idx, :, :]
    curr_joint_kernel = curr_joint_kernel / np.max(curr_joint_kernel)
    curr_joint_kernel = curr_joint_kernel * curr_joint_conf
    curr_joint_kernel[np.where(curr_joint_kernel < 0)] = 0
    curr_joint_kernel = curr_joint_kernel * 255
    curr_joint_kernel = curr_joint_kernel.astype('uint8')
    curr_joint_kernel_hori, curr_joint_kernel_vert = cv2.moments(curr_joint_kernel)['m10'] / \
                                                     cv2.moments(curr_joint_kernel)['m00'], \
                                                     cv2.moments(curr_joint_kernel)['m01'] / \
                                                     cv2.moments(curr_joint_kernel)['m00']
    step8_pred_heatmap[np.where(curr_joint_kernel > 0)[0] - int(curr_joint_kernel_vert) + curr_joint_coord[1],
                       np.where(curr_joint_kernel > 0)[1] - int(curr_joint_kernel_hori) + curr_joint_coord[0]] += curr_joint_kernel[np.where(curr_joint_kernel > 0)[0], np.where(curr_joint_kernel > 0)[1]]
    return step8_pred_heatmap

def generate_gaussian_kernel_v2(width, height, gaussian_center_hori, gaussian_center_vert, isotropicGrayscaleImage, heatmap_peak):
    step8_pred_heatmap = np.zeros((height, width)).astype('uint8')
    curr_joint_coord, curr_joint_conf = [int(round(gaussian_center_hori)), int(round(gaussian_center_vert))], heatmap_peak
    curr_joint_kernel = copy.deepcopy(isotropicGrayscaleImage)  # heatmap_hills_lib[kernel_idx, :, :]
    curr_joint_kernel = curr_joint_kernel / np.max(curr_joint_kernel)
    curr_joint_kernel = curr_joint_kernel * curr_joint_conf
    curr_joint_kernel[np.where(curr_joint_kernel < 0)] = 0
    curr_joint_kernel = curr_joint_kernel * 255
    curr_joint_kernel = curr_joint_kernel.astype('uint8')
    curr_joint_kernel_hori, curr_joint_kernel_vert = cv2.moments(curr_joint_kernel)['m10'] / \
                                                     cv2.moments(curr_joint_kernel)['m00'], \
                                                     cv2.moments(curr_joint_kernel)['m01'] / \
                                                     cv2.moments(curr_joint_kernel)['m00']
    step8_pred_heatmap[np.where(curr_joint_kernel > 0)[0] - int(curr_joint_kernel_vert) + curr_joint_coord[1],
                       np.where(curr_joint_kernel > 0)[1] - int(curr_joint_kernel_hori) + curr_joint_coord[0]] += curr_joint_kernel[np.where(curr_joint_kernel > 0)[0], np.where(curr_joint_kernel > 0)[1]]
    return step8_pred_heatmap

def convert_from_17_to_15_joints_and_heatmaps(curr_human_anno, heatmaps, isotropicGrayscaleImage, img_width, img_height):
    curr_human_anno = curr_human_anno[0, :, :]
    curr_human_anno_converted, add_heatmaps, curr_human_anno_result = {}, {}, []
    heatmaps_result = np.zeros((heatmaps.shape[0], len(posetrack_submission_semantic_to_index), heatmaps.shape[2], heatmaps.shape[3]))
    for curr_human_anno_item in curr_human_anno:
        if curr_human_anno.tolist().index(curr_human_anno_item.tolist()) in manually_labelled_semantics_16_points.keys():
            curr_human_anno_converted[posetrack_submission_semantic_to_index[manually_labelled_semantics_16_points[curr_human_anno.tolist().index(curr_human_anno_item.tolist())]]] = curr_human_anno_item#[:2]
            heatmaps_result[0, posetrack_submission_semantic_to_index[manually_labelled_semantics_16_points[curr_human_anno.tolist().index(curr_human_anno_item.tolist())]], :, :] = heatmaps[0, curr_human_anno.tolist().index(curr_human_anno_item.tolist()), :, :]
    if (8 in curr_human_anno_converted) and (9 in curr_human_anno_converted):
        left_shoulder_coord = curr_human_anno_converted[8]
        right_shoulder_coord = curr_human_anno_converted[9]
        neck_coord = (left_shoulder_coord + right_shoulder_coord) / 2
        curr_human_anno_converted[12] = neck_coord
        add_heatmaps[12] = generate_gaussian_kernel(img_width, img_height, neck_coord[1], neck_coord[0], isotropicGrayscaleImage)
    if (13 in curr_human_anno_converted) and (12 in curr_human_anno_converted):
        curr_human_anno_converted[14] = [2 * curr_human_anno_converted[13][0] - curr_human_anno_converted[12][0], curr_human_anno_converted[12][1], (curr_human_anno_converted[13][2] + curr_human_anno_converted[12][2]) / 2] # 2 * curr_human_anno_converted[1] - curr_human_anno_converted[2]
        add_heatmaps[14] = generate_gaussian_kernel(img_width, img_height, curr_human_anno_converted[14][1], curr_human_anno_converted[14][0], isotropicGrayscaleImage)
    for curr_human_anno_converted_key in curr_human_anno_converted:
        ## coco order: vert-hori, posetrack order: hori-vert
        curr_human_anno_converted[curr_human_anno_converted_key] = np.array([curr_human_anno_converted[curr_human_anno_converted_key][1], curr_human_anno_converted[curr_human_anno_converted_key][0], curr_human_anno_converted[curr_human_anno_converted_key][2]])
    for curr_human_anno_converted_key in sorted(curr_human_anno_converted.keys()):
        curr_human_anno_result.append(curr_human_anno_converted[curr_human_anno_converted_key])
    return np.expand_dims(np.array(curr_human_anno_result), axis=0), heatmaps_result, add_heatmaps

def compute_float_coord_optical_flow(former_img_keypoint, curr_forward_flow):
    former_img_keypoint_left, former_img_keypoint_right, former_img_keypoint_top, former_img_keypoint_bottom = \
        int(former_img_keypoint[0]), int(former_img_keypoint[0]) + 1, int(former_img_keypoint[1]), int(
            former_img_keypoint[1]) + 1
    former_img_keypoint_lefttop_motion = curr_forward_flow[:, former_img_keypoint_top, former_img_keypoint_left]
    former_img_keypoint_leftbottom_motion = curr_forward_flow[:, former_img_keypoint_bottom, former_img_keypoint_left]
    former_img_keypoint_righttop_motion = curr_forward_flow[:, former_img_keypoint_top, former_img_keypoint_right]
    former_img_keypoint_rightbottom_motion = curr_forward_flow[:, former_img_keypoint_bottom, former_img_keypoint_right]

    overall_weight = (1 / ((former_img_keypoint[0] - former_img_keypoint_left) ** 2 + (former_img_keypoint[1] - former_img_keypoint_top) ** 2)) + \
                     (1 / ((former_img_keypoint[0] - former_img_keypoint_left) ** 2 + (former_img_keypoint[1] - former_img_keypoint_bottom) ** 2)) + \
                     (1 / ((former_img_keypoint[0] - former_img_keypoint_right) ** 2 + (former_img_keypoint[1] - former_img_keypoint_top) ** 2)) + \
                     (1 / ((former_img_keypoint[0] - former_img_keypoint_right) ** 2 + (former_img_keypoint[1] - former_img_keypoint_bottom) ** 2))

    lefttop_weight = (1 / ((former_img_keypoint[0] - former_img_keypoint_left) ** 2 + (former_img_keypoint[1] - former_img_keypoint_top) ** 2)) / overall_weight
    leftbottom_weight = (1 / ((former_img_keypoint[0] - former_img_keypoint_left) ** 2 + (former_img_keypoint[1] - former_img_keypoint_bottom) ** 2)) / overall_weight
    righttop_weight = (1 / ((former_img_keypoint[0] - former_img_keypoint_right) ** 2 + (former_img_keypoint[1] - former_img_keypoint_top) ** 2)) / overall_weight
    rightbottom_weight = (1 / ((former_img_keypoint[0] - former_img_keypoint_right) ** 2 + (former_img_keypoint[1] - former_img_keypoint_bottom) ** 2)) / overall_weight

    former_img_keypoint_motion = former_img_keypoint_lefttop_motion * lefttop_weight + former_img_keypoint_leftbottom_motion * leftbottom_weight + \
                                 former_img_keypoint_righttop_motion * righttop_weight + former_img_keypoint_rightbottom_motion * rightbottom_weight

    return former_img_keypoint_motion

def select_from_twice_propagated_samples(latter_img_keypoints_twice, latter_img_keypoints_mask, latter_img_human_idx, former_img_keypoints, former_img_keypoints_mask, former_img_human_idx, curr_forward_flow, former_img, latter_img):
    former_img_human_idx_keypoints_propagated = keypoint_propagation(former_img_keypoints[former_img_human_idx], curr_forward_flow)
    # match
    former_human_to_latter_human_per_joint_matching_error, former_human_to_latter_human_per_joint_visibility_score = {}, {}
    for joint_semantic in np.unique(former_img_keypoints_mask[former_img_human_idx]):
        curr_former_human_keypoint_coords = np.array(former_img_human_idx_keypoints_propagated)[np.where(former_img_keypoints_mask[former_img_human_idx] == joint_semantic)]
        curr_latter_human_keypoint_coords = np.array(latter_img_keypoints_twice[latter_img_human_idx])[np.where(latter_img_keypoints_mask[latter_img_human_idx] == joint_semantic)]
        former_latter_humans_keypoint_matching = np.zeros((len(curr_former_human_keypoint_coords), len(curr_latter_human_keypoint_coords)))
        for curr_former_human_keypoint_coord_idx in range(len(curr_former_human_keypoint_coords)):
            for curr_latter_human_keypoint_coord_idx in range(len(curr_latter_human_keypoint_coords)):
                former_latter_humans_keypoint_matching[curr_former_human_keypoint_coord_idx, curr_latter_human_keypoint_coord_idx] = \
                    np.linalg.norm(curr_former_human_keypoint_coords[curr_former_human_keypoint_coord_idx] - curr_latter_human_keypoint_coords[curr_latter_human_keypoint_coord_idx]) ** 2
        former_latter_humans_keypoint_association = ot_matching_matrix(former_latter_humans_keypoint_matching)
        former_human_to_latter_human_per_joint_matching_error[joint_semantic] = np.sum(np.multiply(former_latter_humans_keypoint_association, former_latter_humans_keypoint_matching))

    ##############################################################################################################################################################
    ######################################## compute visibility scores with graph distance minimization ##########################################################
    ##############################################################################################################################################################
    former_human_to_latter_human_per_joint_matching_error_ori = copy.deepcopy(former_human_to_latter_human_per_joint_matching_error)
    add_term_to_avoid_division_by_zero = min([former_human_to_latter_human_per_joint_matching_error[x] for x in former_human_to_latter_human_per_joint_matching_error if former_human_to_latter_human_per_joint_matching_error[x] > 0]) / 2
    for joint_semantic in np.unique(former_img_keypoints_mask[former_img_human_idx]):
        former_human_to_latter_human_per_joint_matching_error[joint_semantic] = former_human_to_latter_human_per_joint_matching_error[joint_semantic] + add_term_to_avoid_division_by_zero
    former_human_to_latter_human_per_joint_matching_error_normalization = sum([former_human_to_latter_human_per_joint_matching_error[x] for x in former_human_to_latter_human_per_joint_matching_error])
    for joint_semantic in np.unique(former_img_keypoints_mask[former_img_human_idx]):
        former_human_to_latter_human_per_joint_visibility_score[joint_semantic] = former_human_to_latter_human_per_joint_matching_error_normalization / former_human_to_latter_human_per_joint_matching_error[joint_semantic]
    former_human_to_latter_human_per_joint_visibility_score_normalization = sum([former_human_to_latter_human_per_joint_visibility_score[x] for x in former_human_to_latter_human_per_joint_visibility_score])
    for joint_semantic in np.unique(former_img_keypoints_mask[former_img_human_idx]):
        former_human_to_latter_human_per_joint_visibility_score[joint_semantic] = former_human_to_latter_human_per_joint_visibility_score[joint_semantic] / former_human_to_latter_human_per_joint_visibility_score_normalization
    human_to_human_overall_matching_error = 0
    for joint_semantic in np.unique(former_img_keypoints_mask[former_img_human_idx]):
        human_to_human_overall_matching_error += former_human_to_latter_human_per_joint_visibility_score[joint_semantic] * former_human_to_latter_human_per_joint_matching_error_ori[joint_semantic]
    # return overall human-human matching error and visibility scores of joints, the overall error is the weighted sum of joint-level error, ignoring low-vis joints
    return human_to_human_overall_matching_error, former_human_to_latter_human_per_joint_visibility_score

def mask_out_low_vis_regions_and_pose_estimation(latter_img_joint_level_vis_scores, latter_img_joints, latter_img_skeletons, latter_img_bboxes, latter_img, latter_img_dir, self_inference_pose_model_model):
    latter_img_joints_backup = copy.deepcopy(latter_img_joints)
    latter_img_inter_bbox_iou = np.zeros((len(latter_img_bboxes), len(latter_img_bboxes)))
    for latter_img_bbox_row_idx in range(len(latter_img_bboxes)):
        for latter_img_bbox_col_idx in range(len(latter_img_bboxes)):
            row_bbox_left, row_bbox_top, row_bbox_width, row_bbox_height = latter_img_bboxes[latter_img_bbox_row_idx]['bbox'][0], \
                                                                           latter_img_bboxes[latter_img_bbox_row_idx]['bbox'][1], \
                                                                           latter_img_bboxes[latter_img_bbox_row_idx]['bbox'][2], \
                                                                           latter_img_bboxes[latter_img_bbox_row_idx]['bbox'][3]

            col_bbox_left, col_bbox_top, col_bbox_width, col_bbox_height = latter_img_bboxes[latter_img_bbox_col_idx]['bbox'][0], \
                                                                           latter_img_bboxes[latter_img_bbox_col_idx]['bbox'][1], \
                                                                           latter_img_bboxes[latter_img_bbox_col_idx]['bbox'][2], \
                                                                           latter_img_bboxes[latter_img_bbox_col_idx]['bbox'][3]

            latter_img_inter_bbox_iou[latter_img_bbox_row_idx, latter_img_bbox_col_idx] = \
                compute_iou_single_box_normal([row_bbox_top, row_bbox_top + row_bbox_height, row_bbox_left, row_bbox_left + row_bbox_width], \
                                              [col_bbox_top, col_bbox_top + col_bbox_height, col_bbox_left, col_bbox_left + col_bbox_width])
    # for each row in latter_img_inter_bbox_iou, determine the columns with iou > threshold, and corresponding alien sets
    for latter_img_bbox_idx in range(len(latter_img_bboxes)):
        whether_undergoes_valid_modification = 0
        if latter_img_joint_level_vis_scores[latter_img_bbox_idx] is None:
            continue
        latter_img_to_be_modified = copy.deepcopy(latter_img)
        curr_bbox_left, curr_bbox_top, curr_bbox_width, curr_bbox_height = latter_img_bboxes[latter_img_bbox_idx]['bbox'][0], latter_img_bboxes[latter_img_bbox_idx]['bbox'][1], \
                                                                           latter_img_bboxes[latter_img_bbox_idx]['bbox'][2], latter_img_bboxes[latter_img_bbox_idx]['bbox'][3]
        # unmasked_heatmaps, unmasked_joints = model.predict(latter_img, curr_bbox_left, curr_bbox_left + curr_bbox_width, curr_bbox_top, curr_bbox_top + curr_bbox_height)
        unmasked_joints = self_inference_pose_model({'image_name': latter_img_dir, 'image': latter_img, 'nframes': len(os.listdir(latter_img_dir.split(latter_img_dir.split('/')[-1])[0]))}, curr_bbox_left, curr_bbox_top, curr_bbox_width, curr_bbox_height, self_inference_pose_model_model)
        for latter_img_neighbor_bbox_idx in [x for x in np.where(latter_img_inter_bbox_iou[latter_img_bbox_idx, :] > 0.1)[0].tolist() if x != latter_img_bbox_idx]:
            if latter_img_joint_level_vis_scores[latter_img_neighbor_bbox_idx] is None:
                continue
            # list alien joints
            curr_alien_set = []
            neighbor_bbox_left, neighbor_bbox_top, neighbor_bbox_width, neighbor_bbox_height = latter_img_bboxes[latter_img_neighbor_bbox_idx]['bbox'][0], latter_img_bboxes[latter_img_neighbor_bbox_idx]['bbox'][1], \
                                                                                               latter_img_bboxes[latter_img_neighbor_bbox_idx]['bbox'][2], latter_img_bboxes[latter_img_neighbor_bbox_idx]['bbox'][3]
            for neighbor_joint_idx in range(len(latter_img_joints_backup[latter_img_neighbor_bbox_idx])):

                neighbor_joint_idx_hori, neighbor_joint_idx_vert = latter_img_joints_backup[latter_img_neighbor_bbox_idx][neighbor_joint_idx][0], latter_img_joints_backup[latter_img_neighbor_bbox_idx][neighbor_joint_idx][1]
                if neighbor_joint_idx_hori >= curr_bbox_left and neighbor_joint_idx_hori < curr_bbox_left + curr_bbox_width and \
                    neighbor_joint_idx_vert >= curr_bbox_top and neighbor_joint_idx_vert < curr_bbox_top + curr_bbox_height and \
                    neighbor_joint_idx_hori >= neighbor_bbox_left and neighbor_joint_idx_hori < neighbor_bbox_left + neighbor_bbox_width and \
                    neighbor_joint_idx_vert >= neighbor_bbox_top and neighbor_joint_idx_vert < neighbor_bbox_top + neighbor_bbox_height:
                    curr_alien_set.append([neighbor_joint_idx_hori, neighbor_joint_idx_vert, latter_img_joint_level_vis_scores[latter_img_neighbor_bbox_idx][neighbor_joint_idx]])
            # till now, we have obtained alien joints, all native joints
            curr_alien_set = sorted(curr_alien_set, key=lambda x: x[2]) #, reverse=True)
            # native joints outside common area
            native_joints_outside_common_area, native_joints_inside_common_area, native_joints = [], [], []
            for curr_joint_idx in range(len(latter_img_joints_backup[latter_img_bbox_idx])):
                native_joints.append(latter_img_joints_backup[latter_img_bbox_idx][curr_joint_idx][:2] + [latter_img_joint_level_vis_scores[latter_img_bbox_idx][curr_joint_idx]])
                curr_joint_idx_hori, curr_joint_idx_vert = latter_img_joints_backup[latter_img_bbox_idx][curr_joint_idx][0], latter_img_joints_backup[latter_img_bbox_idx][curr_joint_idx][1]
                if curr_joint_idx_hori >= neighbor_bbox_left and curr_joint_idx_hori < neighbor_bbox_left + neighbor_bbox_width and \
                    curr_joint_idx_vert >= neighbor_bbox_top and curr_joint_idx_vert < neighbor_bbox_top + neighbor_bbox_height:
                    native_joints_inside_common_area.append(latter_img_joints_backup[latter_img_bbox_idx][curr_joint_idx][:2] + [latter_img_joint_level_vis_scores[latter_img_bbox_idx][curr_joint_idx]])
                else:
                    native_joints_outside_common_area.append(latter_img_joints_backup[latter_img_bbox_idx][curr_joint_idx][:2] + [latter_img_joint_level_vis_scores[latter_img_bbox_idx][curr_joint_idx]])
            if len(curr_alien_set) < 3: # do not need len(native_joints_inside_common_area) > 0, if no native joints, we can still mask out
                continue
            if len(native_joints_inside_common_area) == 0:
                continue
            while (len(curr_alien_set) >= 3) and (len(native_joints_inside_common_area) > 0) and (np.mean([x[2] for x in curr_alien_set]) < np.mean([x[2] for x in native_joints_inside_common_area])):
                curr_alien_set = curr_alien_set[1:]
            if len(curr_alien_set) >= 3:
                curr_alien_set_row = np.array(curr_alien_set).astype(int)[:, 1:2].transpose()[0]
                curr_alien_set_col = np.array(curr_alien_set).astype(int)[:, :1].transpose()[0]
                curr_alien_set_row_collect, curr_alien_set_col_collect = polygon(curr_alien_set_row, curr_alien_set_col)
                mask_for_convex_hull = np.zeros_like(latter_img_to_be_modified[:, :, 0])
                # curr_alien_set_row_collect = min([max([curr_alien_set_row_collect, 0]), latter_img_to_be_modified.shape[0] - 1])
                # curr_alien_set_col_collect = min([max([curr_alien_set_col_collect, 0]), latter_img_to_be_modified.shape[1] - 1])
                mask_for_convex_hull[curr_alien_set_row_collect, curr_alien_set_col_collect] = 255
                mask_for_convex_hull = convex_hull_image(mask_for_convex_hull)
                latter_img_to_be_modified_backup = copy.deepcopy(latter_img_to_be_modified) # if current neighbor mask is not good, turn back to backup
                latter_img_to_be_modified[:, :, 0][np.where(mask_for_convex_hull == True)] = 128 # [curr_alien_set_row_collect, curr_alien_set_col_collect] = 128
                latter_img_to_be_modified[:, :, 1][np.where(mask_for_convex_hull == True)] = 128 # [curr_alien_set_row_collect, curr_alien_set_col_collect] = 128
                latter_img_to_be_modified[:, :, 2][np.where(mask_for_convex_hull == True)] = 128 # [curr_alien_set_row_collect, curr_alien_set_col_collect] = 128
                whether_undergoes_valid_modification += 1
                # cv2.fillPoly(latter_img_to_be_modified, [np.array(curr_alien_set).astype(int)[:, :2]], (128, 128, 128))
                # cv2.fillPoly(latter_img_to_be_modified[:, :, 0], np.array(curr_alien_set, dtype=np.int32)[:, :2], (128, 128, 128))
                # compare confidence of masked and unmasked pose estimation
                # masked_heatmaps, masked_joints = model.predict(latter_img_to_be_modified, curr_bbox_left, curr_bbox_left + curr_bbox_width, curr_bbox_top, curr_bbox_top + curr_bbox_height)
                masked_joints = self_inference_pose_model({'image_name': latter_img_dir, 'image': latter_img_to_be_modified, 'nframes': len(os.listdir(latter_img_dir.split(latter_img_dir.split('/')[-1])[0]))}, curr_bbox_left, curr_bbox_top, curr_bbox_width, curr_bbox_height, self_inference_pose_model_model)
                # if confidence of masked predictions outperform unmasked, revise latter_img_joints, latter_img_skeletons
                if np.sum(masked_joints[0][:, 2]) <= np.sum(unmasked_joints[0][:, 2]):
                    latter_img_to_be_modified = copy.deepcopy(latter_img_to_be_modified_backup)

        if whether_undergoes_valid_modification > 0:
            # dump
            latter_img_to_be_modified_vis = copy.deepcopy(latter_img_to_be_modified)
            # masked_heatmaps, masked_joints = model.predict(latter_img_to_be_modified, curr_bbox_left,
            #                                                curr_bbox_left + curr_bbox_width, curr_bbox_top,
            #                                                curr_bbox_top + curr_bbox_height)
            masked_joints = self_inference_pose_model({'image_name': latter_img_dir, 'image': latter_img_to_be_modified, 'nframes': len(os.listdir(latter_img_dir.split(latter_img_dir.split('/')[-1])[0]))}, curr_bbox_left, curr_bbox_top, curr_bbox_width, curr_bbox_height, self_inference_pose_model_model)
            dump_increase_in_conf_by_masking = open('dump_increase_in_conf_by_masking.txt', 'a')
            dump_increase_in_conf_by_masking.write(str(np.sum(masked_joints[0][:, 2]) / np.sum(unmasked_joints[0][:, 2])) + '\n')
            dump_increase_in_conf_by_masking.close()
            if np.sum(masked_joints[0][:, 2]) > (np.sum(unmasked_joints[0][:, 2]) * 1.0): # (np.sum(unmasked_joints[0][:, 2]) * 1.01):
                # hrnet format: first vert coord, then hori coord
                # after convertion: first hori coord, then vert coord
                latter_img_curr_bbox_15_keypoints = convert_from_17_to_15(masked_joints, latter_img.shape[0], latter_img.shape[1])
                for latter_img_joints_idx in range(len(latter_img_joints[latter_img_bbox_idx])):
                    if latter_img_joints_idx in latter_img_curr_bbox_15_keypoints:
                        latter_img_joints[latter_img_bbox_idx][latter_img_joints_idx][:3] = latter_img_curr_bbox_15_keypoints[latter_img_joints_idx]
                for latter_img_joints_idx in range(len(latter_img_skeletons[latter_img_bbox_idx])):
                    if latter_img_skeletons[latter_img_bbox_idx][latter_img_joints_idx]['id'][0] in latter_img_curr_bbox_15_keypoints:
                        latter_img_skeletons[latter_img_bbox_idx][latter_img_joints_idx]['x'][0] = float(latter_img_curr_bbox_15_keypoints[latter_img_skeletons[latter_img_bbox_idx][latter_img_joints_idx]['id'][0]][0])
                        latter_img_skeletons[latter_img_bbox_idx][latter_img_joints_idx]['y'][0] = float(latter_img_curr_bbox_15_keypoints[latter_img_skeletons[latter_img_bbox_idx][latter_img_joints_idx]['id'][0]][1])
                        latter_img_skeletons[latter_img_bbox_idx][latter_img_joints_idx]['score'][0] = float(latter_img_curr_bbox_15_keypoints[latter_img_skeletons[latter_img_bbox_idx][latter_img_joints_idx]['id'][0]][2])
                #         # dump
                #         latter_img_to_be_modified_vis = cv2.circle(latter_img_to_be_modified_vis,
                #                                                    (int(latter_img_skeletons[latter_img_bbox_idx][latter_img_joints_idx]['x'][0]), \
                #                                                     int(latter_img_skeletons[latter_img_bbox_idx][latter_img_joints_idx]['y'][0])), 2, \
                #                                                    (0, 0, 255), 2)
                # # dump
                # cv2.imwrite(os.path.join('/home/vipuser/Downloads/TIP_AQ_pose_finetune/dump', \
                #                          latter_img_dir.split('/')[-2] + '__' + latter_img_dir.split('/')[-1].split('.')[0] + '__' + str(latter_img_bbox_idx) + '.png'), \
                #             latter_img)
                #             # latter_img_to_be_modified_vis)
                #
                # curr_frame_curr_box_txt = open(os.path.join('/home/vipuser/Downloads/TIP_AQ_pose_finetune/dump_imgname_boxid_boxcoord', \
                #                                             latter_img_dir.split('/')[-2] + '__' + latter_img_dir.split('/')[-1].split('.')[0] + '__' + str(latter_img_bbox_idx) + '.txt'), 'a')
                # curr_frame_curr_box_txt.write(str(latter_img_bboxes[latter_img_bbox_idx]['bbox'][0]) + ' ')
                # curr_frame_curr_box_txt.write(str(latter_img_bboxes[latter_img_bbox_idx]['bbox'][1]) + ' ')
                # curr_frame_curr_box_txt.write(str(latter_img_bboxes[latter_img_bbox_idx]['bbox'][2]) + ' ')
                # curr_frame_curr_box_txt.write(str(latter_img_bboxes[latter_img_bbox_idx]['bbox'][3]) + '\n')
                # curr_frame_curr_box_txt.close()

        # recover ori img to facilitate other humans
        latter_img_to_be_modified = copy.deepcopy(latter_img)

    return latter_img_joints, latter_img_skeletons

def restrict_coords(former_img_bboxes, former_img):
    for former_img_bbox_idx in range(len(former_img_bboxes)):
        former_img_bboxes[former_img_bbox_idx]['bbox'][0] = max([former_img_bboxes[former_img_bbox_idx]['bbox'][0], 0])
        former_img_bboxes[former_img_bbox_idx]['bbox'][1] = max([former_img_bboxes[former_img_bbox_idx]['bbox'][1], 0])
        former_img_bboxes[former_img_bbox_idx]['bbox'][2] = min([former_img_bboxes[former_img_bbox_idx]['bbox'][2], \
                                                                 former_img.shape[1] - 1 - former_img_bboxes[former_img_bbox_idx]['bbox'][0]])
        former_img_bboxes[former_img_bbox_idx]['bbox'][3] = min([former_img_bboxes[former_img_bbox_idx]['bbox'][3], \
                                                                 former_img.shape[0] - 1 - former_img_bboxes[former_img_bbox_idx]['bbox'][1]])
    return former_img_bboxes

def restrict_coords_skeleton(former_img_skeletons, former_img):
    for former_img_skeleton_idx in range(len(former_img_skeletons)):
        for joint_idx in range(len(former_img_skeletons[former_img_skeleton_idx])):
            former_img_skeletons[former_img_skeleton_idx][joint_idx]['x'][0] = max([former_img_skeletons[former_img_skeleton_idx][joint_idx]['x'][0], 0])
            former_img_skeletons[former_img_skeleton_idx][joint_idx]['x'][0] = min([former_img_skeletons[former_img_skeleton_idx][joint_idx]['x'][0], former_img.shape[1] - 1])
            former_img_skeletons[former_img_skeleton_idx][joint_idx]['y'][0] = max([former_img_skeletons[former_img_skeleton_idx][joint_idx]['y'][0], 0])
            former_img_skeletons[former_img_skeleton_idx][joint_idx]['y'][0] = min([former_img_skeletons[former_img_skeleton_idx][joint_idx]['y'][0], former_img.shape[0] - 1])
    return former_img_skeletons

def bbox_propagation(curr_box_corner_points, curr_forward_flow):
    propagated_box = []
    # lefttop, righttop, leftbottom, rightbottom, "-2" because interpolation requires "+1"
    left_coord, right_coord = max([curr_box_corner_points[0], 0]), min([curr_box_corner_points[0] + curr_box_corner_points[2], curr_forward_flow.shape[1] - 2])
    top_coord, bottom_coord = max([curr_box_corner_points[1], 0]), min([curr_box_corner_points[1] + curr_box_corner_points[3], curr_forward_flow.shape[0] - 2])
    curr_box_corner_points = [[left_coord, top_coord], [right_coord, top_coord], [left_coord, bottom_coord], [right_coord, bottom_coord]]
    for former_img_keypoint in curr_box_corner_points:
        former_img_keypoint_left, former_img_keypoint_right, former_img_keypoint_top, former_img_keypoint_bottom = \
            int(former_img_keypoint[0]), int(former_img_keypoint[0]) + 1, int(former_img_keypoint[1]), int(former_img_keypoint[1]) + 1
        former_img_keypoint_lefttop_motion = curr_forward_flow[former_img_keypoint_top, former_img_keypoint_left, :]
        former_img_keypoint_leftbottom_motion = curr_forward_flow[former_img_keypoint_bottom, former_img_keypoint_left, :]
        former_img_keypoint_righttop_motion = curr_forward_flow[former_img_keypoint_top, former_img_keypoint_right, :]
        former_img_keypoint_rightbottom_motion = curr_forward_flow[former_img_keypoint_bottom, former_img_keypoint_right, :]

        former_img_keypoint_topline_motion = (former_img_keypoint_right - former_img_keypoint[0]) / (former_img_keypoint_right - former_img_keypoint_left) * former_img_keypoint_lefttop_motion + \
                                             (former_img_keypoint[0] - former_img_keypoint_left) / (former_img_keypoint_right - former_img_keypoint_left) * former_img_keypoint_righttop_motion
        former_img_keypoint_bottomline_motion = (former_img_keypoint_right - former_img_keypoint[0]) / (former_img_keypoint_right - former_img_keypoint_left) * former_img_keypoint_leftbottom_motion + \
                                                (former_img_keypoint[0] - former_img_keypoint_left) / (former_img_keypoint_right - former_img_keypoint_left) * former_img_keypoint_rightbottom_motion
        former_img_keypoint_motion = (former_img_keypoint_bottom - former_img_keypoint[1]) / (former_img_keypoint_bottom - former_img_keypoint_top) * former_img_keypoint_topline_motion + \
                                     (former_img_keypoint[1] - former_img_keypoint_top) / (former_img_keypoint_bottom - former_img_keypoint_top) * former_img_keypoint_bottomline_motion

        propagated_box.append(former_img_keypoint + former_img_keypoint_motion)
    return propagated_box

def keypoint_propagation(curr_box_corner_points, curr_forward_flow):
    propagated_box = []
    # "-2" because interpolation requires "+1"
    curr_box_corner_points = [[min([x[0], curr_forward_flow.shape[1] - 2]), min([x[1], curr_forward_flow.shape[0] - 2])] for x in curr_box_corner_points]
    for former_img_keypoint in curr_box_corner_points:
        former_img_keypoint_left, former_img_keypoint_right, former_img_keypoint_top, former_img_keypoint_bottom = \
            int(former_img_keypoint[0]), int(former_img_keypoint[0]) + 1, int(former_img_keypoint[1]), int(former_img_keypoint[1]) + 1
        former_img_keypoint_lefttop_motion = curr_forward_flow[former_img_keypoint_top, former_img_keypoint_left, :]
        former_img_keypoint_leftbottom_motion = curr_forward_flow[former_img_keypoint_bottom, former_img_keypoint_left, :]
        former_img_keypoint_righttop_motion = curr_forward_flow[former_img_keypoint_top, former_img_keypoint_right, :]
        former_img_keypoint_rightbottom_motion = curr_forward_flow[former_img_keypoint_bottom, former_img_keypoint_right, :]

        former_img_keypoint_topline_motion = (former_img_keypoint_right - former_img_keypoint[0]) / (former_img_keypoint_right - former_img_keypoint_left) * former_img_keypoint_lefttop_motion + \
                                             (former_img_keypoint[0] - former_img_keypoint_left) / (former_img_keypoint_right - former_img_keypoint_left) * former_img_keypoint_righttop_motion
        former_img_keypoint_bottomline_motion = (former_img_keypoint_right - former_img_keypoint[0]) / (former_img_keypoint_right - former_img_keypoint_left) * former_img_keypoint_leftbottom_motion + \
                                                (former_img_keypoint[0] - former_img_keypoint_left) / (former_img_keypoint_right - former_img_keypoint_left) * former_img_keypoint_rightbottom_motion
        former_img_keypoint_motion = (former_img_keypoint_bottom - former_img_keypoint[1]) / (former_img_keypoint_bottom - former_img_keypoint_top) * former_img_keypoint_topline_motion + \
                                     (former_img_keypoint[1] - former_img_keypoint_top) / (former_img_keypoint_bottom - former_img_keypoint_top) * former_img_keypoint_bottomline_motion

        propagated_box.append(former_img_keypoint + former_img_keypoint_motion)
    return propagated_box

def select_from_twice_propagated_samples_v2(former_img_keypoints, curr_forward_flow):
    picked_samples_from_former_img_keypoints_twice_propagated = []
    for former_img_keypoint in former_img_keypoints:
        former_img_keypoint = former_img_keypoint[:2]
        former_img_keypoint_left, former_img_keypoint_right, former_img_keypoint_top, former_img_keypoint_bottom = \
            int(former_img_keypoint[0]), int(former_img_keypoint[0]) + 1, int(former_img_keypoint[1]), int(former_img_keypoint[1]) + 1
        former_img_keypoint_lefttop_motion = curr_forward_flow[:, former_img_keypoint_top, former_img_keypoint_left]
        former_img_keypoint_leftbottom_motion = curr_forward_flow[:, former_img_keypoint_bottom, former_img_keypoint_left]
        former_img_keypoint_righttop_motion = curr_forward_flow[:, former_img_keypoint_top, former_img_keypoint_right]
        former_img_keypoint_rightbottom_motion = curr_forward_flow[:, former_img_keypoint_bottom, former_img_keypoint_right]

        overall_weight = (1 / ((former_img_keypoint[0] - former_img_keypoint_left) ** 2 + (former_img_keypoint[1] - former_img_keypoint_top) ** 2)) + \
                         (1 / ((former_img_keypoint[0] - former_img_keypoint_left) ** 2 + (former_img_keypoint[1] - former_img_keypoint_bottom) ** 2)) + \
                         (1 / ((former_img_keypoint[0] - former_img_keypoint_right) ** 2 + (former_img_keypoint[1] - former_img_keypoint_top) ** 2)) + \
                         (1 / ((former_img_keypoint[0] - former_img_keypoint_right) ** 2 + (former_img_keypoint[1] - former_img_keypoint_bottom) ** 2))

        lefttop_weight = (1 / ((former_img_keypoint[0] - former_img_keypoint_left) ** 2 + (former_img_keypoint[1] - former_img_keypoint_top) ** 2)) / overall_weight
        leftbottom_weight = (1 / ((former_img_keypoint[0] - former_img_keypoint_left) ** 2 + (former_img_keypoint[1] - former_img_keypoint_bottom) ** 2)) / overall_weight
        righttop_weight = (1 / ((former_img_keypoint[0] - former_img_keypoint_right) ** 2 + (former_img_keypoint[1] - former_img_keypoint_top) ** 2)) / overall_weight
        rightbottom_weight = (1 / ((former_img_keypoint[0] - former_img_keypoint_right) ** 2 + (former_img_keypoint[1] - former_img_keypoint_bottom) ** 2)) / overall_weight

        former_img_keypoint_motion = former_img_keypoint_lefttop_motion * lefttop_weight + former_img_keypoint_leftbottom_motion * leftbottom_weight + \
                                     former_img_keypoint_righttop_motion * righttop_weight + former_img_keypoint_rightbottom_motion * rightbottom_weight
        picked_samples_from_former_img_keypoints_twice_propagated.append(former_img_keypoint + former_img_keypoint_motion)
    return picked_samples_from_former_img_keypoints_twice_propagated

def organize_joints_to_skeletons(former_img_joints_propagated, former_img_example_skeleton_list):
    former_img_skeleton_propagated_inside_fun = []
    for human_idx in range(len(former_img_joints_propagated)):
        former_img_skeleton_propagated_inside_fun_curr_human = []
        for former_img_joints_propagated_ele in former_img_joints_propagated[human_idx]:
            former_img_skeleton_propagated_inside_fun_curr_human.append(
                {
                    'id': [former_img_joints_propagated[human_idx].index(former_img_joints_propagated_ele)],
                    'x': [former_img_joints_propagated_ele[0]],
                    'y': [former_img_joints_propagated_ele[1]],
                    'score': [former_img_example_skeleton_list[human_idx][former_img_joints_propagated[human_idx].index(former_img_joints_propagated_ele)]['score'][0]]
                }
            )
        former_img_skeleton_propagated_inside_fun.append(former_img_skeleton_propagated_inside_fun_curr_human)
    return former_img_skeleton_propagated_inside_fun

def heatmap_inter_joint_nms_v1(heatmaps, width, height):
    heatmaps_result = np.zeros(heatmaps.shape)
    # channel 0-16, max in channel dimension
    row_col_opt_channel = np.argmax(heatmaps, axis=1)[0, :, :] # row and col dimensions

    for heatmaps_result_ch_idx in range(heatmaps_result.shape[1]):
        heatmaps_result[0, heatmaps_result_ch_idx, :, :][np.where(row_col_opt_channel == heatmaps_result_ch_idx)] += \
            heatmaps[0, heatmaps_result_ch_idx, :, :][np.where(row_col_opt_channel == heatmaps_result_ch_idx)]

    return heatmaps_result

def heatmap_inter_joint_nms(heatmaps, width, height):
    heatmaps_result = np.zeros(heatmaps.shape)
    heatmaps_result = heatmaps
    # channel 0-16, max in channel dimension
    # for heatmaps_result_ch_idx in range(heatmaps_result.shape[1]):
    #     heatmaps_result[0, heatmaps_result_ch_idx, :, :] = heatmaps_result[0, heatmaps_result_ch_idx, :, :] / np.max(heatmaps_result[0, heatmaps_result_ch_idx, :, :])
    return heatmaps_result

def resize_heatmap_aspect_ratio_correction(heatmap_sum, target_width, target_height, enlarge_heatmap, enlarge_heatmap_ratio, valid_gaussian_kernel_area):
    # if lying
    if target_width / target_height >= heatmap_sum.shape[1] / heatmap_sum.shape[0]:
        crop_heatmap_height = heatmap_sum.shape[1] / target_width * target_height
        heatmap_sum = heatmap_sum[int(heatmap_sum.shape[0] / 2 - crop_heatmap_height / 2): int(heatmap_sum.shape[0] / 2 + crop_heatmap_height / 2), :]
    else:
        crop_heatmap_width = heatmap_sum.shape[0] / target_height * target_width
        heatmap_sum = heatmap_sum[:, int(heatmap_sum.shape[1] / 2 - crop_heatmap_width / 2): int(heatmap_sum.shape[1] / 2 + crop_heatmap_width / 2)]

    # whether need to enlarge heatmap here
    if enlarge_heatmap:
        heatmap_sum = cv2.resize(heatmap_sum, (0, 0), fx=enlarge_heatmap_ratio, fy=enlarge_heatmap_ratio)

    heatmap_sum_copy = copy.deepcopy((heatmap_sum / np.max(heatmap_sum) * 255).astype('uint8'))
    heatmap_sum_copy[np.where(heatmap_sum_copy < 10)] = 0
    num_subheatmaps, labels, subheatmaps_sizes, subheatmaps_centers = cv2.connectedComponentsWithStats(heatmap_sum_copy, connectivity=8)

    heatmap_sum_result = np.zeros((target_height, target_width))

    for subheatmap_idx in range(1, num_subheatmaps): # the first one is background
        if subheatmaps_sizes[subheatmap_idx][-1] > valid_gaussian_kernel_area:
            src_center = subheatmaps_centers[subheatmap_idx]
            target_center = [src_center[0] / heatmap_sum.shape[1] * target_width, src_center[1] / heatmap_sum.shape[1] * target_width]
            heatmap_sum_result[(np.where(labels == subheatmap_idx)[0] - int(round(src_center[1])) + int(round(target_center[1])), \
                                np.where(labels == subheatmap_idx)[1] - int(round(src_center[0])) + int(round(target_center[0])))] = heatmap_sum[np.where(labels == subheatmap_idx)]

    # heatmap_sum = cv2.resize(heatmap_sum, (target_width, target_height))
    return heatmap_sum_result

def compute_inter_joint_distances(curr_frame_diff_IDs_skeletons_curr_identity):
    distance_matrix = np.zeros((curr_frame_diff_IDs_skeletons_curr_identity.shape[0], curr_frame_diff_IDs_skeletons_curr_identity.shape[0]))
    for row_idx in range(curr_frame_diff_IDs_skeletons_curr_identity.shape[0]):
        for col_idx in range(curr_frame_diff_IDs_skeletons_curr_identity.shape[0]):
            distance_matrix[row_idx, col_idx] = np.linalg.norm(curr_frame_diff_IDs_skeletons_curr_identity[row_idx, :2] - \
                                                               curr_frame_diff_IDs_skeletons_curr_identity[col_idx, :2])
    distance_matrix_max_value = np.max(distance_matrix)
    for row_idx in range(curr_frame_diff_IDs_skeletons_curr_identity.shape[0]):
        distance_matrix[row_idx, row_idx] = distance_matrix_max_value * 2
    return distance_matrix

def prepare_skeletons_former_latter_imgs(curr_frame_name, identity_idx, joints, former_img_example_skeleton1_hrnet, \
                                         former_img_example_skeleton2_hrnet, former_img_example_skeleton3_hrnet, \
                                         latter_img_example_skeleton1_hrnet, latter_img_example_skeleton2_hrnet, latter_img_example_skeleton3_hrnet):
    if curr_frame_name == '000008.png' and identity_idx == 0:
        for joint_idx in range(joints.shape[1]):
            former_img_example_skeleton1_hrnet.append({
                'id': [joint_idx], 'x': [joints[0][joint_idx][0]], 'y': [joints[0][joint_idx][1]],
                'score': [joints[0][joint_idx][2]]
            })
    if curr_frame_name == '000008.png' and identity_idx == 1:
        for joint_idx in range(joints.shape[1]):
            former_img_example_skeleton2_hrnet.append({
                'id': [joint_idx], 'x': [joints[0][joint_idx][0]], 'y': [joints[0][joint_idx][1]],
                'score': [joints[0][joint_idx][2]]
            })
    if curr_frame_name == '000008.png' and identity_idx == 2:
        for joint_idx in range(joints.shape[1]):
            former_img_example_skeleton3_hrnet.append({
                'id': [joint_idx], 'x': [joints[0][joint_idx][0]], 'y': [joints[0][joint_idx][1]],
                'score': [joints[0][joint_idx][2]]
            })
    if curr_frame_name == '000012.png' and identity_idx == 0:
        for joint_idx in range(joints.shape[1]):
            latter_img_example_skeleton1_hrnet.append({
                'id': [joint_idx], 'x': [joints[0][joint_idx][0]], 'y': [joints[0][joint_idx][1]],
                'score': [joints[0][joint_idx][2]]
            })
    if curr_frame_name == '000012.png' and identity_idx == 1:
        for joint_idx in range(joints.shape[1]):
            latter_img_example_skeleton2_hrnet.append({
                'id': [joint_idx], 'x': [joints[0][joint_idx][0]], 'y': [joints[0][joint_idx][1]],
                'score': [joints[0][joint_idx][2]]
            })
    if curr_frame_name == '000012.png' and identity_idx == 2:
        for joint_idx in range(joints.shape[1]):
            latter_img_example_skeleton3_hrnet.append({
                'id': [joint_idx], 'x': [joints[0][joint_idx][0]], 'y': [joints[0][joint_idx][1]],
                'score': [joints[0][joint_idx][2]]
            })
    return former_img_example_skeleton1_hrnet, former_img_example_skeleton2_hrnet, former_img_example_skeleton3_hrnet, \
           latter_img_example_skeleton1_hrnet, latter_img_example_skeleton2_hrnet, latter_img_example_skeleton3_hrnet

def convert_heatmaps_to_keypoints(heatmap_grayscale_dir, num_ids, keyword, valid_gaussian_kernel_area):
    img_example_skeleton_list = []
    for human_idx in range(num_ids):
        img_example_curr_img_skeleton = []
        for gray_heatmap_img_name in sorted([x for x in os.listdir(os.path.join(heatmap_grayscale_dir, 'ID' + str(human_idx))) if keyword in x], key=lambda x:int(x.split('Step8_two_frames_heatmaps_')[1].split('_ID')[0])):
            curr_gray_heatmap = cv2.imread(os.path.join(heatmap_grayscale_dir, 'ID' + str(human_idx), gray_heatmap_img_name))
            curr_gray_heatmap[np.where(curr_gray_heatmap < 10)] = 0
            num_subheatmaps, labels, subheatmaps_sizes, subheatmaps_centers = cv2.connectedComponentsWithStats(curr_gray_heatmap[:, :, 0], connectivity=8)

            for subheatmap_idx in range(1, num_subheatmaps):  # the first one is background
                if subheatmaps_sizes[subheatmap_idx][-1] > valid_gaussian_kernel_area:
                    gaussian_kernel_center = subheatmaps_centers[subheatmap_idx]
                    gaussian_radius = max([subheatmaps_sizes[subheatmap_idx][-2], subheatmaps_sizes[subheatmap_idx][-3]]) / 2
                    img_example_curr_img_skeleton.append({
                        'id': [sorted([x for x in os.listdir(os.path.join(heatmap_grayscale_dir, 'ID' + str(human_idx))) if keyword in x], key=lambda x:int(x.split('Step8_two_frames_heatmaps_')[1].split('_ID')[0])).index(gray_heatmap_img_name)],
                        'x': [gaussian_kernel_center[0]],
                        'y': [gaussian_kernel_center[1]],
                        'score': [gaussian_radius],
                        'heatmap_temperature': np.mean(curr_gray_heatmap[:, :, 0][np.where(labels == subheatmap_idx)])
                    })
        img_example_skeleton_list.append(img_example_curr_img_skeleton)
    return img_example_skeleton_list

def enlarge_bbox(former_img_example_bbox3, enlarge_ratio):
    former_box_left, former_box_top, former_box_width, former_box_height = former_img_example_bbox3['bbox'][0], former_img_example_bbox3['bbox'][1], \
                                                                           former_img_example_bbox3['bbox'][2], former_img_example_bbox3['bbox'][3]
    former_box_hcenter = former_box_left + former_box_width / 2
    former_box_vcenter = former_box_top + former_box_height / 2
    former_img_example_bbox3['bbox'] = [former_box_hcenter - former_box_width / 2 - former_box_width * enlarge_ratio / 2, \
                                        former_box_vcenter - former_box_height / 2 - former_box_height * enlarge_ratio / 2, \
                                        former_box_width * (1 + enlarge_ratio), former_box_height * (1 + enlarge_ratio)]
    return former_img_example_bbox3

def pose_estimation(curr_frame_curr_human_bbox, curr_img, isotropicGrayscaleImage, convert_coco_to_posetrack_format):
    center_coord = [curr_frame_curr_human_bbox[0] + curr_frame_curr_human_bbox[2] / 2, curr_frame_curr_human_bbox[1] + curr_frame_curr_human_bbox[3] / 2]  #
    size_single_person = [curr_frame_curr_human_bbox[2], curr_frame_curr_human_bbox[3]]
    joints_list = []
    width, height = curr_frame_curr_human_bbox[2], curr_frame_curr_human_bbox[3]
    scales_heatmaps_list, scales_joints_list, scales_add_heatmaps_list = [], [], []
    scales_skeleton = [0.6, 0.8, 1.0, 1.1, 1.2, 1.3, 1.4]
    for scale_skeleton in scales_skeleton:
        curr_id_curr_frame_left, curr_id_curr_frame_top, curr_id_curr_frame_right, curr_id_curr_frame_bottom = int(
            center_coord[0] - width / 2 * scale_skeleton), int(center_coord[1] - height / 2 * scale_skeleton), int(
            center_coord[0] + width / 2 * scale_skeleton), int(center_coord[1] + height / 2 * scale_skeleton)
        heatmaps, joints = model.predict(curr_img, curr_id_curr_frame_left, curr_id_curr_frame_right, curr_id_curr_frame_top, curr_id_curr_frame_bottom)
        # convert coco format to posetrack format
        if convert_coco_to_posetrack_format:
            joints, heatmaps, add_heatmaps = convert_from_17_to_15_joints_and_heatmaps(joints, heatmaps, isotropicGrayscaleImage, curr_img.shape[1], curr_img.shape[0])
        # convert coco format to posetrack format end
        scales_heatmaps_list.append(heatmaps)
        if convert_coco_to_posetrack_format:
            scales_add_heatmaps_list.append(add_heatmaps)
        scales_joints_list.append(joints)
    optimal_scale = np.argmax([np.sum(x[0, :, 2]) for x in scales_joints_list])
    curr_id_curr_frame_left, curr_id_curr_frame_top, curr_id_curr_frame_right, curr_id_curr_frame_bottom = int(
        center_coord[0] - width / 2 * scales_skeleton[optimal_scale]), int(center_coord[1] - height / 2 * scales_skeleton[optimal_scale]), int(center_coord[0] + width / 2 * scales_skeleton[optimal_scale]), int(center_coord[1] + height / 2 * scales_skeleton[optimal_scale])
    heatmaps, joints = scales_heatmaps_list[optimal_scale], scales_joints_list[optimal_scale]
    return joints, [curr_id_curr_frame_left, curr_id_curr_frame_top, curr_id_curr_frame_right - curr_id_curr_frame_left, curr_id_curr_frame_bottom - curr_id_curr_frame_top], heatmaps


















