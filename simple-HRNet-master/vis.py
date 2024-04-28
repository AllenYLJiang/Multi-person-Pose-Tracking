import cv2
import os
import json
import random
import numpy as np
import ot
import math
from SimpleHRNet import SimpleHRNet
from scipy.interpolate import griddata
import copy
import torch

model = SimpleHRNet(48, 17, "./weights/pose_hrnet_w48_384x288.pth", device=torch.device('cuda'))

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

manually_labelled_semantics = {
    'head top': 0,
    'nose': 1,
    'neck': 2,
    'left shoulder': 3,
    'right shoulder': 4,
    'left elbow': 5,
    'right elbow': 6,
    'left wrist': 7,
    'right wrist': 8,
    'left hip': 9,
    'right hip': 10,
    'left knee': 11,
    'right knee': 12,
    'left ankle': 13,
    'right ankle': 14
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

def skeleton_nms_format2(next_img_propagated_and_detected_bboxes_skeleton, next_img_propagated_and_detected_bboxes):
    all_people_skeleton_boundaries_list = []
    for single_person_skeleton in next_img_propagated_and_detected_bboxes_skeleton:
        # top, bottom, left, right
        all_people_skeleton_boundaries_list.append([min(single_person_skeleton[:, 0]), max(single_person_skeleton[:, 0]), \
                                                    min(single_person_skeleton[:, 1]), max(single_person_skeleton[:, 1])])
    iou_matrix = np.zeros((len(all_people_skeleton_boundaries_list), len(all_people_skeleton_boundaries_list)))
    curr_img_all_skeletons_valid_masks = np.ones((len(all_people_skeleton_boundaries_list)))
    # not symmetric because iou is head-body
    for iou_matrix_row in range(iou_matrix.shape[0]):
        for iou_matrix_col in range(iou_matrix.shape[1]):
            if iou_matrix_row != iou_matrix_col:
                iou_matrix[iou_matrix_row, iou_matrix_col] = compute_iou_single_box(all_people_skeleton_boundaries_list[iou_matrix_row], \
                    all_people_skeleton_boundaries_list[iou_matrix_col])
                if iou_matrix[iou_matrix_row, iou_matrix_col] >= iou_thresh:
                    if np.mean(next_img_propagated_and_detected_bboxes_skeleton[iou_matrix_row][:, 2]) > np.mean(next_img_propagated_and_detected_bboxes_skeleton[iou_matrix_col][:, 2]):
                        curr_img_all_skeletons_valid_masks[iou_matrix_col] = 0
                    else:
                        curr_img_all_skeletons_valid_masks[iou_matrix_row] = 0

    curr_video_bboxes_curr_frame_corrected = []
    for curr_video_bboxes_curr_frame_idx in range(len(next_img_propagated_and_detected_bboxes)):
        if curr_img_all_skeletons_valid_masks[curr_video_bboxes_curr_frame_idx] == 1:
            curr_video_bboxes_curr_frame_corrected.append(next_img_propagated_and_detected_bboxes[curr_video_bboxes_curr_frame_idx])

    curr_img_all_skeletons_annorect = []
    for curr_img_all_skeletons_valid_masks_idx in range(len(curr_img_all_skeletons_valid_masks)):
        if curr_img_all_skeletons_valid_masks[curr_img_all_skeletons_valid_masks_idx] == 1:
            curr_img_all_skeletons_annorect.append(next_img_propagated_and_detected_bboxes_skeleton[curr_img_all_skeletons_valid_masks_idx])
    next_img_propagated_and_detected_bboxes_skeleton = curr_img_all_skeletons_annorect

    return next_img_propagated_and_detected_bboxes_skeleton, curr_video_bboxes_curr_frame_corrected

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

def vis_curr_img_joints_keypoints(curr_img_all_skeletons, curr_video_bboxes, dst_vis_dir, video_name, curr_img_name, curr_img_keypoints = []):
    for curr_person_skeleton_idx in range(len(curr_img_all_skeletons['annorect'])):
        curr_img = cv2.imread(os.path.join(src_video_dir, video_name.split('_backward')[0].split('_forward')[0], curr_img_name.split('/')[-1]))
        curr_person_skeleton = curr_img_all_skeletons['annorect'][curr_person_skeleton_idx]
        curr_person_bbox = [x for x in curr_video_bboxes if (curr_img_name in x['image_name'])][curr_person_skeleton_idx]['bbox']
        joint_list_coord = curr_person_skeleton['annopoints'][0]['point']
        left, top, width, height = curr_person_bbox[0], curr_person_bbox[1], curr_person_bbox[2], curr_person_bbox[3]
        cv2.rectangle(curr_img, pt1=(int(left), int(top)), pt2=(int(left + width), int(top + height)), color=(255, 0, 0), thickness=1)
        for curr_joint in joint_list_coord:
            radius, thickness, color = 2, 2, keypoints_colors[joint_list_coord.index(curr_joint)]  # (255, 0, 0)
            curr_img = cv2.circle(curr_img, (int(curr_joint['x'][0]), int(curr_joint['y'][0])), radius, color, thickness)
        if not os.path.exists(os.path.join(dst_vis_dir, video_name)):
            os.mkdir(os.path.join(dst_vis_dir, video_name))
        cv2.imwrite(os.path.join(dst_vis_dir, video_name, curr_img_name.split('/')[-1].split('.')[0] + '_' + str(curr_person_skeleton_idx) + '.jpg'), curr_img)
    curr_img = cv2.imread(os.path.join(src_video_dir, video_name.split('_backward')[0].split('_forward')[0], curr_img_name.split('/')[-1]))
    if len(curr_img_keypoints) != 0:
        for curr_img_keypoint in curr_img_keypoints:
            curr_img = cv2.circle(curr_img, (int(curr_img_keypoint[0]), int(curr_img_keypoint[1])), 1, (0, 0, 255), 1)
        cv2.imwrite(os.path.join(dst_vis_dir, video_name, curr_img_name.split('/')[-1].split('.')[0] + '_samples' + '.jpg'), curr_img)
    else:
        cv2.imwrite(os.path.join(dst_vis_dir, video_name, curr_img_name.split('/')[-1].split('.')[0] + '_samples' + '.jpg'), curr_img)

def vis_curr_img_joints_keypoints_format2(curr_img_expanded_bboxes_propagated_to_latter_frame_skeletons, curr_img_expanded_bboxes_propagated_to_latter_frame, dst_vis_dir, video_name, next_img_name, curr_img_expanded_samples_propagated_to_next_img=[]):
    for curr_person_skeleton_idx in range(len(curr_img_expanded_bboxes_propagated_to_latter_frame_skeletons)):
        curr_img = cv2.imread(os.path.join(src_video_dir, video_name.split('_backward')[0].split('_forward')[0], next_img_name.split('/')[-1]))
        curr_person_skeleton = curr_img_expanded_bboxes_propagated_to_latter_frame_skeletons[curr_person_skeleton_idx]
        curr_person_bbox = curr_img_expanded_bboxes_propagated_to_latter_frame[curr_person_skeleton_idx]
        joint_list_coord = curr_person_skeleton
        left, top, width, height = curr_person_bbox[0], curr_person_bbox[1], curr_person_bbox[2], curr_person_bbox[3]
        cv2.rectangle(curr_img, pt1=(int(left), int(top)), pt2=(int(left + width), int(top + height)), color=(255, 0, 0), thickness=1)
        for curr_joint_idx in range(joint_list_coord.shape[0]):
            curr_joint = joint_list_coord[curr_joint_idx]
            radius, thickness, color = 2, 2, keypoints_colors[curr_joint_idx]  # (255, 0, 0)
            curr_img = cv2.circle(curr_img, (int(curr_joint[1]), int(curr_joint[0])), radius, color, thickness)
        if not os.path.exists(os.path.join(dst_vis_dir, video_name)):
            os.mkdir(os.path.join(dst_vis_dir, video_name))
        cv2.imwrite(os.path.join(dst_vis_dir, video_name, next_img_name.split('/')[-1].split('.')[0] + '_' + str(curr_person_skeleton_idx) + '.jpg'), curr_img)
    curr_img = cv2.imread(os.path.join(src_video_dir, video_name.split('_backward')[0].split('_forward')[0], next_img_name.split('/')[-1]))
    if len(curr_img_expanded_samples_propagated_to_next_img) != 0:
        for curr_img_keypoint in curr_img_expanded_samples_propagated_to_next_img:
            curr_img = cv2.circle(curr_img, (int(curr_img_keypoint[0]), int(curr_img_keypoint[1])), 1, (0, 0, 255), 1)
        cv2.imwrite(os.path.join(dst_vis_dir, video_name, next_img_name.split('/')[-1].split('.')[0] + '_samples' + '.jpg'), curr_img)
    else:
        cv2.imwrite(os.path.join(dst_vis_dir, video_name, next_img_name.split('/')[-1].split('.')[0] + '_samples' + '.jpg'), curr_img)

def locate_joints_keypoints_allhumans_former_img(curr_img_all_skeletons, src_video_dir, video_name, curr_img_name, curr_video_bboxes, Num_samples_former_img, Num_samples_latter_img, dst_vis_dir, curr_img_joints, curr_img_keypoints):
    for curr_person_skeleton_idx in range(len(curr_img_all_skeletons['annorect'])):
        curr_img = cv2.imread(os.path.join(src_video_dir, video_name.split('_backward')[0].split('_forward')[0], curr_img_name.split('/')[-1]))
        curr_person_skeleton = curr_img_all_skeletons['annorect'][curr_person_skeleton_idx]
        curr_person_bbox = [x for x in curr_video_bboxes if (curr_img_name in x['image_name'])][curr_person_skeleton_idx]['bbox']
        joint_list_coord = curr_person_skeleton['annopoints'][0]['point']
        left, top, width, height = curr_person_bbox[0], curr_person_bbox[1], curr_person_bbox[2], curr_person_bbox[3]

        # traverse joints
        # determine number of keypoints around each joint: across the frame, proportional to confidence,
        # at least one keypoint around each joint
        for curr_joint in joint_list_coord:
            curr_img_joints.append([curr_joint['x'][0], curr_joint['y'][0], curr_joint['score'][0]])

    # after traversing all joints of all humans, organize curr_img_joints to curr_img_keypoints
    for curr_img_joints_ele in curr_img_joints:
        num_samples_around_curr_joint = math.ceil(Num_samples_former_img / sum([x[2] for x in curr_img_joints]) * curr_img_joints_ele[2])
        nearest_neighbor_distance = min([np.linalg.norm(np.array(y[:2]) - np.array(curr_img_joints_ele[:2])) for y in [x for x in curr_img_joints if (x != curr_img_joints_ele)]])
        curr_img_keypoints += sampling_around_joint_return_samples([curr_img_joints_ele[0], curr_img_joints_ele[1]],
                                                                   num_samples_around_curr_joint,
                                                                   nearest_neighbor_distance / 2)
    return curr_img_joints, curr_img_keypoints

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

def merge_boxes_with_same_head(curr_img_expanded_bboxes_propagated_to_latter_frame, curr_img_expanded_bboxes_propagated_to_latter_frame_skeletons, same_face_thresh, next_img):
    same_head_pair = []
    head_iou_matrix = np.zeros((len(curr_img_expanded_bboxes_propagated_to_latter_frame_skeletons), len(curr_img_expanded_bboxes_propagated_to_latter_frame_skeletons)))
    for row_idx in range(len(curr_img_expanded_bboxes_propagated_to_latter_frame_skeletons)):
        for col_idx in range(len(curr_img_expanded_bboxes_propagated_to_latter_frame_skeletons)):
            if row_idx != col_idx:
                row_bbox_left = np.min(curr_img_expanded_bboxes_propagated_to_latter_frame_skeletons[row_idx][:5, 1])
                row_bbox_right = np.max(curr_img_expanded_bboxes_propagated_to_latter_frame_skeletons[row_idx][:5, 1])
                row_bbox_top = np.min(curr_img_expanded_bboxes_propagated_to_latter_frame_skeletons[row_idx][:5, 0])
                row_bbox_bottom = np.max(curr_img_expanded_bboxes_propagated_to_latter_frame_skeletons[row_idx][:5, 0])
                col_bbox_left = np.min(curr_img_expanded_bboxes_propagated_to_latter_frame_skeletons[col_idx][:5, 1])
                col_bbox_right = np.max(curr_img_expanded_bboxes_propagated_to_latter_frame_skeletons[col_idx][:5, 1])
                col_bbox_top = np.min(curr_img_expanded_bboxes_propagated_to_latter_frame_skeletons[col_idx][:5, 0])
                col_bbox_bottom = np.max(curr_img_expanded_bboxes_propagated_to_latter_frame_skeletons[col_idx][:5, 0])
                head_iou_matrix[row_idx, col_idx] = compute_iou_single_box_normal([row_bbox_top, row_bbox_bottom, row_bbox_left, row_bbox_right], \
                                                                                  [col_bbox_top, col_bbox_bottom, col_bbox_left, col_bbox_right])
                if head_iou_matrix[row_idx, col_idx] > same_face_thresh and ([row_idx, col_idx] not in same_head_pair) and ([col_idx, row_idx] not in same_head_pair):
                    same_head_pair.append([row_idx, col_idx])
    valid_box_indicator, merged_box_indicator = np.ones((len(curr_img_expanded_bboxes_propagated_to_latter_frame))), np.zeros((len(curr_img_expanded_bboxes_propagated_to_latter_frame)))
    for same_head_pair_ele in same_head_pair:# left, top, width, height
        curr_img_expanded_bboxes_propagated_to_latter_frame[min(same_head_pair_ele)] = merge_box(curr_img_expanded_bboxes_propagated_to_latter_frame[min(same_head_pair_ele)], \
                                                                                                 curr_img_expanded_bboxes_propagated_to_latter_frame[max(same_head_pair_ele)])
        valid_box_indicator[max(same_head_pair_ele)] = 0
        merged_box_indicator[min(same_head_pair_ele)] = 1
    for box_idx in range(len(merged_box_indicator)):
        if merged_box_indicator[box_idx] == 1:
            curr_img_expanded_bboxes_propagated_to_latter_frame_skeletons[box_idx] = \
                model.predict(next_img, curr_img_expanded_bboxes_propagated_to_latter_frame[box_idx][0], \
                                        curr_img_expanded_bboxes_propagated_to_latter_frame[box_idx][0] + curr_img_expanded_bboxes_propagated_to_latter_frame[box_idx][2], \
                                        curr_img_expanded_bboxes_propagated_to_latter_frame[box_idx][1], \
                                        curr_img_expanded_bboxes_propagated_to_latter_frame[box_idx][1] + curr_img_expanded_bboxes_propagated_to_latter_frame[box_idx][3])[0]
    merged_boxes, merged_skeletons = [], []
    for box_idx in range(len(valid_box_indicator)):
        if valid_box_indicator[box_idx] == 1:
            merged_boxes.append(curr_img_expanded_bboxes_propagated_to_latter_frame[box_idx])
            merged_skeletons.append(curr_img_expanded_bboxes_propagated_to_latter_frame_skeletons[box_idx])

    return merged_boxes, merged_skeletons

posewarper_result_dir = 'D:\\usr\\local\\TIP_26494\\posewarper\\per_frame_prediction'
bbox_all_videos = 'D:\\usr\\local\\TIP_26494\\posewarper\\test_boxes.json'
src_video_dir = 'D:\\usr\\local\\TIP_26494\\'
dst_vis_dir = 'G:\\usr\\local\\TIP_26494\\vis'
box_thresh = 0.2
Num_samples_former_img, Num_samples_latter_img = 1024, 2048
qualified_joint_conf_thresh = 0.1
forward_flow_dir = 'D:\\usr\\local\\TIP_26494\\GMFlowNet-master\\GMFlowNet-master\\results_forward'
backward_flow_dir = 'D:\\usr\\local\\TIP_26494\\GMFlowNet-master\\GMFlowNet-master\\results_backward'
iou_thresh = 0.7
num_joints_posetrack_format = 15
num_joints_hrnet_format = 17
same_face_thresh = 0.5

# determine maximum motion vector based on optical flow, also obtain statistics about <= here
max_hori_flow_vector, max_vert_flow_vector = 0.0, 0.0
for json_name in os.listdir(posewarper_result_dir):
    video_name = json_name.split('.')[0]
    curr_video_bboxes = [x for x in json.load(open(bbox_all_videos, 'r')) if (video_name in x['image_name'])]
    curr_video_bboxes = [x for x in curr_video_bboxes if x['score'] >= box_thresh]
    curr_video_posewarper_skeletons = json.load(open(os.path.join(posewarper_result_dir, json_name), 'r'))['annolist']
    for curr_img_all_skeletons_idx in range(len(curr_video_posewarper_skeletons) - 1):
        curr_img_all_skeletons = curr_video_posewarper_skeletons[curr_img_all_skeletons_idx]
        next_img_all_skeletons = curr_video_posewarper_skeletons[curr_img_all_skeletons_idx + 1]
        curr_img_name = curr_img_all_skeletons['image']['name']
        next_img_name = next_img_all_skeletons['image']['name']
        curr_forward_flow = np.load(os.path.join(forward_flow_dir, video_name, next_img_name.split('/')[-1].split('.')[0] + '.npy'))[0]
        curr_backward_flow = np.load(os.path.join(backward_flow_dir, video_name, curr_img_name.split('/')[-1].split('.')[0] + '.npy'))[0]
        max_hori_flow_vector = max([max_hori_flow_vector, np.max(abs(curr_forward_flow[0, :, :])), np.max(abs(curr_backward_flow[0, :, :]))])
        max_vert_flow_vector = max([max_vert_flow_vector, np.max(abs(curr_forward_flow[1, :, :])), np.max(abs(curr_backward_flow[1, :, :]))])
print('maximal optical flow vector obtained')

for json_name in os.listdir(posewarper_result_dir):
    video_name = json_name.split('.')[0]
    curr_video_bboxes = [x for x in json.load(open(bbox_all_videos, 'r')) if (video_name in x['image_name'])]
    curr_video_bboxes = [x for x in curr_video_bboxes if x['score'] >= box_thresh]
    curr_video_posewarper_skeletons = json.load(open(os.path.join(posewarper_result_dir, json_name), 'r'))['annolist']
    assert(len(curr_video_bboxes) == sum([len(x['annorect']) for x in curr_video_posewarper_skeletons]))
    # traverse frames
    for curr_img_all_skeletons_idx in range(len(curr_video_posewarper_skeletons) - 1):
        curr_img_all_skeletons = curr_video_posewarper_skeletons[curr_img_all_skeletons_idx]
        # NMS
        curr_img_name = curr_img_all_skeletons['image']['name']

        vis_curr_img_joints_keypoints(curr_img_all_skeletons, curr_video_bboxes, dst_vis_dir + '_step1', video_name, curr_img_name)

        curr_img_all_skeletons, curr_video_bboxes = skeleton_nms(curr_img_all_skeletons, curr_video_bboxes, iou_thresh, curr_img_name)

        vis_curr_img_joints_keypoints(curr_img_all_skeletons, curr_video_bboxes, dst_vis_dir + '_step2', video_name, curr_img_name)

        # traverse humans, sample around each joint, the radius around each joint is different with minimum distance between
        # two keypoints being 2. Normal radius around a join is half the distance between it and the nearest neighbor
        # Collect keypoints around current human, the keyppints on all humans compose keypoints in current image
        # associations between keypoints determine the associations between humans

        # flow estimation is not specific to each human. Instead, all keypoints' motion vectors in current image are estimated
        # then the motion vectors on each human are propagated to the next frame, determining the region that corresponds to
        # the human in the former frame

        # the samples around all joints of all humans in curr frame, each element includes: (hori coord in former frame,
        # vert coord in former frame)
        curr_img_joints, curr_img_keypoints = [], []
        curr_img_joints, curr_img_keypoints = locate_joints_keypoints_allhumans_former_img(curr_img_all_skeletons, src_video_dir, video_name, curr_img_name,
                                                                                           curr_video_bboxes, Num_samples_former_img, Num_samples_latter_img,
                                                                                           dst_vis_dir, curr_img_joints, curr_img_keypoints)
        print('Curr img all joints and all keypoints obtained !')

        vis_curr_img_joints_keypoints(curr_img_all_skeletons, curr_video_bboxes, dst_vis_dir + '_step3', video_name, curr_img_name, curr_img_keypoints)

        next_img_all_skeletons = curr_video_posewarper_skeletons[curr_img_all_skeletons_idx + 1]
        next_img_name = next_img_all_skeletons['image']['name']

        vis_curr_img_joints_keypoints(next_img_all_skeletons, curr_video_bboxes, dst_vis_dir + '_step4', video_name, next_img_name)

        next_img_all_skeletons, curr_video_bboxes = skeleton_nms(next_img_all_skeletons, curr_video_bboxes, iou_thresh, next_img_name)

        vis_curr_img_joints_keypoints(next_img_all_skeletons, curr_video_bboxes, dst_vis_dir + '_step5', video_name, next_img_name)

        # also fetch the results in the next frame, the strategy diverge from the former frame, samples more around the bboxes with
        # unconfident joints
        # 1. enlarge bbox partially based on unconfident joints or significantly varying bboxes 2. hrnet 3. sample
        # if all joints have confidence over qualified_joint_conf_thresh, curr bbox does not need to be enlarged

        # In sparse samples, no divergence between forward and backward flow
        # 1. Perform the same sampling in the latter frame with the same number of points in the former frame. what we've conducted is sparse flow estimation instead
        #    of dense ones so the input data is only sparse samples
        # 2. Associate humans based on flow estimations
        # 3. depend on cases:
        #    a. one joint in former frame has low confidence, if the corresponding human in next frame has a high-confidence joint with same semantic, propagate back.
        #    b. if the corresponding human in next frame also has low confidence joint with same semantic, then enlarge region to different extents based on confidence,
        #       proportion to N times half head size, then adopt 3 different scales. For each joint find one scale which gives it the highest confidence
        #       if the re-conducted joint is more confident than the former, then fix it. Note that in comparison, firstly normalize them to be with sum 1
        #    c. one joint in former frame has high confidence, if the corresponding human in next frame has a low-confidence joint with same semantic, propagate forward
        #    d. one joint in former frame has high confidence, if the corresponding human in next frame has high confidence
        #    e. the human in former frame cannot find its counterpart in latter frame
        # 4. solutions: (refine already included in the following steps)
        #    a. continue with Step 1, add samples in the latter frame corresponding to the locations in the former frame
        #    b. the spatial distribution of samples in the former frame does not apply any more. Continue with step 1 and re-apply hrnet to locate (not all) joints in
        #       the latter frame, distribute added samples around the renewed joints. Re-design spatial distribution of samples in the former frame as a downsampled
        #       version of the latter frame. Finally propagate back
        #  c & d & e. sample twice samples evenly from the latter frame, refine

        # For each box in the former frame, enlarge and propagate to the next frame
        curr_img_all_joints_confidence_average = np.mean([x[2] for x in curr_img_joints])
        # traverse boxes
        curr_img = cv2.imread(os.path.join(src_video_dir, video_name.split('_backward')[0].split('_forward')[0], curr_img_name.split('/')[-1]))
        next_img = cv2.imread(os.path.join(src_video_dir, video_name.split('_backward')[0].split('_forward')[0], next_img_name.split('/')[-1]))
        curr_img_expanded_bboxes_propagated_to_latter_frame_skeletons = []
        curr_img_expanded_bboxes_propagated_to_latter_frame = []
        curr_img_expanded_bboxes_without_skeletons_propagated_to_latter_frame = []
        curr_img_expanded_samples_propagated_to_next_img = []
        for curr_person_skeleton_idx in range(len(curr_img_all_skeletons['annorect'])):
            curr_person_expanded_samples = []
            curr_person_skeleton = curr_img_all_skeletons['annorect'][curr_person_skeleton_idx]
            curr_person_bbox = [x for x in curr_video_bboxes if (curr_img_name in x['image_name'])][curr_person_skeleton_idx]['bbox']
            joint_list_coord = curr_person_skeleton['annopoints'][0]['point']
            left, top, width, height = curr_person_bbox[0], curr_person_bbox[1], curr_person_bbox[2], curr_person_bbox[3]
            curr_img_expanded_bboxes_without_skeletons_propagated_to_latter_frame.append([max([left - max([max_hori_flow_vector, max_vert_flow_vector]), 0]), \
                                                                                          max([top - max([max_hori_flow_vector, max_vert_flow_vector]), 0]), \
                                                                                          min([left + width + max([max_hori_flow_vector, max_vert_flow_vector]), next_img.shape[1]]) - max([left - max([max_hori_flow_vector, max_vert_flow_vector]), 0]), \
                                                                                          min([top + height + max([max_hori_flow_vector, max_vert_flow_vector]), next_img.shape[0]]) - max([top - max([max_hori_flow_vector, max_vert_flow_vector]), 0])])
            # first determine range, then determine #samples around each joint
            # range: traverse joints, a joint with confidence >= mean value, use range=maximum flow vector, lower than mean values, enlarge with upper limit
            # #samples: first compute the number of samples in the bbox in former frame, multiply by 2, then divide it to joints according to circle areas
            curr_person_joints_range_list, curr_person_all_samples_latter_frame = [], Num_samples_latter_img / sum([x[2] for x in curr_img_joints]) * sum([x['score'][0] for x in joint_list_coord])
            for joint_info in joint_list_coord:
                curr_joint_range = 0
                if joint_info['score'][0] >= curr_img_all_joints_confidence_average:
                    curr_joint_range = max([max_hori_flow_vector, max_vert_flow_vector])
                else:
                    curr_joint_range = (curr_img_all_joints_confidence_average / max([joint_info['score'][0], 1e-5])) * max([max_hori_flow_vector, max_vert_flow_vector])
                    # 1/4 * body size = upper limit
                    curr_joint_range = min([curr_joint_range, max([width, height]) / 4])
                curr_person_joints_range_list.append(curr_joint_range)
            for joint_info in joint_list_coord:
                curr_joint_num_samples = curr_person_all_samples_latter_frame / (np.linalg.norm(curr_person_joints_range_list) ** 2) * (curr_person_joints_range_list[joint_list_coord.index(joint_info)] ** 2)
                curr_person_expanded_samples += sampling_around_joint_return_samples([joint_info['x'][0], joint_info['y'][0]], int(curr_joint_num_samples) + 1, curr_person_joints_range_list[joint_list_coord.index(joint_info)])
            curr_img_expanded_samples_propagated_to_next_img += curr_person_expanded_samples
            left_expanded, top_expanded, right_expanded, bottom_expanded = min([x[0] for x in curr_person_expanded_samples]), \
                                                                           min([x[1] for x in curr_person_expanded_samples]), \
                                                                           max([x[0] for x in curr_person_expanded_samples]), \
                                                                           max([x[1] for x in curr_person_expanded_samples]),
            # perform pose estimation on the expanded and propagated bboxes, note that conducted on next frame
            curr_expanded_box_joints = model.predict(next_img, left_expanded, right_expanded, top_expanded, bottom_expanded)
            curr_img_expanded_bboxes_propagated_to_latter_frame_skeletons.append(curr_expanded_box_joints[0])
            curr_img_expanded_bboxes_propagated_to_latter_frame.append([left_expanded, top_expanded, right_expanded-left_expanded, bottom_expanded-top_expanded])
        curr_img_expanded_bboxes_propagated_to_latter_frame, curr_img_expanded_bboxes_propagated_to_latter_frame_skeletons = merge_boxes_with_same_head(curr_img_expanded_bboxes_propagated_to_latter_frame, curr_img_expanded_bboxes_propagated_to_latter_frame_skeletons, same_face_thresh, next_img)
        vis_curr_img_joints_keypoints_format2(curr_img_expanded_bboxes_propagated_to_latter_frame_skeletons, curr_img_expanded_bboxes_propagated_to_latter_frame, dst_vis_dir + '_step6', video_name, next_img_name, curr_img_expanded_samples_propagated_to_next_img)

        # run hrnet also on next frame because the range of values of skeleton confidences are not the same
        next_img_bboxes = []
        next_img_bboxes_skeletons = []
        for next_img_person_bbox_idx in range(len([x for x in curr_video_bboxes if (next_img_name in x['image_name'])])):
            next_person_bbox = [x for x in curr_video_bboxes if (next_img_name in x['image_name'])][next_img_person_bbox_idx]['bbox']
            left, top, width, height = next_person_bbox[0], next_person_bbox[1], next_person_bbox[2], next_person_bbox[3]
            curr_box_in_next_img_joints = model.predict(next_img, left, left+width, top, top+height)
            next_img_bboxes.append([left, top, width, height])
            next_img_bboxes_skeletons.append(curr_box_in_next_img_joints[0])
        next_img_bboxes, next_img_bboxes_skeletons = merge_boxes_with_same_head(next_img_bboxes, next_img_bboxes_skeletons, same_face_thresh, next_img)
        # propagated from former boxes instead of skeletons, evenly enlarged
        next_img_evenly_enlarged_bboxes = []
        next_img_evenly_enlarged_bboxes_skeletons = []
        for curr_img_expanded_bboxes_without_skeletons_propagated_to_latter_frame_idx in range(len(curr_img_expanded_bboxes_without_skeletons_propagated_to_latter_frame)):
            left, top, width, height = curr_img_expanded_bboxes_without_skeletons_propagated_to_latter_frame[curr_img_expanded_bboxes_without_skeletons_propagated_to_latter_frame_idx][0], \
                                       curr_img_expanded_bboxes_without_skeletons_propagated_to_latter_frame[curr_img_expanded_bboxes_without_skeletons_propagated_to_latter_frame_idx][1], \
                                       curr_img_expanded_bboxes_without_skeletons_propagated_to_latter_frame[curr_img_expanded_bboxes_without_skeletons_propagated_to_latter_frame_idx][2], \
                                       curr_img_expanded_bboxes_without_skeletons_propagated_to_latter_frame[curr_img_expanded_bboxes_without_skeletons_propagated_to_latter_frame_idx][3]
            curr_box_in_next_img_joints = model.predict(next_img, left, left + width, top, top + height)
            next_img_evenly_enlarged_bboxes.append([left, top, width, height])
            next_img_evenly_enlarged_bboxes_skeletons.append(curr_box_in_next_img_joints[0])
        next_img_evenly_enlarged_bboxes, next_img_evenly_enlarged_bboxes_skeletons = merge_boxes_with_same_head(next_img_evenly_enlarged_bboxes, next_img_evenly_enlarged_bboxes_skeletons, same_face_thresh, next_img)

        vis_curr_img_joints_keypoints_format2(next_img_bboxes_skeletons, next_img_bboxes, dst_vis_dir + '_step7', video_name, next_img_name)
        vis_curr_img_joints_keypoints_format2(next_img_evenly_enlarged_bboxes_skeletons, next_img_evenly_enlarged_bboxes, dst_vis_dir + '_step8', video_name, next_img_name)

        next_img_propagated_and_detected_bboxes = curr_img_expanded_bboxes_propagated_to_latter_frame + next_img_bboxes
        next_img_propagated_and_detected_bboxes_skeleton = curr_img_expanded_bboxes_propagated_to_latter_frame_skeletons + next_img_bboxes_skeletons

        # merge skeletons with skeleton nms
        next_img_propagated_and_detected_bboxes_skeleton, next_img_propagated_and_detected_bboxes = skeleton_nms_format2(next_img_propagated_and_detected_bboxes_skeleton, next_img_propagated_and_detected_bboxes)
        print('Next img all joints and all keypoints obtained with the same number of samples!')

        # * NB: if re-localized joints are outside expanded boxes in latter frame,  After skeleton nms, re-conduct sampling on next_img_propagated_and_detected_bboxes_skeleton, update curr_img_expanded_samples_propagated_to_next_img
        # Do not do this: also update curr_img_keypoints = (curr_img_keypoints + curr_img_expanded_samples_propagated_to_next_img) / 2
        # * flow estimation between curr_img_keypoints and curr_img_expanded_samples_propagated_to_next_img
        # * human-level matching based on sparse flow
        # * do not consider the boxes which exist in latter frame but not exist in former frame
        # this is exactly dynamic bbox which is our advantage ! but how to merge with existing bboxes? skeleton nms
        # match curr_img_keypoints with next_img_keypoints using both forward flow and backward flow

        # next_img_all_skeletons, curr_video_bboxes = reformat_skeletons_boxes(next_img_propagated_and_detected_bboxes_skeleton, next_img_propagated_and_detected_bboxes)
        #
        # next_img_joints, next_img_keypoints = [], []
        # next_img_joints, next_img_keypoints = locate_joints_keypoints_allhumans_former_img(next_img_all_skeletons, \
        #                                                                                    src_video_dir, \
        #                                                                                    video_name + '_backward', \
        #                                                                                    next_img_name, \
        #                                                                                    curr_video_bboxes, \
        #                                                                                    Num_samples_former_img, \
        #                                                                                    Num_samples_latter_img, \
        #                                                                                    dst_vis_dir, \
        #                                                                                    next_img_joints, \
        #                                                                                    next_img_keypoints)

        curr_forward_flow = np.load(os.path.join(forward_flow_dir, video_name, next_img_name.split('/')[-1].split('.')[0] + '.npy'))[0]
        curr_img_keypoints_propagated = []
        for curr_img_keypoint in curr_img_keypoints:
            curr_img_keypoint_left, curr_img_keypoint_right, curr_img_keypoint_top, curr_img_keypoint_bottom = \
                int(curr_img_keypoint[0]), int(curr_img_keypoint[0]) + 1, int(curr_img_keypoint[1]), int(curr_img_keypoint[1]) + 1
            curr_img_keypoint_lefttop_motion = curr_forward_flow[:, curr_img_keypoint_top, curr_img_keypoint_left]
            curr_img_keypoint_leftbottom_motion = curr_forward_flow[:, curr_img_keypoint_bottom, curr_img_keypoint_left]
            curr_img_keypoint_righttop_motion = curr_forward_flow[:, curr_img_keypoint_top, curr_img_keypoint_right]
            curr_img_keypoint_rightbottom_motion = curr_forward_flow[:, curr_img_keypoint_bottom, curr_img_keypoint_right]

            overall_weight = (1 / ((curr_img_keypoint[0] - curr_img_keypoint_left) ** 2 + (curr_img_keypoint[1] - curr_img_keypoint_top) ** 2)) + \
                             (1 / ((curr_img_keypoint[0] - curr_img_keypoint_left) ** 2 + (curr_img_keypoint[1] - curr_img_keypoint_bottom) ** 2)) + \
                             (1 / ((curr_img_keypoint[0] - curr_img_keypoint_right) ** 2 + (curr_img_keypoint[1] - curr_img_keypoint_top) ** 2)) + \
                             (1 / ((curr_img_keypoint[0] - curr_img_keypoint_right) ** 2 + (curr_img_keypoint[1] - curr_img_keypoint_bottom) ** 2))

            lefttop_weight = (1 / ((curr_img_keypoint[0] - curr_img_keypoint_left) ** 2 + (curr_img_keypoint[1] - curr_img_keypoint_top) ** 2)) / overall_weight
            leftbottom_weight = (1 / ((curr_img_keypoint[0] - curr_img_keypoint_left) ** 2 + (curr_img_keypoint[1] - curr_img_keypoint_bottom) ** 2)) / overall_weight
            righttop_weight = (1 / ((curr_img_keypoint[0] - curr_img_keypoint_right) ** 2 + (curr_img_keypoint[1] - curr_img_keypoint_top) ** 2)) / overall_weight
            rightbottom_weight = (1 / ((curr_img_keypoint[0] - curr_img_keypoint_right) ** 2 + (curr_img_keypoint[1] - curr_img_keypoint_bottom) ** 2)) / overall_weight

            curr_img_keypoint_motion = curr_img_keypoint_lefttop_motion * lefttop_weight + \
                                       curr_img_keypoint_leftbottom_motion * leftbottom_weight + \
                                       curr_img_keypoint_righttop_motion * righttop_weight + \
                                       curr_img_keypoint_rightbottom_motion * rightbottom_weight

            curr_img_keypoints_propagated.append([curr_img_keypoint[0] + curr_img_keypoint_motion[0], curr_img_keypoint[1] + curr_img_keypoint_motion[1]])

        similarity_matrix = np.zeros((len(curr_img_keypoints), len(curr_img_expanded_samples_propagated_to_next_img)))
        for similarity_matrix_row_idx in range(similarity_matrix.shape[0]):
            for similarity_matrix_col_idx in range(similarity_matrix.shape[1]):
                similarity_matrix[similarity_matrix_row_idx, similarity_matrix_col_idx] = \
                        np.linalg.norm(np.array(curr_img_keypoints_propagated[similarity_matrix_row_idx]) - np.array(curr_img_expanded_samples_propagated_to_next_img[similarity_matrix_col_idx]))
        similarity_matrix = similarity_matrix / np.max(similarity_matrix)

        flow_result = ot_matching_matrix(similarity_matrix)

        # * forward-backward skeleton fusion, update boxs and skeletons in both frames
        # boxes: curr_video_bboxes and next_img_propagated_and_detected_bboxes,
        # joints: curr_img_all_skeletons and next_img_propagated_and_detected_bboxes_skeleton,
        # keypoints: curr_img_keypoints and curr_img_expanded_samples_propagated_to_next_img
        # meshgrid fitting, build a dict mapping curr_img_keypoints to flow field vectors
        map_former_coords_to_motion_vectors_hori = {}
        map_former_coords_to_motion_vectors_vert = {}
        for flow_result_row_idx in range(flow_result.shape[0]):
            if np.max(flow_result[flow_result_row_idx, :]) == 1:
                map_former_coords_to_motion_vectors_hori[curr_img_keypoints[flow_result_row_idx]] = curr_img_expanded_samples_propagated_to_next_img[np.argmax(flow_result[flow_result_row_idx, :])][0] - curr_img_keypoints[flow_result_row_idx][0]
                map_former_coords_to_motion_vectors_vert[curr_img_keypoints[flow_result_row_idx]] = curr_img_expanded_samples_propagated_to_next_img[np.argmax(flow_result[flow_result_row_idx, :])][1] - curr_img_keypoints[flow_result_row_idx][1]

        hori_motion_field = griddata(np.array(curr_img_keypoints), [map_former_coords_to_motion_vectors_hori[x] for x in map_former_coords_to_motion_vectors_hori.keys()], (np.mgrid[0:1:1280j, 0:1:720j]), method = 'cubic')
        vert_motion_field = griddata(np.array(curr_img_keypoints), [map_former_coords_to_motion_vectors_vert[x] for x in map_former_coords_to_motion_vectors_vert.keys()], (np.mgrid[0:1:1280j, 0:1:720j]), method = 'cubic')

        # for each box in former frame, find its counterpart in latter frame
        for curr_person_skeleton_idx in range(len(curr_img_all_skeletons['annorect'])):
            curr_person_skeleton = curr_img_all_skeletons['annorect'][curr_person_skeleton_idx]
            curr_person_bbox = [x for x in curr_video_bboxes if (curr_img_name in x['image_name'])][curr_img_all_skeletons['annorect'].index(curr_person_skeleton)]['bbox']
            left, top, width, height = curr_person_bbox[0], curr_person_bbox[1], curr_person_bbox[2], curr_person_bbox[3]
            propagated_bbox_left, propagated_bbox_top, propagated_bbox_right, propagated_bbox_bottom = curr_img.shape[1], curr_img.shape[0], 0, 0
            for curr_bbox_pixel_hori in range(int(left), int(left + width)):
                for curr_bbox_pixel_vert in range(int(top), int(top + height)):
                    propagated_bbox_left = min([curr_bbox_pixel_hori + hori_motion_field[curr_bbox_pixel_hori, curr_bbox_pixel_vert], propagated_bbox_left])
                    propagated_bbox_right = max([curr_bbox_pixel_hori + hori_motion_field[curr_bbox_pixel_hori, curr_bbox_pixel_vert], propagated_bbox_right])
                    propagated_bbox_top = min([curr_bbox_pixel_vert + vert_motion_field[curr_bbox_pixel_hori, curr_bbox_pixel_vert], propagated_bbox_top])
                    propagated_bbox_bottom = max([curr_bbox_pixel_vert + vert_motion_field[curr_bbox_pixel_hori, curr_bbox_pixel_vert], propagated_bbox_bottom])
            iou_list = []
            for next_img_propagated_and_detected_bboxes_idx in range(len(next_img_propagated_and_detected_bboxes)):
                left, top, width, height = next_img_propagated_and_detected_bboxes[next_img_propagated_and_detected_bboxes_idx]
                iou_list.append(compute_iou_single_box_normal([propagated_bbox_top, propagated_bbox_bottom, propagated_bbox_left, propagated_bbox_right], \
                                                              [top, top+height, left, left+width]))

            former_skeleton = copy.deepcopy(curr_person_skeleton['annopoints'][0]['point'])
            latter_skeleton = copy.deepcopy(next_img_propagated_and_detected_bboxes_skeleton[np.argmax(iou_list)])

            # only use latter for fixing former because latter is hrnet and does not conform to posetrack format
            former_normalizing_factor = sum([x['score'][0] for x in former_skeleton])
            for former_skeleton_joint_idx in range(num_joints_posetrack_format):
                former_skeleton[former_skeleton_joint_idx]['score'][0] = former_skeleton[former_skeleton_joint_idx]['score'][0] / former_normalizing_factor
            latter_normalizing_factor = np.sum(latter_skeleton[:, 2])
            for latter_skeleton_joint_idx in range(num_joints_hrnet_format):
                latter_skeleton[latter_skeleton_joint_idx, 2] = latter_skeleton[latter_skeleton_joint_idx, 2] / latter_normalizing_factor

            whether_need_to_update_bbox = 0
            for hrnet_semantic in hrnet_semantics.keys():
                if former_skeleton[posetrack_submission_semantics[latter_skeleton[hrnet_semantic]]]['score'][0] < latter_skeleton[hrnet_semantic, 2]:
                    former_skeleton[posetrack_submission_semantics[latter_skeleton[hrnet_semantic]]]['x'][0] = latter_skeleton[hrnet_semantic, 1]
                    former_skeleton[posetrack_submission_semantics[latter_skeleton[hrnet_semantic]]]['y'][0] = latter_skeleton[hrnet_semantic, 0]
                    whether_need_to_update_bbox = 1

            curr_person_skeleton['annopoints'][0]['point'] = former_skeleton
            curr_img_all_skeletons['annorect'][curr_person_skeleton_idx] = curr_person_skeleton
            if whether_need_to_update_bbox == 1:
                update_bbox_left, update_bbox_top, update_bbox_right, update_bbox_bottom = curr_img.shape[1], curr_img.shape[0], 0, 0
                for former_skeleton_idx in range(len(former_skeleton)):
                    update_bbox_left = min([former_skeleton[former_skeleton_idx]['x'][0], update_bbox_left])
                    update_bbox_right = max([former_skeleton[former_skeleton_idx]['x'][0], update_bbox_right])
                    update_bbox_top = min([former_skeleton[former_skeleton_idx]['y'][0], update_bbox_top])
                    update_bbox_bottom = max([former_skeleton[former_skeleton_idx]['y'][0], update_bbox_bottom])
                curr_video_bboxes_part_belonging_to_curr_frame = [x for x in curr_video_bboxes if (curr_img_name in x['image_name'])]
                curr_video_bboxes_part_not_belonging_to_curr_frame = [x for x in curr_video_bboxes if (curr_img_name not in x['image_name'])]
                curr_video_bboxes_part_belonging_to_curr_frame[curr_person_skeleton_idx]['bbox'] = [update_bbox_left, \
                                                                                                    update_bbox_top, \
                                                                                                    update_bbox_right - update_bbox_left, \
                                                                                                    update_bbox_bottom - update_bbox_top]
                curr_video_bboxes = curr_video_bboxes_part_belonging_to_curr_frame + curr_video_bboxes_part_not_belonging_to_curr_frame
        curr_video_posewarper_skeletons[curr_img_all_skeletons_idx] = curr_img_all_skeletons

    # save updated predictions
    # out_file = open(os.path.join(dst_json_dir, folder_name + '.json'), "w")
    # json.dump(curr_json, out_file)
    # out_file.close()







