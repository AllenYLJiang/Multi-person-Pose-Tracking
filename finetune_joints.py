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
#########################################################################################################################
########################################## initialize pose model ########################################################
#########################################################################################################################
import sys
sys.path.append('simple-HRNet-master')
from SimpleHRNet import SimpleHRNet
model = SimpleHRNet(48, 17, "simple-HRNet-master/weights/pose_hrnet_w48_384x288.pth", device=torch.device('cuda'))
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from finetune_joints_funcs import *

#########################################################################################################################
########################################## initialize flow model ########################################################
#########################################################################################################################
sys.path.append('flow_estimation')
sys.path.append('flow_estimation/core')
import argparse
from core.utils.utils import InputPadder
from core import create_model

args = argparse.Namespace(model='gmflownet', ckpt='flow_estimation/pretrained_models/flow.pth', \
                          use_mix_attn=False, mixed_precision=False, alternate_corr=False)
flow_model = torch.nn.DataParallel(create_model(args))
flow_model.load_state_dict(torch.load(args.ckpt))
DEVICE = 'cuda'
flow_model = flow_model.module
flow_model.to(DEVICE)
flow_model.eval()

#########################################################################################################################
########################################## define functions #############################################################
#########################################################################################################################
def associate_ids_former_latter_img(former_img, former_img_bboxes, former_img_keypoints, former_img_keypoints_mask, latter_img, latter_img_bboxes, latter_img_keypoints, latter_img_keypoints_mask):
    ####################################################################################################################################################################################
    # coarse-grained association
    ####################################################################################################################################################################################
    # former_latter_local_flow_estimations: enlarge the bboxes of 3 frames by 25% and the flows in regions inside bboxes are estimated
    former_latter_local_flow_estimations = np.zeros((former_img.shape[0], former_img.shape[1], 2))
    former_img_all_bboxes_left, former_img_all_bboxes_top, former_img_all_bboxes_right, former_img_all_bboxes_bottom = \
    former_img.shape[1], former_img.shape[0], 0, 0
    for former_img_bbox in former_img_bboxes + latter_img_bboxes:
        former_img_bbox_enlarged = enlarge_bbox(former_img_bbox, 0.25)
        former_img_bbox_left, former_img_bbox_top, former_img_bbox_width, former_img_bbox_height = int(
            round(former_img_bbox_enlarged['bbox'][0])), \
            int(round(former_img_bbox_enlarged['bbox'][1])), \
            int(round(former_img_bbox_enlarged['bbox'][2])), \
            int(round(former_img_bbox_enlarged['bbox'][3]))

        former_img_all_bboxes_left = min([former_img_all_bboxes_left, former_img_bbox_left])
        former_img_all_bboxes_top = min([former_img_all_bboxes_top, former_img_bbox_top])
        former_img_all_bboxes_right = max([former_img_all_bboxes_right, former_img_bbox_left + former_img_bbox_width])
        former_img_all_bboxes_bottom = max([former_img_all_bboxes_bottom, former_img_bbox_top + former_img_bbox_height])
        former_img_all_bboxes_left = max([former_img_all_bboxes_left, 0])
        former_img_all_bboxes_top = max([former_img_all_bboxes_top, 0])
        former_img_all_bboxes_right = min([former_img_all_bboxes_right, former_img.shape[1] - 1])
        former_img_all_bboxes_bottom = min([former_img_all_bboxes_bottom, former_img.shape[0] - 1])

    former_img_region = former_img[former_img_all_bboxes_top:former_img_all_bboxes_bottom, former_img_all_bboxes_left:former_img_all_bboxes_right, :]
    latter_img_region = latter_img[former_img_all_bboxes_top:former_img_all_bboxes_bottom, former_img_all_bboxes_left:former_img_all_bboxes_right, :]
    # the flow model requires RGB instead of BGR
    former_img_region = torch.from_numpy(cv2.cvtColor(former_img_region, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()[None].to(DEVICE)
    latter_img_region = torch.from_numpy(cv2.cvtColor(latter_img_region, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()[None].to(DEVICE)
    # per region sparse flow estimation
    padder = InputPadder(former_img_region.shape)
    former_img_region, latter_img_region = padder.pad(former_img_region, latter_img_region)
    flow_low, flow_up = flow_model(former_img_region, latter_img_region, iters=20, test_mode=True)
    flow_up = padder.unpad(flow_up)
    flow_up = flow_up[0].permute(1, 2, 0).cpu().detach().numpy()

    former_latter_local_flow_estimations[former_img_all_bboxes_top:former_img_all_bboxes_bottom, former_img_all_bboxes_left:former_img_all_bboxes_right, :] = flow_up

    # propagate boxes in former image to latter image
    former_img_bboxes_propagated_to_latter_img = []
    for former_img_bbox in former_img_bboxes:
        former_img_bboxes_propagated_to_latter_img.append(
            bbox_propagation(former_img_bbox['bbox'], former_latter_local_flow_estimations))
    # iou matrix
    former_latter_img_bbox_iou_matrix = np.zeros((len(former_img_bboxes_propagated_to_latter_img), len(latter_img_bboxes)))
    former_latter_img_bbox_cost_matrix = np.ones((len(former_img_bboxes_propagated_to_latter_img), len(latter_img_bboxes))) * (-1)
    for former_img_bbox_propagated_to_latter_img_idx in range(len(former_img_bboxes_propagated_to_latter_img)):
        for latter_img_bbox_idx in range(len(latter_img_bboxes)):
            # lefttop, righttop, leftbottom, rightbottom
            former_img_bbox_propagated_to_latter_img = former_img_bboxes_propagated_to_latter_img[
                former_img_bbox_propagated_to_latter_img_idx]
            former_img_bbox_propagated_to_latter_img_top = min([former_img_bbox_propagated_to_latter_img[0][1], former_img_bbox_propagated_to_latter_img[1][1]])
            former_img_bbox_propagated_to_latter_img_bottom = max([former_img_bbox_propagated_to_latter_img[2][1], former_img_bbox_propagated_to_latter_img[3][1]])
            former_img_bbox_propagated_to_latter_img_left = min([former_img_bbox_propagated_to_latter_img[0][0], former_img_bbox_propagated_to_latter_img[2][0]])
            former_img_bbox_propagated_to_latter_img_right = max([former_img_bbox_propagated_to_latter_img[1][0], former_img_bbox_propagated_to_latter_img[3][0]])
            latter_img_bbox = latter_img_bboxes[latter_img_bbox_idx]

            former_latter_img_bbox_iou_matrix[former_img_bbox_propagated_to_latter_img_idx, latter_img_bbox_idx] = \
                compute_iou_single_box_normal([former_img_bbox_propagated_to_latter_img_top, former_img_bbox_propagated_to_latter_img_bottom, \
                                               former_img_bbox_propagated_to_latter_img_left, former_img_bbox_propagated_to_latter_img_right], \
                                              [latter_img_bbox['bbox'][1], latter_img_bbox['bbox'][1] + latter_img_bbox['bbox'][3], \
                                               latter_img_bbox['bbox'][0], latter_img_bbox['bbox'][0] + latter_img_bbox['bbox'][2]])
    ####################################################################################################################################################################################
    # fine-grained association
    # for pairs with iou > 0, perform fine-grained matching with flow estimation, for each joint, determine joint-joint similarity by averaging the inter-keypoint distances
    # do not need to consider similarity in colors because inter-keypoint distances show intensity similarity and have high discriminability
    ####################################################################################################################################################################################
    for former_img_human_idx in range(len(former_img_bboxes_propagated_to_latter_img)):
        for latter_img_human_idx in range(len(latter_img_bboxes)):
            if former_latter_img_bbox_iou_matrix[former_img_human_idx, latter_img_human_idx] == 0:
                continue
            human_to_human_overall_matching_error, former_human_to_latter_human_per_joint_visibility_score = \
                select_from_twice_propagated_samples(latter_img_keypoints, latter_img_keypoints_mask, latter_img_human_idx,
                                                     former_img_keypoints, former_img_keypoints_mask, former_img_human_idx,
                                                     former_latter_local_flow_estimations, former_img, latter_img)
            former_latter_img_bbox_cost_matrix[former_img_human_idx, latter_img_human_idx] = human_to_human_overall_matching_error
    former_latter_img_bbox_cost_matrix_max_value = np.max(former_latter_img_bbox_cost_matrix)
    assert(former_latter_img_bbox_cost_matrix_max_value > 0)
    former_latter_img_bbox_cost_matrix[np.where(former_latter_img_bbox_cost_matrix == -1)] = former_latter_img_bbox_cost_matrix_max_value * 2
    ###################################################################################################################################################################################
    # associate propagated boxes with detected boxes in latter image
    ###################################################################################################################################################################################
    # former_latter_img_bbox_matching_matrix = ot_matching_matrix(former_latter_img_bbox_cost_matrix)
    # # if iou too small, remove connection
    # former_latter_img_bbox_matching_matrix[np.where(former_latter_img_bbox_iou_matrix < 0.4)] = 0
    # # a pair must have both highest iou and lowest cost
    # for latter_img_human_idx in range(former_latter_img_bbox_matching_matrix.shape[1]):
    #     if np.argmax(former_latter_img_bbox_matching_matrix[:, latter_img_human_idx]) != np.argmin(former_latter_img_bbox_cost_matrix[:, latter_img_human_idx]):
    #         former_latter_img_bbox_matching_matrix[:, latter_img_human_idx] = 0
    #     # if np.argmax(former_latter_img_bbox_matching_matrix[:, latter_img_human_idx]) != np.argmax(former_latter_img_bbox_iou_matrix[:, latter_img_human_idx]):
    #     #     former_latter_img_bbox_matching_matrix[:, latter_img_human_idx] = 0
    ###################################################################################################################################################################################
    # store visibility scores of human joints in latter image, generate masks
    ###################################################################################################################################################################################
    latter_img_joint_level_vis_scores = {}
    for latter_img_human_idx in range(len(latter_img_bboxes)):
        if np.max(former_latter_img_bbox_iou_matrix[:, latter_img_human_idx]) < 0.5:
            latter_img_joint_level_vis_scores[latter_img_human_idx] = None
            continue
        former_img_human_idx = np.argmin(former_latter_img_bbox_cost_matrix[:, latter_img_human_idx])
        _, former_human_to_latter_human_per_joint_visibility_score = \
            select_from_twice_propagated_samples(latter_img_keypoints, latter_img_keypoints_mask, latter_img_human_idx,
                                                 former_img_keypoints, former_img_keypoints_mask, former_img_human_idx,
                                                 former_latter_local_flow_estimations, former_img, latter_img)
        latter_img_joint_level_vis_scores[latter_img_human_idx] = former_human_to_latter_human_per_joint_visibility_score

    return latter_img_joint_level_vis_scores

#########################################################################################################################
########################################## initialize parameters ########################################################
#########################################################################################################################
bbox_all_videos = 'demo_data/PoseTrack2018/box_predictions/test/test_boxes.json' # boxes in all images from all videos
curr_split = 'test'
src_video_dir = 'demo_data/PoseTrack2018/images'
skeleton_pred_dir = 'demo_data/PoseTrack2018/skeleton_predictions'
skeleton_pred_refined_dir = 'demo_data/PoseTrack2018/skeleton_predictions_refined'
dst_vis_dir = 'demo_data/PoseTrack2018/vis_results'
box_thresh = 0.2 # as set in the reference code
Num_samples_each_human_in_former_img = 100 #1024, 2048
qualified_joint_conf_thresh = 0.1
iou_thresh = 0.7
num_joints_posetrack_format = 15
num_joints_hrnet_format = 17
same_face_thresh = 0.5
max_hori_flow_vector, max_vert_flow_vector = 22.92, 17.55
blur_deeper_thresh = 0.3
enlarge_heatmap_ratio = 2
convert_coco_to_posetrack_format, perform_heatmap_nms, enlarge_heatmap = 1, 0, 1
valid_gaussian_kernel_area = 300
temporal_stride_between_imgs_in_each_pair = 4

# traverse videos
for curr_video_skeleton_pred_file in os.listdir(os.path.join(skeleton_pred_dir, curr_split)):
    video_name = curr_video_skeleton_pred_file.split('.')[0]
    curr_video_bboxes = [x for x in json.load(open(bbox_all_videos, 'r')) if (video_name in x['image_name'])]
    curr_video_bboxes = [x for x in curr_video_bboxes if x['score'] >= box_thresh]
    # traverse pairs of frames
    # sample pairs of frames each with 3 frames
    for curr_img_pair_first_idx in range(0, len(os.listdir(os.path.join(src_video_dir, curr_split, video_name))) - 1 - 2 * temporal_stride_between_imgs_in_each_pair):
        ###########################################################################################################################################################
        ################################ visualize joints and keypoints in former image ###########################################################################
        ###########################################################################################################################################################
        former_img_name = os.listdir(os.path.join(src_video_dir, curr_split, video_name))[curr_img_pair_first_idx]
        latter_img_name = os.listdir(os.path.join(src_video_dir, curr_split, video_name))[curr_img_pair_first_idx + temporal_stride_between_imgs_in_each_pair]
        third_img_name = os.listdir(os.path.join(src_video_dir, curr_split, video_name))[curr_img_pair_first_idx + temporal_stride_between_imgs_in_each_pair * 2]

        former_img = cv2.imread(os.path.join(src_video_dir, curr_split, video_name, former_img_name))
        latter_img = cv2.imread(os.path.join(src_video_dir, curr_split, video_name, latter_img_name))
        third_img = cv2.imread(os.path.join(src_video_dir, curr_split, video_name, third_img_name))

        former_img_bboxes = [x for x in curr_video_bboxes if x['image_name'].split('/')[-1] == former_img_name]
        latter_img_bboxes = [x for x in curr_video_bboxes if x['image_name'].split('/')[-1] == latter_img_name]
        third_img_bboxes = [x for x in curr_video_bboxes if x['image_name'].split('/')[-1] == third_img_name]

        curr_video_baseposemodel_skeletons = json.load(open(os.path.join(skeleton_pred_dir, curr_split, curr_video_skeleton_pred_file), 'r'))['annolist']

        former_img_skeletons = [y['annopoints'][0]['point'] for y in [x for x in curr_video_baseposemodel_skeletons if (former_img_name in x['image']['name'])][0]['annorect']]
        latter_img_skeletons = [y['annopoints'][0]['point'] for y in [x for x in curr_video_baseposemodel_skeletons if (latter_img_name in x['image']['name'])][0]['annorect']]
        third_img_skeletons = [y['annopoints'][0]['point'] for y in [x for x in curr_video_baseposemodel_skeletons if (third_img_name in x['image']['name'])][0]['annorect']]

        # curr_img_joints: a list whose sublists are joints of humans
        # curr_img_keypoints: a list whose sublists are keypoints of humans
        curr_img_joints, curr_img_keypoints, curr_img_keypoints_mask = [], [], []
        # former_img_joints: the joints of all humans in former image
        # former_img_keypoints: the samples of all humans in former image
        # former_img_keypoints_mask: the joint semantic ids of samples of all humans in former image
        joint_color = generate_distinct_colors(len(former_img_bboxes))
        keypoint_color = copy.deepcopy(joint_color)
        former_img_joints, former_img_keypoints, former_img_keypoints_mask, vis_img_joints, vis_img_keypoints = locate_joints_keypoints_allhumans_former_img(
                former_img_bboxes, former_img, Num_samples_each_human_in_former_img, \
                former_img_skeletons, curr_img_joints, curr_img_keypoints, curr_img_keypoints_mask, max([max_hori_flow_vector, max_vert_flow_vector]) / 2, \
                joint_color, keypoint_color, draw_rectangle_or_not=0)
        # cv2.imwrite('Step1_former_img_init_joints.png', vis_img_joints)
        # cv2.imwrite('Step2_former_img_init_samples.png', vis_img_keypoints)

        ###########################################################################################################################################################
        ################################ visualize joints and keypoints in latter image ###########################################################################
        ###########################################################################################################################################################
        curr_img_joints, curr_img_keypoints, curr_img_keypoints_mask = [], [], []
        joint_color = generate_distinct_colors(len(latter_img_bboxes))
        keypoint_color = copy.deepcopy(joint_color)
        latter_img_joints, latter_img_keypoints, latter_img_keypoints_mask, vis_img_joints, vis_img_keypoints = locate_joints_keypoints_allhumans_former_img(
                latter_img_bboxes, latter_img, Num_samples_each_human_in_former_img * 2, \
                latter_img_skeletons, curr_img_joints, curr_img_keypoints, curr_img_keypoints_mask, max([max_hori_flow_vector, max_vert_flow_vector]) / 2, \
                joint_color, keypoint_color, draw_rectangle_or_not=0)
        # cv2.imwrite('Step1_latter_img_init_joints.png', vis_img_joints)
        # cv2.imwrite('Step2_latter_img_init_samples.png', vis_img_keypoints)

        ###########################################################################################################################################################
        ################################ visualize joints and keypoints in latter image ###########################################################################
        ###########################################################################################################################################################
        curr_img_joints, curr_img_keypoints, curr_img_keypoints_mask = [], [], []
        joint_color = generate_distinct_colors(len(third_img_bboxes))
        keypoint_color = copy.deepcopy(joint_color)
        third_img_joints, third_img_keypoints, third_img_keypoints_mask, vis_img_joints, vis_img_keypoints = locate_joints_keypoints_allhumans_former_img(
                third_img_bboxes, third_img, Num_samples_each_human_in_former_img * 2, \
                third_img_skeletons, curr_img_joints, curr_img_keypoints, curr_img_keypoints_mask, max([max_hori_flow_vector, max_vert_flow_vector]) / 2, \
                joint_color, keypoint_color, draw_rectangle_or_not=0)
        # cv2.imwrite('Step1_third_img_init_joints.png', vis_img_joints)
        # cv2.imwrite('Step2_third_img_init_samples.png', vis_img_keypoints)

        ###########################################################################################################################################################
        ################################ associate human graphs from frames #######################################################################################
        ###########################################################################################################################################################
        latter_img_joint_level_vis_scores = associate_ids_former_latter_img(former_img, former_img_bboxes, former_img_keypoints, former_img_keypoints_mask, latter_img, latter_img_bboxes, latter_img_keypoints, latter_img_keypoints_mask)
        third_img_joint_level_vis_scores = associate_ids_former_latter_img(former_img, former_img_bboxes, former_img_keypoints, former_img_keypoints_mask, third_img, third_img_bboxes, third_img_keypoints, third_img_keypoints_mask)
        # if a box in former cannot find counterpart in latter and third frames, remove it
        # !!!!!!!!!!!!!!!!!!! if two bboxes are matched but iou < 0.5, should not match them !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!! we recommend to ues joint-level propagation iou for matching instead of bbox-level which is too coarse-grained
        # np.sum(former_latter_img_bbox_matching_matrix, axis=1)
        # np.sum(former_third_img_bbox_matching_matrix, axis=1)

        ###########################################################################################################################################################
        ################################################################ mask out low-vis regions #################################################################
        ###########################################################################################################################################################
        latter_img_joints, latter_img_skeletons = mask_out_low_vis_regions_and_pose_estimation(latter_img_joint_level_vis_scores, latter_img_joints, latter_img_skeletons, latter_img_bboxes, latter_img, model)

        curr_video_baseposemodel_skeletons_revised = copy.deepcopy(curr_video_baseposemodel_skeletons)
        idx_of_latter_img_in_curr_video = [x for x in range(len(curr_video_baseposemodel_skeletons)) if curr_video_baseposemodel_skeletons[x]['image']['name'].split('/')[-1]==latter_img_name][0]
        for idx_in_latter_img_skeletons in range(len(latter_img_skeletons)):
            curr_video_baseposemodel_skeletons_revised[idx_of_latter_img_in_curr_video]['annorect'][idx_in_latter_img_skeletons]['annopoints'][0]['point'] = latter_img_skeletons[idx_in_latter_img_skeletons]

        out_file = open(os.path.join(skeleton_pred_refined_dir, curr_split, curr_video_skeleton_pred_file), "w")
        json.dump({'annolist': curr_video_baseposemodel_skeletons_revised}, out_file)
        out_file.close()







