import json
import numpy as np
import os
import copy
import cv2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import sys
sys.path.append('Base_pose_estimation_model/Base_pose_estimation_model')
sys.path.append('Base_pose_estimation_model/Base_pose_estimation_model/lib')
# from lib import *
from lib.models import pose_hrnet_PoseAgg
from lib.models import pose_hrnet
import argparse
from lib.config import cfg
from lib.config import update_config
from lib.core.inference import get_final_preds

src_file_format = {
    'head top': 14,     'nose': 13,     'neck': 12,     'left shoulder': 9,     'right shoulder': 8,     'left elbow': 10,
    'right elbow': 7,     'left wrist': 11,     'right wrist': 6,     'left hip': 3,     'right hip': 2,     'left knee': 4,
    'right knee': 1,     'left ankle': 5,     'right ankle': 0
}

dst_file_format = {'nose': 0, 'neck': 1, 'head top': 2, 'left shoulder': 5, 'right shoulder': 6, 'left elbow': 7, 'right elbow': 8,
                   'left wrist': 9, 'right wrist': 10, 'left hip': 11, 'right hip': 12, 'left knee': 13, 'right knee': 14,
                   'left ankle': 15, 'right ankle': 16}

########################################################################################################################################
################################################################# Input data ###########################################################
########################################################################################################################################
bbox_confidence_thresh = 0.2
num_joints = 17
kpt_db = []
num_boxes = 0
image_width, image_height = 288, 384
image_size = np.array([image_width, image_height])
aspect_ratio = image_width * 1.0 / image_height
pixel_std = 200
timestep_delta_range = 5
is_posetrack18 = True
color_rgb = True
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
num_joints = 17
target_type = 'gaussian'
heatmap_size = np.array([72, 96])
sigma = 3
use_different_joints_weight = False
joints_weight = np.array([[1. ], [1. ], [1. ], [1. ], [1. ], [1. ], [1. ], [1.2], [1.2], [1.5], [1.5], [1. ], [1. ], [1.2], [1.2], [1.5], [1.5]])

########################################################################################################################################
############################################################## define functions ########################################################
########################################################################################################################################
def _box2cs(box):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h)

def _xywh2cs(x, y, w, h):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale

def read_image(image_path):
    r = open(image_path,'rb').read()
    img_array = np.asarray(bytearray(r), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    return img

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def generate_target(joints, joints_vis):
    '''
    :param joints:  [num_joints, 3]
    :param joints_vis: [num_joints, 3]
    :return: target, target_weight(1: visible, 0: invisible)
    '''
    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    target_weight[:, 0] = joints_vis[:, 0]

    assert target_type == 'gaussian', \
        'Only support gaussian map now!'

    if target_type == 'gaussian':
        target = np.zeros((num_joints, heatmap_size[1], heatmap_size[0]), dtype=np.float32)

        tmp_size = sigma * 3

        for joint_id in range(num_joints):
            feat_stride = image_size / heatmap_size
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    if use_different_joints_weight:
        target_weight = np.multiply(target_weight, joints_weight)

    return target, target_weight

def copy_prev_models(prev_models_dir, model_dir):
    import shutil

    vc_folder = '/hdfs/' \
        + '/' + os.environ['PHILLY_VC']
    source = prev_models_dir
    # If path is set as "sys/jobs/application_1533861538020_2366/models" prefix with the location of vc folder
    source = vc_folder + '/' + source if not source.startswith(vc_folder) \
        else source
    destination = model_dir

    if os.path.exists(source) and os.path.exists(destination):
        for file in os.listdir(source):
            source_file = os.path.join(source, file)
            destination_file = os.path.join(destination, file)
            if not os.path.exists(destination_file):
                print("=> copying {0} to {1}".format(
                    source_file, destination_file))
                shutil.copytree(source_file, destination_file)
    else:
        print('=> {} or {} does not exist'.format(source, destination))

########################################################################################################################################
############################################################# Initialize models ########################################################
########################################################################################################################################
# self_inference_pose_model_args = argparse.Namespace(cfg='/home/vipuser/Downloads/Base_pose_estimation_model/experiments/posetrack/hrnet/temporal_pose_aggregation.yaml', opts=['OUTPUT_DIR', '/home/vipuser/Downloads/Base_pose_estimation_model/val_results/out/', 'LOG_DIR', '/home/vipuser/Downloads/Base_pose_estimation_model/val_results/log/', 'DATASET.NUM_LABELED_VIDEOS', '-1', 'DATASET.NUM_LABELED_FRAMES_PER_VIDEO', '-1', 'DATASET.JSON_DIR', '/home/vipuser/Downloads/Base_pose_estimation_model/supp_files/posetrack18_json_files/', 'DATASET.IMG_DIR', '/home/vipuser/Downloads/Dataset/PoseTrack2018/', 'TEST.MODEL_FILE', '/home/vipuser/Downloads/Base_pose_estimation_model/supp_files/pretrained_models/posetrack18.pth', 'TEST.COCO_BBOX_FILE', '/home/vipuser/Downloads/Base_pose_estimation_model/supp_files/posetrack18_precomputed_boxes/val_boxes.json', 'POSETRACK_ANNOT_DIR', '/home/vipuser/Downloads/Dataset/PoseTrack2018/posetrack_annotations/annotations/val/', 'TEST.USE_GT_BBOX', 'False', 'EXPERIMENT_NAME', '', 'DATASET.IS_POSETRACK18', 'True', 'TEST.IMAGE_THRE', '0.2', 'PRINT_FREQ', '5000'], modelDir='', logDir='', dataDir='', prevModelDir='')
# update_config(cfg, self_inference_pose_model_args)
# if self_inference_pose_model_args.prevModelDir and self_inference_pose_model_args.modelDir:
#     # copy pre models for philly
#     copy_prev_models(self_inference_pose_model_args.prevModelDir, self_inference_pose_model_args.modelDir)
#
# # cudnn related setting
# cudnn.benchmark = cfg.CUDNN.BENCHMARK
# torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
# torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
#
# self_inference_pose_model_model = eval(cfg.MODEL.NAME + '.get_pose_net')(
#     cfg, is_train=False
# )
#
# if cfg.TEST.MODEL_FILE:
#     self_inference_pose_model_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
#
# self_inference_pose_model_model = torch.nn.DataParallel(self_inference_pose_model_model, device_ids=cfg.GPUS).cuda()
# self_inference_pose_model_model.eval()
################### sys.path = sys_path_ori

########################################################################################################################################
############################################################## Preprocessing ###########################################################
########################################################################################################################################
def self_inference_pose_model(det_res, curr_box_left, curr_box_top, curr_box_width, curr_box_height, self_inference_pose_model_model):
    img_name = det_res['image_name']
    box = [curr_box_left, curr_box_top, curr_box_width, curr_box_height] # det_res['bbox']
    nframes = det_res['nframes']
    input_img = det_res['image']

    center, scale = _box2cs(box)
    joints_3d = np.zeros((num_joints, 3), dtype=np.float)
    joints_3d_vis = np.ones(
        (num_joints, 3), dtype=np.float)
    curr_box_info = {
        'image': img_name,
        'center': center,
        'scale': scale,
        'score': 1.0,
        'joints_3d': joints_3d,
        'joints_3d_vis': joints_3d_vis,
        'nframes': nframes
    }

    ########################################################################################################################################
    ############################################# Prepare stacked augmented data ###########################################################
    ########################################################################################################################################
    db_rec = copy.deepcopy(curr_box_info)

    image_file = db_rec['image']
    prev_image_file1 = db_rec['image']
    prev_image_file2 = db_rec['image']
    next_image_file1 = db_rec['image']
    next_image_file2 = db_rec['image']

    filename = db_rec['filename'] if 'filename' in db_rec else ''
    imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

    data_numpy = input_img # read_image(image_file)

    ##### supporting frames
    T = timestep_delta_range
    temp = prev_image_file1.split('/')
    prev_nm = temp[len(temp)-1]
    ref_idx = int(prev_nm.replace('.png','')) # int(prev_nm.replace('.jpg','')) # Yalong revised on 20220529

    ### setting deltas
    prev_delta1 = -1
    prev_delta2 = -2
    next_delta1 = 1
    next_delta2 = 2

    #### image indices
    prev_idx1 = ref_idx + prev_delta1
    prev_idx2 = ref_idx + prev_delta2
    next_idx1 = ref_idx + next_delta1
    next_idx2 = ref_idx + next_delta2

    if 'nframes' in db_rec:
        nframes = db_rec['nframes']
        if not is_posetrack18:
            prev_idx1 = np.clip(prev_idx1,1,nframes)
            prev_idx2 = np.clip(prev_idx2,1,nframes)
            next_idx1 = np.clip(next_idx1,1,nframes)
            next_idx2 = np.clip(next_idx2,1,nframes)
        else:
            prev_idx1 = np.clip(prev_idx1,0,nframes-1)
            prev_idx2 = np.clip(prev_idx2,0,nframes-1)
            next_idx1 = np.clip(next_idx1,0,nframes-1)
            next_idx2 = np.clip(next_idx2,0,nframes-1)

    if is_posetrack18:
        z = 6
    else:
        z = 8

    ### delta -1
    new_prev_image_file1 = prev_image_file1.replace(prev_nm, str(prev_idx1).zfill(z) + '.jpg')
    #### delta -2
    new_prev_image_file2 = prev_image_file1.replace(prev_nm, str(prev_idx2).zfill(z) + '.jpg')
    ### delta 1
    new_next_image_file1 = next_image_file1.replace(prev_nm, str(next_idx1).zfill(z) + '.jpg')
    #### delta 2
    new_next_image_file2 = next_image_file1.replace(prev_nm, str(next_idx2).zfill(z) + '.jpg')

    ###### checking for files existence
    if os.path.exists(new_prev_image_file1):
       prev_image_file1 = new_prev_image_file1
    if os.path.exists(new_prev_image_file2):
       prev_image_file2 = new_prev_image_file2
    if os.path.exists(new_next_image_file1):
       next_image_file1 = new_next_image_file1
    if os.path.exists(new_next_image_file2):
       next_image_file2 = new_next_image_file2

    data_numpy_prev1 = input_img # read_image(prev_image_file1)
    data_numpy_prev2 = input_img # read_image(prev_image_file2)
    data_numpy_next1 = input_img # read_image(next_image_file1)
    data_numpy_next2 = input_img # read_image(next_image_file2)

    if color_rgb:
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
        data_numpy_prev1 = cv2.cvtColor(data_numpy_prev1, cv2.COLOR_BGR2RGB)
        data_numpy_prev2 = cv2.cvtColor(data_numpy_prev2, cv2.COLOR_BGR2RGB)
        data_numpy_next1 = cv2.cvtColor(data_numpy_next1, cv2.COLOR_BGR2RGB)
        data_numpy_next2 = cv2.cvtColor(data_numpy_next2, cv2.COLOR_BGR2RGB)

    joints = db_rec['joints_3d']
    joints_vis = db_rec['joints_3d_vis']

    c = db_rec['center']
    s = db_rec['scale']
    score = db_rec['score'] if 'score' in db_rec else 1
    r = 0

    trans = get_affine_transform(c, s, r, image_size)
    input = cv2.warpAffine(
        data_numpy,
        trans,
        (int(image_size[0]), int(image_size[1])),
        flags=cv2.INTER_LINEAR)

    input_prev1 = cv2.warpAffine(
        data_numpy_prev1,
        trans,
        (int(image_size[0]), int(image_size[1])),
        flags=cv2.INTER_LINEAR)
    input_prev2 = cv2.warpAffine(
        data_numpy_prev2,
        trans,
        (int(image_size[0]), int(image_size[1])),
        flags=cv2.INTER_LINEAR)
    input_next1 = cv2.warpAffine(
        data_numpy_next1,
        trans,
        (int(image_size[0]), int(image_size[1])),
        flags=cv2.INTER_LINEAR)
    input_next2 = cv2.warpAffine(
        data_numpy_next2,
        trans,
        (int(image_size[0]), int(image_size[1])),
        flags=cv2.INTER_LINEAR)

    if transform:
        input = transform(input)
        input_prev1 = transform(input_prev1)
        input_prev2 = transform(input_prev2)
        input_next1 = transform(input_next1)
        input_next2 = transform(input_next2)
        ############
    for i in range(num_joints):
        if joints_vis[i, 0] > 0.0:
            joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

    target, target_weight = generate_target(joints, joints_vis)

    target = torch.from_numpy(target)
    target_weight = torch.from_numpy(target_weight)

    meta = {
        'image': [image_file],
        'sup_image': [prev_image_file1],
        'filename': [filename],
        'imgnum': [imgnum],
        'joints': torch.from_numpy(joints).unsqueeze(0),
        'joints_vis': torch.from_numpy(joints_vis).unsqueeze(0),
        'center': torch.from_numpy(c).unsqueeze(0),
        'scale': torch.from_numpy(s).unsqueeze(0),
        'rotation': torch.tensor(r).unsqueeze(0),
        'score': torch.tensor(score).unsqueeze(0)
    }

    ########################################################################################################################################
    ######################################################## Inference #####################################################################
    ########################################################################################################################################
    # input, input_prev1, input_prev2, input_next1, input_next2, target, target_weight, meta
    input, input_prev1, input_prev2, input_next1, input_next2, target, target_weight = \
        input.unsqueeze(0), input_prev1.unsqueeze(0), input_prev2.unsqueeze(0), input_next1.unsqueeze(0), input_next2.unsqueeze(0), target.unsqueeze(0), target_weight.unsqueeze(0)
    concat_input = torch.cat((input, input_prev1, input_prev2, input_next1, input_next2), 1)
    with torch.no_grad():
        outputs = self_inference_pose_model_model(concat_input)
        if isinstance(outputs, list):
            output = outputs[-1]
        else:
            output = outputs

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        num_images = input.size(0)

        c = meta['center'].numpy()
        s = meta['scale'].numpy()
        score = meta['score'].numpy()

        preds, maxvals = get_final_preds(cfg, output.clone().cpu().numpy(), c, s)

        del target, target_weight #, output, outputs
        torch.cuda.empty_cache()

    return np.concatenate((preds, maxvals), axis=2)


