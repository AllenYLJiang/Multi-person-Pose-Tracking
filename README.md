## Quick start
### Environment
1. Install pytorch:
   ```
   conda install pytorch torchvision 
   ```
2. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   cd Base_pose_estimation_model/Base_pose_estimation_model/cocoapi/PythonAPI
   python setup.py install --user
   ```
3. Install other dependencies:
   ```
   cd Base_pose_estimation_model/Base_pose_estimation_model
   pip install -r requirements.txt
   ```
4. Compile external modules:
   ```
   cd Base_pose_estimation_model/Base_pose_estimation_model/lib
   make
   cd Base_pose_estimation_model/Base_pose_estimation_model/lib/deform_conv
   python setup.py develop
   ```
5. Download our pretrained models, and some supplementary data files from [this link](https://pan.baidu.com/s/1y5dg4Z3rIxw9FOVXKcvy4w?pwd=72qh)  Code: 72qh and extract it to Base_pose_estimation_model/Base_pose_estimation_model directory.

### Dataset 
**For PoseTrack data**, please download the data from [PoseTrack download page](https://posetrack.net/users/download.php). The extracted PoseTrack images directory should look like this:
```
demo_data/PoseTrack2018
|--images
`-- |-- test
    `-- train
    `-- val
|--skeleton_predictions
`-- |-- test (one json file per video, storing the results of base pose estimation model)
    `-- train 
    `-- val (one json file per video, storing the results of base pose estimation model)
|--skeleton_predictions_refined
`-- |-- test (one json file per video, storing the results of base pose estimation model refined by finetune_joints.py)
    `-- train 
    `-- val (one json file per video, storing the results of base pose estimation model refined by finetune_joints.py)
|--box_predictions
`-- |-- test (all videos in one file, included in supplementary data files)
    `-- train 
    `-- val (all videos in one file, included in supplementary data files)
```

### Base Pose Estimation Model
```
python tools/test.py --cfg Base_pose_estimation_model/experiments/posetrack/hrnet/temporal_pose_aggregation.yaml OUTPUT_DIR Base_pose_estimation_model/val_results/out/ LOG_DIR Base_pose_estimation_model/val_results/log/ DATASET.NUM_LABELED_VIDEOS -1 DATASET.NUM_LABELED_FRAMES_PER_VIDEO -1 DATASET.JSON_DIR Base_pose_estimation_model/supp_files/posetrack18_json_files/ DATASET.IMG_DIR Dataset/PoseTrack2018/ TEST.MODEL_FILE Base_pose_estimation_model/supp_files/pretrained_models/posetrack18.pth TEST.COCO_BBOX_FILE Base_pose_estimation_model/supp_files/posetrack18_precomputed_boxes/val_boxes.json POSETRACK_ANNOT_DIR Dataset/PoseTrack2018/posetrack_annotations/annotations/val/ TEST.USE_GT_BBOX False EXPERIMENT_NAME "" DATASET.IS_POSETRACK18 True TEST.IMAGE_THRE 0.2 PRINT_FREQ 5000
```

### The spaese keypoint flow estimating model is being updated  



### Refinement with graph distance minimization and motion flow 
Download weights from [Weights](https://pan.baidu.com/s/1F3xlhcITxydSIIdaR2yokg?pwd=mvrd). Code: mvrd. Placed at: simple-HRNet-master/weights
```
python finetune_joints.py 
```

## Acknowledgement

Our PoseWarper implementation is built on top of [*Deep High Resolution Network implementation*](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) and [*Posewarper*](https://github.com/facebookresearch/PoseWarper). We thank the authors for releasing their code.
