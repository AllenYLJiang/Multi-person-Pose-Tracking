## Quick start
### Installation
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


## Acknowledgement

Our PoseWarper implementation is built on top of [*Deep High Resolution Network implementation*](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch). We thank the authors for releasing their code.
