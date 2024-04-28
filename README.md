## Quick start
### Installation
1. Create a conda virtual environment and activate it:
   ```
   conda create -n posewarper python=3.7 -y
   source activate posewarper
   ```
2. Install pytorch v1.1.0:
   ```
   conda install pytorch=1.1.0 torchvision -c pytorch
   ```
3. Install mmcv:
   ```
   pip install mmcv
   ```
4. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   python setup.py install --user
   ```
5. Clone this repo. Let's refer to it as ${POSEWARPER_ROOT}:
   ```
   git clone https://github.com/facebookresearch/PoseWarper.git
   ```
6. Install other dependencies:
   ```
   cd ${POSEWARPER_ROOT}
   pip install -r requirements.txt
   ```
7. Compile external modules:
   ```
   cd ${POSEWARPER_ROOT}/lib
   make
   cd ${POSEWARPER_ROOT}/lib/deform_conv
   python setup.py develop
   ```
8. Download our pretrained models, and some supplementary data files from [this link](https://www.dropbox.com/s/ygfy6r8nitoggfq/PoseWarper_supp_files.zip?dl=0) and extract it to ${POSEWARPER_SUPP_ROOT} directory.
