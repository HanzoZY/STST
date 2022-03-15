# STST
STST: Spatial-Temporal Specialized Transformer for Skeleton-based Action Recognition in ACM MultiMedia2021


# Result
A little different with paper due the reimplementation.

 - NTU-RGB+D X-Sub: ~91.9%
 - SHREC-28: ~95.3%

# Packages Required
Python=3.6, Torch=1.6, Pickle, Numpy, Tqdm, Time, Opencv, Collections, Pyyaml, EasyDict, Shutil, Colorama, Argparse, TensorboardX, Itertools, Math, Inspect, Imutils

# Data Preparation

 - NTU-RGBD
    - Download the NTU-RGBD from the "https://rose1.ntu.edu.sg/dataset/actionRecognition"
    - Then you can use utils in folder gen_data to generate datasets for training.
 - SHREC
    - We have provided and split the dataset here.

# Training & Testing

- Train 
  - Change the config file depending on what you want.

    `python train.py --config ./config/shrec/shrec_stst_28.yaml`
- Test
  - Change the config file depending on what you want.
  
    `python eval.py --config ./workdir/val/shrec/stst_toy_28_val.yaml` 
  
     Here we provide a small version of the model that has been trained for you to test.

     
# Citation
Please cite the following paper if you use this repository in your reseach.

    @inproceedings{zhang2021stst,
        title={STST: Spatial-Temporal Specialized Transformer for Skeleton-based Action Recognition},
        author={Zhang, Yuhan and Wu, Bo and Li, Wen and Duan, Lixin and Gan, Chuang},
        booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
        pages={3229--3237},
        year={2021}
    }
# Reference
The code of this project is based on the DSTA-Net(Decoupled Spatial-Temporal Attention Network for Skeleton-Based Action-Gesture Recognition)

# Contact
For any questions, feel free to contact: `yuhanzhan9@gmail.com`
