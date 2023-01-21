# PointPillars with LiDAR-Camera Fusion for 3D Object Detection

- Implementation of PointPillars Network with camera fusion for 3D object Detection in Autonomous Driving.  
- Object Detection outputs from PointPillars and a 2D object detector are fused using a Fusion Network ([CLOCs](https://arxiv.org/pdf/2009.00784.pdf)) to achieve improved performance compared to LiDAR only baseline.

## Performance

|     Model     |   3D AP (Easy,Moderate,Hard)   |
| ------------- |:-------------:|
| PointPillars  |77.98, 57.86, 66.02|         
| PointPillars+CLOCs|         81.43, 60.05, 67.15         |
| **Improvement**|              **3.45, 2.19, 1.13**    |


## Setup
Clone this repository and install required libraries in a seperate virtual environment.
```
git clone https://github.com/Loahit5101/PointPillars-Camera-LiDAR-Fusion.git   
pip install -r requirements.txt
```
## Dataset


1. Download [point cloud](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip)(29GB), [images](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip)(12 GB), [calibration files](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip)(16 MB) and [labels](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip)(5 MB).

2. Pre-process KITTI dataset

    ```
    python pre_process_kitti.py --data_root your_dataset_path
    ```

    Expected structure:
    ```
    kitti_dataset
        |- training
            |- calib (#7481 .txt)
            |- image_2 (#7481 .png)
            |- label_2 (#7481 .txt)
            |- velodyne (#7481 .bin)
            |- velodyne_reduced (#7481 .bin)
        |- testing
            |- calib (#7518 .txt)
            |- image_2 (#7518 .png)
            |- velodyne (#7518 .bin)
            |- velodyne_reduced (#7518 .bin)
        |- kitti_gt_database (# 19700 .bin)
        |- kitti_infos_train.pkl
        |- kitti_infos_val.pkl
        |- kitti_infos_trainval.pkl
        |- kitti_infos_test.pkl
        |- kitti_dbinfos_train.pkl
    
    ```
## PointPillars

### Training
```
python train_pillars.py --data_root your_dataset_path
```

### Testing 
```
python test.py --ckpt pretrained_model_path --pc_path your_pc_path
```
### Evaluation
```
python evaluate.py --ckpt pretrained_model_path --data_root your_dataset_path
```
## CLOC Fusion Network
Code to train CLOCs is inside CLOC_fusion folder

## Dataset

CLOCs requires 3D detection results and 2D detection results (from Cascade R-CNN in this case) before nms step.
ANy 3D or 2D detector can be used with CLOC provided the detections are in KITTI format.

Run the below command to generate 3D detections or download from below link
```
python evaluate.py --ckpt pretrained_model_path --data_root your_dataset_path
```
2D and 3D detections can also be downloaded from this [link](https://drive.google.com/drive/folders/1sL91TnLjprSRiEzQtmC4NBaP2zm4flQe?usp=share_link).

```
python generate_data.py
```
Generated inputs are stored in input_data folder

Expected structure:
```
.
└── clocs_data
    ├── 2D
    │   ├── 000000.txt
    │   ├── 000001.txt
    │   └── 000002.txt
    ├── 3D
    │   ├── 000000.pkl
    │   ├── 000001.pkl
    │   └── 000002.pkl
    ├── index
    │   ├── train.txt
    │   ├── trainval.txt
    │   └── val.txt
    ├── info
    │   ├── kitti_infos_trainval.pkl
    │   └── kitti_infos_val.pkl
    └── input_data
        ├── 000000.pt
        ├── 000001.pt
        └── 000002.pt
```

### Training
```
python train.py 
```
### Testing 
```
python test.py 
```
### Evaluation
```
python evaluate.py 
```
### Pretrained models
Pretrained models are available in the pretrained models folder 

## References
 
 1. [PointPillars](https://arxiv.org/pdf/1812.05784.pdf)  
 2. [CLOCs: Camera-LiDAR Object Candidates Fusion for 3D Object Detection](https://arxiv.org/pdf/2009.00784.pdf)
 


