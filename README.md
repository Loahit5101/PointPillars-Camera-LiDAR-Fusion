# PointPillars with LiDAR-Camera Fusion for 3D Object Detection

- Implementation of PointPillars Network with camera fusion for 3D object Detection in Autonomous Driving.  
- Object Detection outputs from PointPillars and a 2D object detector is fused using a Fusion Network ([CLOCs](https://arxiv.org/pdf/2009.00784.pdf)) to achieve improved performance compared to LiDAR only baseline.

## Setup
Clone this repository and install required libraries in a seperate virtual environment.
```
git clone https://github.com/Loahit5101/PointPillars-Camera-LiDAR-Fusion.git   
pip install -r requirements.txt
```
## Dataset


1. Download [point cloud](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip)(29GB), [images](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip)(12 GB), [calibration files](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip)(16 M and [labels](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip)(5 MB).

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
python train.py --data_root your_dataset_path
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

[Soon]

## Tasks 
 
- [x] Implement PointPillars Network
- [x] Train and evaluate PointPillars Network
- [x] Implement CLOC Fusion Network
- [ ] Train and evaluate performance after fusion
- [ ] Add results
- [ ] Clean up code

## References
 
 1. [PointPillars](https://arxiv.org/pdf/1812.05784.pdf)  
 2. [CLOCs: Camera-LiDAR Object Candidates Fusion for 3D Object Detection](https://arxiv.org/pdf/2009.00784.pdf)
 


