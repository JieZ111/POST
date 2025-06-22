# Prototype Optimization and Self-Training for Few-Shot 3D Point Cloud Semantic Segmentation

## Code under the experimental setup of attMPTI [1]

### Installation
```
pip install torch==1.4.0
pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install faiss-cpu tensorboard h5py transforms3d
 ```

### Usage
#### Data preparation
**S3DIS**
1. **Download**: [S3DIS Dataset Version 1.2](http://buildingparser.stanford.edu/dataset.html).
2. **Preprocessing**: Re-organize raw data into `npy` files:
   ```
   cd ./preprocess
   python collect_s3dis_data.py --data_path $path_to_S3DIS_raw_data
   ```
   The generated numpy files will be stored in `PATH_to_S3DIS_processed_data/scenes`.
3. **Splitting Rooms into Blocks**:
    ```
   python ./preprocess/room2blocks.py --data_path ./datasets/S3DIS/scenes/
    ```


**ScanNet**
1. **Download**: [ScanNet V2](http://www.scan-net.org/).
2. **Preprocessing**: Re-organize raw data into `npy` files:
   ```
   cd ./preprocess
   python collect_scannet_data.py --data_path $path_to_ScanNet_raw_data
   ```
   The generated numpy files will be stored in `PATH_to_ScanNet_processed_data/scenes`.
3. **Splitting Rooms into Blocks**:
   ```
   python ./preprocess/room2blocks.py --data_path ./datasets/ScanNet/scenes/ --dataset scannet
   ```
    
#### Running 
##### Training
First, pretrain the segmentor which includes feature extractor module on the available training set:

	```
	cd scripts
	bash pretrain_segmentor.sh
	```
Second, train our method:
	```
	bash train.sh
	```

##### Evaluation
	```
	bash eval.sh
	```


## Code under the experimental setup of COSeg [2]

### Installation

```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch_points3d==1.3.0
pip install torch-scatter==2.1.1
pip install torch-points-kernels==0.6.10
pip install torch-geometric==1.7.2
pip install timm==0.9.2
pip install tensorboardX==2.6
pip install numpy==1.20.3
```
pip install pointops2


### Usage

#### Preprocessed Datasets
| Dataset | Download |
| ------------------ | -------|
| S3DIS | [Download link](https://drive.google.com/file/d/1frJ8nf9XLK_fUBG4nrn8Hbslzn7914Ru/view?usp=drive_link) |
| ScanNet | [Download link](https://drive.google.com/file/d/19yESBZumU-VAIPrBr8aYPaw7UqPia4qH/view?usp=drive_link) |

#### Data preparation

**S3DIS**
1. **Download**: [S3DIS Dataset Version 1.2](http://buildingparser.stanford.edu/dataset.html).
2. **Preprocessing**: Re-organize raw data into `npy` files:
   ```bash
   cd preprocess
   python collect_s3dis_data.py --data_path [PATH_to_S3DIS_raw_data] --save_path [PATH_to_S3DIS_processed_data]
   ```
   The generated numpy files will be stored in `PATH_to_S3DIS_processed_data/scenes`.
3. **Splitting Rooms into Blocks**:
    ```bash
    python room2blocks.py --data_path [PATH_to_S3DIS_processed_data]/scenes
    ```


**ScanNet**
1. **Download**: [ScanNet V2](http://www.scan-net.org/).
2. **Preprocessing**: Re-organize raw data into `npy` files:
	```bash
	cd preprocess
	python collect_scannet_data.py --data_path [PATH_to_ScanNet_raw_data] --save_path [PATH_to_ScanNet_processed_data]
	```
   The generated numpy files will be stored in `PATH_to_ScanNet_processed_data/scenes`.
3. **Splitting Rooms into Blocks**:
    ```bash
    python room2blocks.py --data_path [PATH_to_ScanNet_processed_data]/scenes
    ```

After preprocessing the datasets, a folder named `blocks_bs1_s1` will be generated under `PATH_to_DATASET_processed_data`. Make sure to update the `data_root` entry in the .yaml config file to `[PATH_to_DATASET_processed_data]/blocks_bs1_s1/data`.

#### Running
##### Training
First, pretrain the segmentor which includes feature extractor module on the available training set:
```bash
python3 train_backbone.py --config config/[PRETRAIN_CONFIG] save_path [PATH_to_SAVE_BACKBONE] cvfold [CVFOLD]
```

Next, let us start the few-shot training. Set the configs in `config/[CONFIG_FILE]` (`s3dis_COSeg_fs.yaml` or `scannetv2_COSeg_fs.yaml`) for few-shot training. Adjust `cvfold`, `n_way`, and `k_shot` according to your task:

```bash
# 1 way 1/5 shot
python3 main_fs.py --config config/[CONFIG_FILE] save_path [PATH_to_SAVE_MODEL] pretrain_backbone [PATH_to_SAVED_BACKBONE] cvfold [CVFOLD] n_way 1 k_shot [K_SHOT] num_episode_per_comb 1000
# 2 way 1/5 shot
python3 main_fs.py --config config/[CONFIG_FILE] save_path [PATH_to_SAVE_MODEL] pretrain_backbone [PATH_to_SAVED_BACKBONE] cvfold [CVFOLD] n_way 2 k_shot [K_SHOT] num_episode_per_comb 100
```

Note: By default, when `n_way=1`, `num_episode_per_comb` is set to `1000`. When `n_way=2`, `num_episode_per_comb` is adjusted to `100` to maintain consistency in test set magnitude.


##### Evaluation
For testing, modify `cvfold`, `n_way`, `k_shot` and `num_episode_per_comb` accordingly, then run:
```bash
python3 main_fs.py --config config/[CONFIG_FILE] test True eval_split test weight [PATH_to_SAVED_MODEL]
```

## Trained Models 
We provide some trained models at [Download link](https://drive.google.com/drive/folders/1U9OFfEdse2J6Qa8CxRiF7JBDgLHcwAUZ?usp=sharing). 

# Reference
[1] N. Zhao, T. Chua, G. H. Lee, Few-shot 3D point cloud semantic segmentation, in: IEEE Conference on Computer Vision and Pattern Recognition, 2021, pp. 8873–8882.

[2] Z. An, G. Sun, Y. Liu, F. Liu, Z. Wu, D. Wang, L. V. Gool, S. J. Belongie, Rethinking few-shot 3D point cloud semantic segmentation, in:IEEE Conference on Computer Vision and Pattern Recognition, 2024, pp. 3996–4006.

