# Prototype Optimization and Self-Training for Few-Shot 3D Point Cloud Semantic Segmentation

## Installation
- Install 'python' --This repo is tested with 'python 3.6.8'.
- Install 'pytorch' with CUDA -- This repo is tested with 'torch 1.4.0', 'CUDA 10.1'. 
It may work with newer versions, but that is not gauranteed.
- Install 'faiss' with cpu version
- Install 'torch-cluster' with the corrreponding torch and cuda version
    """
    pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
    """
- Install dependencies
    """
    pip install tensorboard h5py transforms3d
    """

## Usage
### Data preparation
#### S3DIS
1. Download [S3DIS Dataset Version 1.2]
2. Re-organize raw data into 'npy' files by running
   """
   cd ./preprocess
   python collect_s3dis_data.py --data_path $path_to_S3DIS_raw_data
   """
   The generated numpy files are stored in './datasets/S3DIS/scenes/data' by default.
3. To split rooms into blocks, run 
    """
    python ./preprocess/room2blocks.py --data_path ./datasets/S3DIS/scenes/
    """
    One folder named 'blocks_bs1_s1' will be generated under './datasets/S3DIS/' by default. 


#### ScanNet
1. Download [ScanNet V2]
2. Re-organize raw data into 'npy' files by running
    """
    cd ./preprocess
    python collect_scannet_data.py --data_path $path_to_ScanNet_raw_data
    """
   The generated numpy files are stored in './datasets/ScanNet/scenes/data' by default.
3. To split rooms into blocks, run 

    """
    python ./preprocess/room2blocks.py --data_path ./datasets/ScanNet/scenes/ --dataset scannet
    """
    
    One folder named 'blocks_bs1_s1' will be generated under './datasets/ScanNet/' by default. 


### Running 
#### Training
First, pretrain backbone on the available training set:
    """
    cd scripts
    bash pretrain_segmentor.sh
    """
Second, train our method:
    """	
    bash train.sh
    """

#### Evaluation
    """
    bash eval.sh
    """

