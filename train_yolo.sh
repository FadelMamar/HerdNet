# training routine args
yamldata="/home/ubuntu/workspace/wildlife_localizer_data/data_config.yaml"
model_path="/home/ubuntu/workspace/HerdNet/runs/detect/train/weights/best.pt"
epochs=100
imgsz=640
batch=32
single_cls=True
patience=20
device=0
dropout=0.0
iou=0.5
optimizer='Adam'
pretrained=True
cos_lr=False
lr0=0.001
lrf=0.00001
resume_training=True

# data augmentation args
rotation=45
mixup=0.4

# other
project="yolo-runs"


# activate environment
conda activate pytorch

# start training
yolo detect train data=$yamldata model=$model_path epochs=$epochs\
    imgsz=$imgsz batch=$batch signle_cls=$single_cls patience=$patience\
    device $device dropout=$dropout iou=$iou optimizer=$optimizer\
    pretrained=$pretrained cos_lr=$cos_lr lr0=$lr0 lrf=$lrf\
    degrees=$rotation mixup=$mixup\
    project=$project\
    resume=$resume_training

# deactivate environment
conda deactivate