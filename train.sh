models=("resnet" "vgg" "google" "effnet")

for model in "${models[@]}"; do
    echo "model: $model start training"
    python train_main.py --model "$model"
    echo "model: $model finish training"
done