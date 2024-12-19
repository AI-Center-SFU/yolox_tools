# yolox_train


docker build -t yolox_container .

docker run --gpus all -it --rm -p 8888:8888 yolox_container
