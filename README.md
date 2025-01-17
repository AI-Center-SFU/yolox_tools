# YOLOX Training Guide

This guide will walk you through setting up a YOLOX environment and preparing your dataset.

---

## **1. Build and Run the Docker Container**

### **Step 1: Build the Docker Image**
```bash
docker build -t yolox_container .
```

### **Alternative: Load a Prebuilt Docker Image**
If you already have a saved Docker image file (e.g., `yolox_container.tar`), you can load it instead of building:
```bash
docker load -i yolox_container.tar
```

### **Step 2: Run the Container with GPU Support**
Run the container, exposing the necessary port for Jupyter Lab and sharing the dataset folder:
```bash
docker run --gpus all -it --rm -p 8888:8888 --memory=64g --memory-swap=64g --shm-size=32g -v /path/to/shared/dataset:/workspace/shared yolox_container
```
Replace `/path/to/shared/dataset` with the absolute path to your dataset directory on the host machine.

---

## **2. Dataset Preparation**

### **Dataset Structure**
Ensure your dataset is in COCO format and organized as follows:
```
your-dataset/
├── annotations/
│   ├── instances_train2017.json
│   ├── instances_val2017.json
├── train2017/
│   ├── <training set images>
├── val2017/
│   ├── <validation set images>
```
Place your dataset folder (e.g., `your-dataset`) into the shared directory specified when running the container.

---

## **3. Launch Jupyter Lab**

After starting the container, launch Jupyter Lab by navigating to `http://localhost:8888` in your web browser. You can find the Jupyter Lab access token in the terminal output of the container.

---

From here, follow the instructions provided in the Jupyter Notebook to set up your training environment, configure parameters, and start training.
