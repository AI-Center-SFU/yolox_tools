# YOLOX Training Guide

This guide will walk you through setting up a YOLOX environment, preparing your dataset, and training a model.

---

## **1. Build and Run the Docker Container**

### **Step 1: Build the Docker Image**
Build the Docker container with the YOLOX environment:
```bash
docker build -t yolox_container .
```

### **Step 2: Run the Container with GPU Support**
Run the container, exposing the necessary port for Jupyter Lab:
```bash
docker run --gpus all -it --rm -p 8888:8888 yolox_container
```

---

## **2. Launch Jupyter Lab**

After the container starts, launch Jupyter Lab by navigating to `http://localhost:8888` in your web browser. You can find the Jupyter Lab access token in the terminal output of the container.

---

## **3. Prepare Your Dataset**

### **Dataset Structure**
Ensure your dataset is in COCO format and organized as follows:
```
your-dataset.zip/
├── annotations/
│   ├── instances_train2017.json
│   ├── instances_val2017.json
├── train2017/
│   ├── <training set images>
├── val2017/
│   ├── <validation set images>
```

### **Upload the Dataset**
Upload the dataset (`your-dataset.zip`) to Jupyter Lab.

---

## **4. Configure and Train**

### **Step 1: Set the Dataset Path**
In the `train.ipynb` notebook, set the `zip_file_path` variable to the path of your uploaded dataset:
```python
zip_file_path = "your-dataset.zip"  # Replace with the path to your dataset
```

When you run this cell, if you see the message:
```
Created symbolic link
```
It means the dataset has been successfully prepared.

---

### **Step 2: Choose a YOLOX Model**
YOLOX supports various models. You can set the `-n` parameter to specify the desired model:
- `yolox-nano`
- `yolox-tiny`
- `yolox-s`
- `yolox-m`
- `yolox-l`
- `yolox-x`

For example, to train a **YOLOX-S** model, use:
```bash
-n yolox-s
```

---

### **Step 3: Start Training**
Run the following command to start training:
```bash
!python3 -m yolox.tools.train -expn "my_yolox_model" -n yolox-nano -d 1 -b 64 --fp16 -o --cache ram max_epoch 300 print_interval 50 eval_interval 30
```

Replace parameters as needed:
- `-n`: Choose the model (e.g., `yolox-s`, `yolox-m`, etc.).
- `max_epoch`: Number of training epochs.
- `print_interval`: Frequency of logging training progress.
- `eval_interval`: Frequency of evaluation during training.
- `-b`: Batch size (adjust based on GPU memory).

---

## **5. View Results**

Trained models and logs are saved in the `YOLOX_outputs` directory. You can evaluate the results or use the trained model for inference.

---