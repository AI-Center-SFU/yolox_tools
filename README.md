Below is an updated inference section for your YOLOX Training Guide that uses the `YOLOXInferencer` from `utils.py` (instead of `tools/demo.py`). This example demonstrates how to use the inferencer in a standard Python script.

---

# YOLOX Training and Inference Guide

This guide will walk you through setting up a YOLOX environment, preparing your dataset, and performing both training and inference.

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

From here, follow the instructions provided in your Jupyter Notebook to set up your training environment, configure parameters, and start training.

---

Below is the updated **Inference with YOLOXInferencer** section in the guide. In this version, instead of requiring an explicit experiment configuration file (`exp_file`), you can simply specify the model name (e.g. `"yolox_m"`) which is more convenient.

---

## **4. Inference with YOLOXInferencer**

After training your model (or when using a pretrained checkpoint), you can perform inference using the `YOLOXInferencer` class from `utils.py`.

### **Example Usage**

Below is an example of how to use `YOLOXInferencer` in a Python script:

```python
import cv2
from yolox.utils import YOLOXInferencer  # YOLOXInferencer is defined in utils.py

# Instead of specifying the experiment configuration file,
# you can simply specify the model name.
model_name = "yolox_m"  # For example, "yolox_m" or "yolo-s"
ckpt_path = "/workspace/shared/YOLOX_outputs/yolox_m/best_ckpt.pth"  # Path to the checkpoint file

# Initialize the inferencer.
# Here, we set exp_file to None so that the inferencer uses the default configuration for the given model_name.
inferencer = YOLOXInferencer(exp_file=None, model_name=model_name, ckpt_path=ckpt_path)

# Specify the path to the image (or provide a numpy array in BGR format).
image_path = "/workspace/shared/your-dataset/val2017/sample.jpg"

# Perform inference.
predictions = inferencer.predict(image_path)
print("Predictions:", predictions)

# Visualize the detections.
img = cv2.imread(image_path)
img_with_dets = inferencer.visualize(predictions, img)

# Optionally, save the output image.
cv2.imwrite("output.jpg", img_with_dets)
```

### **Notes**

- In the example above, adjust the paths (`ckpt_path` and `image_path`) to match your setup.
- By setting `exp_file` to `None` and providing the `model_name`, the inferencer will automatically use the default configuration corresponding to that model.
- The `YOLOXInferencer` class handles image loading, pre-processing, model inference, and post-processing.
- The `visualize` method overlays detection boxes, confidence scores, and class labels on the original image. You can save or further process the resulting image as needed.

---

Follow these instructions to set up your environment for both training and inference with YOLOX. Happy training and testing!