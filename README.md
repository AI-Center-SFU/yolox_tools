# **YOLOX Training Guide**

---

## **1. Build and Run the Docker Container**

### **Step 1: Build the Docker Image**
```bash
docker build -t yolox_container .
```

### **Step 2: Run the Container with GPU Support and Shared Folder**
```bash
docker run --gpus all -it --rm -p 8888:8888 --memory=64g --memory-swap=64g --shm-size=32g -v C:/shared:/workspace/shared yolox_container
```

---

## **2. Launch Jupyter Lab**

Access Jupyter Lab at `http://localhost:8888`.

---

## **3. Prepare Your Dataset**

Place the dataset (`your-dataset.zip`) in `C:\shared` to access it in the container as `/workspace/shared/your-dataset.zip`.

---

## **4. Configure and Train**

### **Step 1: Set the Dataset Path**
```python
zip_file_path = "/workspace/shared/your-dataset.zip"
```

### **Step 2: Choose a YOLOX Model**
Set `-n` to your desired model (e.g., `yolox-nano`, `yolox-s`, `yolox-x`).

---

### **Step 3: Start Training**

**Basic Training Command:**
```bash
!python3 -m yolox.tools.train -expn "yolox_training" -n yolox-nano -d 1 -b 64 --fp16 -o --cache ram max_epoch 300 print_interval 50 eval_interval 30 output_dir "/workspace/shared/YOLOX_outputs"
```

---

### **Explanation of Common Training Parameters:**
- `-n <model>`: Model name (e.g., `yolox-nano`, `yolox-s`, etc.).
- `-d`: Number of GPUs to use (e.g., `1` for a single GPU).
- `-b`: Batch size (adjust based on GPU memory).
- `--fp16`: Enables mixed-precision training.
- `--cache ram`: Caches the dataset in RAM for faster data loading.
- `max_epoch`: Total number of training epochs.
- `print_interval`: Frequency of logging progress.
- `eval_interval`: Frequency of evaluation during training.
- `output_dir`: Path where training results will be saved (set to `/workspace/shared/YOLOX_outputs` to store results in `C:\shared`).
- `--ckpt`: Path to a checkpoint file for fine-tuning (e.g., `/workspace/shared/YOLOX_outputs/yolox_pretrained.pth`).
- `--resume`: Resume training from a specific checkpoint.
- `--start_epoch`: Specify the epoch to resume training from.

---

### **Example Commands:**

1. **Start Training from Scratch:**
   ```bash
   !python3 -m yolox.tools.train -expn "yolox_training" -n yolox-nano -d 1 -b 64 --fp16 -o --cache ram max_epoch 300 print_interval 50 eval_interval 30 output_dir "/workspace/shared/YOLOX_outputs"
   ```

2. **Fine-tuning from a Pre-trained Model:**
   ```bash
   !python3 -m yolox.tools.train -expn "yolox_finetune" -n yolox-nano -d 1 -b 64 --fp16 -o --cache ram max_epoch 300 --ckpt "/workspace/shared/YOLOX_outputs/yolox_pretrained.pth"
   ```

3. **Resume Training after Interruption:**
   ```bash
   !python3 -m yolox.tools.train -expn "yolox_resume" -n yolox-nano -d 1 -b 64 --fp16 -o --cache ram max_epoch 3600 --resume --start_epoch 150 --ckpt "/workspace/shared/YOLOX_outputs/latest_ckpt.pth"
   ```

---

## **5. View Results**

Check the `/workspace/shared/YOLOX_outputs` folder for:
- `latest_ckpt.pth`: Last saved checkpoint.
- `best_ckpt.pth`: Best performing model.
- `train_log.txt`: Training logs.
