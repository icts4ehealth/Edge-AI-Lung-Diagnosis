# Edge-AI-Lung-Diagnosis

This repository contains all necessary resources to reproduce the embedded inference experiments presented in our paper:

**Performance Comparison of Embedded AI Solutions for Classification and Detection in Lung Disease Diagnosis**

## Objective

To evaluate the feasibility of deploying quantized deep learning models for lung disease classification on a Raspberry Pi 4, enabling low-latency AI inference in resource-constrained environments.

---

## 📁 Repository Structure

- `models/` – Pre-trained and quantized `.tflite` models (INT8)
- `inference/` – Inference script for Raspberry Pi (`rpi_inference.py`)
- `sample_images/` – Sample full-resolution chest X-ray image
- `requirements.txt` – Python dependencies for reproducibility

---

## 🧠 Provided Models

The following post-training quantized models (INT8 TFLite format) are available:

- `MobileNetV3-Large`
- `EfficientNetV2-B0`
- `DenseNet201`
- `ResNet101`
- `Xception`
- `InceptionResNetV2`

---

## 🔁 Reproducibility Steps

```bash
### 1. Clone This Repository
git clone https://github.com/icts4ehealth/Edge-AI-Lung-Diagnosis.git
cd Edge-AI-Lung-Diagnosis

### 2. Environment Setup (Python ≥ 3.8)
    We recommend using a virtual environment:
    
    python3 -m venv YOUR_vENV
    source YOUR_vENV/bin/activate

    Install dependencies:

    pip install --upgrade pip
    pip install -r requirements.txt

    These dependencies are tested on Raspberry Pi 4 running Raspberry Pi OS (64-bit) aarch64.
    
### 3. Run Inference on Raspberry Pi


    Use the provided script to run inference with a quantized model and sample image (Please note that inference script takes direct arguments (not --model and --image flags)):

    python3 inference/rpi_inference.py models/mobilenetv3_int8.tflite sample_images/test_sample_bacterial.png
	
    The script will output:

         Predicted class
         Softmax probabilities
         Timing breakdown (model load time, image preprocessing time, inference time, and total latency)

Notes
The test image (test_sample_bacterial.png) has a resolution of 2566×2566 and is representative of real-world clinical X-rays.

In our experiments, model load time was not included in the end-to-end latency as it is performed only once during initialization.

