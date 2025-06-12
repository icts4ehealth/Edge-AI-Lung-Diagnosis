import sys
import os
import time
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter  # tflite-runtime version

# === Class Labels ===
class_names = ["Bacterial", "Covid-19", "Lung Opacity", "Normal", "Viral"]

def load_tflite_model(model_path):
    start = time.time()
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    load_time = time.time() - start
    return interpreter, load_time

def preprocess_image(image_path, input_shape):
    start = time.time()
    img = Image.open(image_path).convert("RGB")
    img = img.resize((input_shape[1], input_shape[2]))  # (H, W)
    img_array = np.asarray(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preprocess_time = time.time() - start
    return img_array.astype(np.float32), preprocess_time

def run_inference(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)

    start = time.time()
    interpreter.invoke()
    inference_time = time.time() - start

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0], inference_time

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 tflite_predict_with_latency_pi.py <model_path> <image_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]
    model_name = os.path.basename(model_path).replace(".tflite", "")
    output_filename = f"{model_name}_tflite_result.txt"

    # === Load model ===
    interpreter, model_load_time = load_tflite_model(model_path)
    input_shape = interpreter.get_input_details()[0]['shape']

    # === Preprocess image ===
    input_data, preprocess_time = preprocess_image(image_path, input_shape)

    # === Run inference ===
    softmax, inference_time = run_inference(interpreter, input_data)
    predicted_index = int(np.argmax(softmax))
    predicted_label = class_names[predicted_index]

    # === Print and save output ===
    output_text = (
        f"Model: {model_name}\n"
        f"Image: {os.path.basename(image_path)}\n"
        f"Predicted class     : {predicted_label} (index: {predicted_index})\n"
        f"Softmax probabilities: {np.round(softmax, 4).tolist()}\n\n"
        f" Model Load Time          : {model_load_time:.4f} sec\n"
        f" Image Preprocessing Time : {preprocess_time:.4f} sec\n"
        f" Inference Time           : {inference_time:.4f} sec\n"
        f" Total End-to-End Latency : {preprocess_time + inference_time:.4f} sec\n"
    )

    print(output_text)
    with open(output_filename, "w") as f:
        f.write(output_text)
    print(f"Saved result to: {output_filename}")

if __name__ == "__main__":
    main()
