import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Fix path — adjust if running from a different directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
tflite_path = os.path.join(BASE_DIR, "saved_models", "face_model.tflite")
test_dir    = os.path.join(BASE_DIR, "..", "data", "clean_dataset", "test")

print(f"Loading model from: {tflite_path}")
print(f"Test dir: {test_dir}")

interp = tf.lite.Interpreter(model_path=tflite_path)
interp.allocate_tensors()
ind  = interp.get_input_details()[0]
outd = interp.get_output_details()[0]

print(f"Input  scale={ind['quantization'][0]:.8f}  zero_point={ind['quantization'][1]}")
print(f"Input  dtype={ind['dtype']}  shape={ind['shape']}")

def run_inference(img_path):
    img = Image.open(img_path).convert("RGB").resize((160, 120))
    arr = np.array(img, dtype=np.float32) / 255.0
    # Apply same quantization as ESP32: r - 128
    inp = (np.array(img, dtype=np.float32) - 128).astype(np.int8)
    interp.set_tensor(ind['index'], inp[np.newaxis])
    interp.invoke()
    scores = interp.get_tensor(outd['index'])[0]
    labels = ['you (0)', 'unknown face (1)', 'background (2)']
    winner = labels[np.argmax(scores)]
    print(f"    Scores: [{scores[0]:4d}, {scores[1]:4d}, {scores[2]:4d}]  → {winner}")

# Test on samples from each class
for class_id in ['0', '2', '3']:
    class_path = os.path.join(test_dir, class_id)
    if not os.path.exists(class_path):
        print(f"\n--- Class {class_id}: folder not found ---")
        continue
    files = sorted([f for f in os.listdir(class_path)
                    if f.lower().endswith('.jpg')])[:5]
    print(f"\n--- Class {class_id} ({len(files)} samples) ---")
    for f in files:
        print(f"  {f}:")
        run_inference(os.path.join(class_path, f))