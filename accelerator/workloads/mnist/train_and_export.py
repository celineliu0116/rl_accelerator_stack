import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
from pathlib import Path

# =============================================================
# Step 1: Define the MLP
# =============================================================
# 784 input pixels -> 128 hidden neurons -> 10 output classes
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)          # flatten 28x28 image to 784
        x = torch.relu(self.fc1(x))  # hidden layer + ReLU
        x = self.fc2(x)              # output layer (raw scores)
        return x

# =============================================================
# Step 2: Train
# =============================================================
def train():
    # MNIST dataset - downloads automatically on first run
    transform = transforms.ToTensor()  # converts images to [0, 1] floats
    full_train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Split into 50k training + 10k validation (test set is NOT loaded here)
    train_data, val_data = random_split(full_train_data, [50000, 10000])
    print(f"Data split: {len(train_data)} train / {len(val_data)} validation")
    print(f"Test set: hidden until after training\n")
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1000)
    
    model = MLP()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    # Train for 15 epochs for more robust quantization
    for epoch in range(15):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Evaluate on VALIDATION set (not test set)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                output = model(images)
                predictions = output.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/15 | Loss: {total_loss:.2f} | Val Accuracy: {acc:.2f}%")
    
    return model

# =============================================================
# Step 3: Quantize weights from float32 -> int8 fixed-point
# =============================================================
def quantize_weights(model):
    """
    For each weight matrix/bias vector:
    - Find the max absolute value
    - Compute a scale factor so values map to [-127, 127]
    - Round to int8
    
    We also need to store the scale factors so the C code
    can de-scale the results after computation.
    """
    quantized = {}
    
    for name, param in model.named_parameters():
        data = param.detach().numpy()
        max_val = np.max(np.abs(data))
        
        # scale maps the range [-max_val, max_val] -> [-127, 127]
        scale = 127.0 / max_val if max_val > 0 else 1.0
        
        quantized_data = np.round(data * scale).astype(np.int8)
        
        quantized[name] = {
            'data': quantized_data,
            'scale': scale,
            'shape': data.shape
        }
        
        print(f"Quantized {name}: shape={data.shape}, scale={scale:.4f}, "
              f"range=[{quantized_data.min()}, {quantized_data.max()}]")
    
    return quantized

# =============================================================
# Step 4: Quantize input images (0.0-1.0 floats -> 0-127 ints)
# =============================================================
def quantize_input(image_tensor):
    """Convert a [0,1] float image to [0, 127] int8."""
    image = image_tensor.numpy().flatten()
    return np.round(image * 127).astype(np.int8)

# =============================================================
# Step 5: Export everything to C header files
# =============================================================
def export_to_c(quantized, test_data, num_test_images=100, out_dir="../runtime"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    
    # Pre-compute bias scales for accurate fixed-point inference
    s_w1 = quantized['fc1.weight']['scale']
    s_b1 = quantized['fc1.bias']['scale']
    s_w2 = quantized['fc2.weight']['scale']
    s_b2 = quantized['fc2.bias']['scale']
    
    # Layer 1 bias: accumulator scale = s_w1 * 127 (input scale)
    # To add bias correctly: bias_scaled = round(bias_q * s_w1 * 127 / s_b1)
    b1_q = quantized['fc1.bias']['data'].astype(np.int64)
    fc1_bias_scaled = np.round(b1_q * s_w1 * 127.0 / s_b1).astype(np.int32)
    
    # Layer 2 bias: pre-scale with same approach
    b2_q = quantized['fc2.bias']['data'].astype(np.int64)
    fc2_bias_scaled = np.round(b2_q * s_w2 * 127.0 / s_b2).astype(np.int32)
    
    # --- Export weights ---
    with open(out_dir / 'weights.h', 'w') as f:
        f.write("// Auto-generated by train_and_export.py\n")
        f.write("// Quantized INT8 weights with pre-scaled INT32 biases\n")
        f.write("#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n\n")
        
        # Export weight matrices as int8
        for name in ['fc1.weight', 'fc2.weight']:
            q = quantized[name]
            c_name = name.replace('.', '_')
            data = q['data']
            scale = q['scale']
            shape = q['shape']
            
            f.write(f"// {name}: shape={shape}, scale={scale:.6f}\n")
            f.write(f"#define {c_name.upper()}_SCALE {scale:.6f}f\n")
            
            flat = data.flatten()
            f.write(f"const int8_t {c_name}[{len(flat)}] = {{\n")
            for i in range(0, len(flat), 16):
                chunk = flat[i:i+16]
                f.write("    " + ", ".join(str(x) for x in chunk) + ",\n")
            f.write("};\n\n")
        
        # Export pre-scaled biases as int32
        f.write(f"// fc1.bias: pre-scaled to accumulator scale (bias_q * s_w1 * 127 / s_b1)\n")
        f.write(f"const int32_t fc1_bias[{len(fc1_bias_scaled)}] = {{\n")
        for i in range(0, len(fc1_bias_scaled), 8):
            chunk = fc1_bias_scaled[i:i+8]
            f.write("    " + ", ".join(str(x) for x in chunk) + ",\n")
        f.write("};\n\n")
        
        f.write(f"// fc2.bias: pre-scaled to accumulator scale (bias_q * s_w2 * 127 / s_b2)\n")
        f.write(f"const int32_t fc2_bias[{len(fc2_bias_scaled)}] = {{\n")
        for i in range(0, len(fc2_bias_scaled), 8):
            chunk = fc2_bias_scaled[i:i+8]
            f.write("    " + ", ".join(str(x) for x in chunk) + ",\n")
        f.write("};\n\n")
        
        # Write dimensions
        f.write("#define INPUT_SIZE 784\n")
        f.write("#define HIDDEN_SIZE 128\n")
        f.write("#define OUTPUT_SIZE 10\n\n")
        
        f.write("#endif // WEIGHTS_H\n")
    
    # --- Export test images ---
    with open(out_dir / 'test_images.h', 'w') as f:
        f.write("// Auto-generated test images from MNIST\n")
        f.write("#ifndef TEST_IMAGES_H\n#define TEST_IMAGES_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define NUM_TEST_IMAGES {num_test_images}\n\n")
        
        # Store expected labels
        labels = []
        for i in range(num_test_images):
            image, label = test_data[i]
            labels.append(label)
            q_image = quantize_input(image)
            
            f.write(f"// Test image {i}: label = {label}\n")
            f.write(f"const int8_t test_image_{i}[784] = {{\n")
            flat = q_image.flatten()
            for j in range(0, len(flat), 16):
                chunk = flat[j:j+16]
                f.write("    " + ", ".join(str(x) for x in chunk) + ",\n")
            f.write("};\n\n")
        
        # Array of pointers to test images
        f.write("const int8_t* test_images[NUM_TEST_IMAGES] = {\n")
        for i in range(num_test_images):
            f.write(f"    test_image_{i},\n")
        f.write("};\n\n")
        
        # Expected labels
        f.write("const int expected_labels[NUM_TEST_IMAGES] = {\n")
        f.write("    " + ", ".join(str(l) for l in labels) + "\n")
        f.write("};\n\n")
        
        f.write("#endif // TEST_IMAGES_H\n")
    
    print(f"\nExported weights to {out_dir}/weights.h")
    print(f"Exported {num_test_images} test images to {out_dir}/test_images.h")

# =============================================================
# Step 6: Verify quantization accuracy in Python
# =============================================================
def load_hidden_test_set():
    """Load the MNIST test set — only called AFTER training is complete."""
    transform = transforms.ToTensor()
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
    print(f"Loaded hidden test set: {len(test_data)} images")
    return test_data

def verify_quantized_accuracy(model, quantized, test_data, num_test=1000):
    """
    Run inference using quantized int8 weights in pure numpy
    to verify accuracy before porting to RISC-V.
    """
    w1 = quantized['fc1.weight']['data'].astype(np.int32)  # (128, 784)
    b1 = quantized['fc1.bias']['data'].astype(np.int32)     # (128,)
    w2 = quantized['fc2.weight']['data'].astype(np.int32)  # (10, 128)
    b2 = quantized['fc2.bias']['data'].astype(np.int32)     # (10,)
    
    s_w1 = quantized['fc1.weight']['scale']
    s_b1 = quantized['fc1.bias']['scale']
    s_w2 = quantized['fc2.weight']['scale']
    s_b2 = quantized['fc2.bias']['scale']
    
    correct = 0
    for i in range(num_test):
        image, label = test_data[i]
        x = quantize_input(image).astype(np.int32)  # (784,)
        
        # Layer 1: result has scale = s_w1 * 127 (input scale)
        # We use int32 accumulation to avoid overflow
        h = w1 @ x + (b1 * 127 / s_b1 * s_w1).astype(np.int32)
        
        # ReLU
        h = np.maximum(h, 0)
        
        # Rescale hidden layer to int8 range for next layer
        h_max = np.max(np.abs(h)) if np.max(np.abs(h)) > 0 else 1
        h_scale = 127.0 / h_max
        h = np.round(h * h_scale).astype(np.int32)
        
        # Layer 2
        out = w2 @ h + (b2 * h_scale * 127 / s_b2 * s_w2).astype(np.int32)
        
        pred = np.argmax(out)
        if pred == label:
            correct += 1
    
    acc = 100.0 * correct / num_test
    print(f"\nQuantized accuracy (Python, {num_test} hidden test samples): {acc:.1f}%")
    return acc

# =============================================================
# Main
# =============================================================
if __name__ == '__main__':
    print("=" * 50)
    print("RISC-V ML Inference — Training & Export")
    print("=" * 50)
    
    # Train (uses only train + validation splits, test set is hidden)
    model = train()
    
    # Quantize
    print("\n--- Quantizing weights ---")
    quantized = quantize_weights(model)
    
    # NOW load the hidden test set for the first time
    print("\n--- Loading hidden test set ---")
    test_data = load_hidden_test_set()
    
    # Verify quantized accuracy on hidden test data
    print("\n--- Verifying quantized accuracy on hidden test set ---")
    verify_quantized_accuracy(model, quantized, test_data)
    
    # Export hidden test images to C
    print("\n--- Exporting to C ---")
    export_to_c(quantized, test_data, num_test_images=100)
    
    print("\nDone! Next step: compile and run runtime/inference.c on RISC-V")