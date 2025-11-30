#!/bin/bash

# Raspberry Pi Setup Script for Weld Defect Detection
# This script installs all dependencies and sets up the deployment environment

set -e

echo "========================================"
echo "Raspberry Pi Setup for Weld Detection"
echo "========================================"

# Detect Raspberry Pi version
if [ -f /proc/device-tree/model ]; then
    RPI_MODEL=$(cat /proc/device-tree/model)
    echo "Detected: $RPI_MODEL"
else
    echo "Warning: Could not detect Raspberry Pi model"
fi

# Update system
echo ""
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo ""
echo "Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-opencv \
    libopencv-dev \
    libatlas-base-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    libqtgui4 \
    libqtwebkit4 \
    libqt4-test \
    libjasper-dev \
    libilmbase25 \
    libopenexr25 \
    libgstreamer1.0-0 \
    libavcodec58 \
    libavformat58 \
    libswscale5 \
    libwebp6 \
    libtiff5 \
    libopenjp2-7 \
    cmake

# Create virtual environment
echo ""
echo "Creating Python virtual environment..."
python3 -m venv ~/weld_detection_env
source ~/weld_detection_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install TFLite Runtime (lightweight, no full TensorFlow)
echo ""
echo "Installing TensorFlow Lite Runtime..."

# Detect Python version and architecture
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
ARCH=$(uname -m)

echo "Python version: $PYTHON_VERSION"
echo "Architecture: $ARCH"

# Install tflite-runtime
if [ "$ARCH" = "armv7l" ] || [ "$ARCH" = "aarch64" ]; then
    # For Raspberry Pi
    pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_armv7l.whl || \
    pip install tflite-runtime
else
    # For other platforms
    pip install tflite-runtime
fi

# Install other Python dependencies
echo ""
echo "Installing Python packages..."
pip install numpy opencv-python-headless

# Optional: Install Coral Edge TPU support
echo ""
read -p "Do you want to install Coral Edge TPU support? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing Edge TPU runtime..."
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    sudo apt-get update
    sudo apt-get install -y libedgetpu1-std
    pip install pycoral
    echo "✓ Edge TPU support installed"
fi

# Test camera (optional)
echo ""
read -p "Do you want to test the camera? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Testing camera..."
    python3 << EOF
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("✓ Camera is working!")
    ret, frame = cap.read()
    if ret:
        print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
    cap.release()
else:
    print("✗ Camera not detected")
EOF
fi

# Enable camera interface
echo ""
echo "Enabling camera interface..."
sudo raspi-config nonint do_camera 0

# Increase swap space for better performance (optional)
echo ""
read -p "Do you want to increase swap space to 2GB? (recommended) (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Increasing swap space..."
    sudo dphys-swapfile swapoff
    sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
    sudo dphys-swapfile setup
    sudo dphys-swapfile swapon
    echo "✓ Swap space increased to 2GB"
fi

# Create directories
echo ""
echo "Creating directories..."
mkdir -p ~/weld_detection
mkdir -p ~/weld_detection/models
mkdir -p ~/weld_detection/test_images
mkdir -p ~/weld_detection/outputs
mkdir -p ~/weld_detection/data

# Download script
echo ""
echo "Deployment script should be copied to ~/weld_detection/"
echo "Model file should be placed in ~/weld_detection/models/"

# Copy test images from data folder
echo ""
read -p "Do you want to copy test images from data/ folder? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -d "data" ]; then
        echo "Copying test images from data/ folder..."
        
        # Copy sample images from each subfolder to test_images
        for folder in data/*/; do
            if [ -d "$folder" ]; then
                folder_name=$(basename "$folder")
                echo "  Copying samples from $folder_name..."
                
                # Copy first 3 images from each folder
                count=0
                for img in "$folder"*.{jpg,jpeg,png,JPG,JPEG,PNG} 2>/dev/null; do
                    if [ -f "$img" ] && [ $count -lt 3 ]; then
                        cp "$img" ~/weld_detection/test_images/
                        count=$((count + 1))
                    fi
                done
            fi
        done
        
        # Count copied images
        num_images=$(ls ~/weld_detection/test_images/*.{jpg,jpeg,png,JPG,JPEG,PNG} 2>/dev/null | wc -l)
        echo "✓ Copied $num_images test images to ~/weld_detection/test_images/"
        
        # Copy the entire data folder for reference
        echo "Copying data folder structure..."
        cp -r data ~/weld_detection/
        echo "✓ Data folder copied to ~/weld_detection/data/"
    else
        echo "✗ data/ folder not found in current directory"
        echo "  Make sure to copy test images manually to ~/weld_detection/test_images/"
    fi
fi

# Test with sample image if available
echo ""
read -p "Do you want to run a test inference on a sample image? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Check if deployment script and model exist
    if [ -f "src/raspberry_pi_deploy.py" ] && [ -f "models/model_int8.tflite" ]; then
        echo "Running test inference..."
        
        # Copy deployment script
        cp src/raspberry_pi_deploy.py ~/weld_detection/
        
        # Copy model
        cp models/model_int8.tflite ~/weld_detection/models/
        
        # Find a test image
        test_image=$(ls ~/weld_detection/test_images/*.{jpg,jpeg,png,JPG,JPEG,PNG} 2>/dev/null | head -n 1)
        
        if [ -f "$test_image" ]; then
            echo "Testing with image: $test_image"
            cd ~/weld_detection
            python raspberry_pi_deploy.py \
                --mode image \
                --image "$test_image" \
                --model models/model_int8.tflite \
                --quantized
            
            echo ""
            echo "✓ Test inference completed!"
            echo "  Check output in ~/weld_detection/outputs/"
        else
            echo "✗ No test image found in ~/weld_detection/test_images/"
        fi
    else
        echo "✗ Deployment script or model not found"
        echo "  Please copy them manually and run:"
        echo "  python raspberry_pi_deploy.py --mode image --image test.jpg --model models/model_int8.tflite --quantized"
    fi
fi

# Create systemd service (optional)
echo ""
read -p "Do you want to create a systemd service for auto-start? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cat << EOF | sudo tee /etc/systemd/system/weld-detection.service
[Unit]
Description=Weld Defect Detection Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/weld_detection
ExecStart=$HOME/weld_detection_env/bin/python raspberry_pi_deploy.py --mode camera
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    echo "✓ Service created"
    echo "  Enable with: sudo systemctl enable weld-detection"
    echo "  Start with:  sudo systemctl start weld-detection"
    echo "  Status:      sudo systemctl status weld-detection"
fi

# Performance tuning
echo ""
echo "Applying performance optimizations..."

# Set GPU memory split
sudo bash -c 'echo "gpu_mem=256" >> /boot/config.txt'

# Overclock (safe settings for Pi 4)
if [[ "$RPI_MODEL" == *"Raspberry Pi 4"* ]]; then
    read -p "Do you want to apply safe overclock settings? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo bash -c 'cat >> /boot/config.txt << EOF

# Overclock settings
over_voltage=2
arm_freq=1750
gpu_freq=600
EOF'
        echo "✓ Overclock settings applied (requires reboot)"
    fi
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Directory structure created:"
echo "  ~/weld_detection/"
echo "  ~/weld_detection/models/          - TFLite models"
echo "  ~/weld_detection/test_images/     - Test images for inference"
echo "  ~/weld_detection/outputs/         - Detection results"
echo "  ~/weld_detection/data/            - Full dataset (if copied)"
echo ""
echo "Next steps:"
echo "1. Copy raspberry_pi_deploy.py to ~/weld_detection/ (if not already done)"
echo "2. Copy your TFLite model to ~/weld_detection/models/model_int8.tflite (if not already done)"
echo "3. Activate environment: source ~/weld_detection_env/bin/activate"
echo "4. Run detection on test images:"
echo ""
echo "Usage examples:"
echo "  # Test with sample image:"
echo "  cd ~/weld_detection"
echo "  python raspberry_pi_deploy.py --mode image --image test_images/sample.jpg --model models/model_int8.tflite --quantized"
echo ""
echo "  # Batch test all images in test_images/:"
echo "  for img in test_images/*.jpg; do"
echo "    python raspberry_pi_deploy.py --mode image --image \"\$img\" --model models/model_int8.tflite --quantized"
echo "  done"
echo ""
echo "  # Camera detection:"
echo "  python raspberry_pi_deploy.py --mode camera --model models/model_int8.tflite --quantized"
echo ""
echo "  # Benchmark:"
echo "  python raspberry_pi_deploy.py --mode benchmark --model models/model_int8.tflite --quantized"
echo ""
echo "  # With Edge TPU:"
echo "  python raspberry_pi_deploy.py --mode camera --model models/model_int8_edgetpu.tflite --quantized --edgetpu"
echo ""
echo "A reboot is recommended to apply all changes."
read -p "Reboot now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo reboot
fi
