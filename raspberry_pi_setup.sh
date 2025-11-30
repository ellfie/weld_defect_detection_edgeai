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

# Download script
echo ""
echo "Deployment script should be copied to ~/weld_detection/"
echo "Model file should be placed in ~/weld_detection/models/"

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
echo "Next steps:"
echo "1. Copy raspberry_pi_deploy.py to ~/weld_detection/"
echo "2. Copy your TFLite model to ~/weld_detection/models/model_int8.tflite"
echo "3. Activate environment: source ~/weld_detection_env/bin/activate"
echo "4. Run detection: python raspberry_pi_deploy.py --mode camera"
echo ""
echo "Usage examples:"
echo "  # Camera detection:"
echo "  python raspberry_pi_deploy.py --mode camera"
echo ""
echo "  # Image detection:"
echo "  python raspberry_pi_deploy.py --mode image --image test.jpg --output result.jpg"
echo ""
echo "  # Benchmark:"
echo "  python raspberry_pi_deploy.py --mode benchmark"
echo ""
echo "  # With Edge TPU:"
echo "  python raspberry_pi_deploy.py --mode camera --edgetpu"
echo ""
echo "A reboot is recommended to apply all changes."
read -p "Reboot now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo reboot
fi
