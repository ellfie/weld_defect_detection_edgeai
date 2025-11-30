# Weld Defect Detection System

Team: Sudharshan and Rohit 
Application Track

A complete end-to-end machine learning pipeline for detecting weld defects using computer vision. We created a dataset for thermal defect detection. we created our own welding samples and recorded the images using raspberry pi and FLIR Lepton.
It's explained well in technical report.

## Overview

This system detects 5 types of weld defects:
- **Concavity**: Insufficient weld material causing depression
- **Convexity**: Excessive weld material causing protrusion
- **Excessive Reinforcement**: Too much weld material deposited
- **Misalignment**: Improper alignment of welded parts
- **Spatter**: Metal particles deposited around weld area

## Project Structure

get the weld-cpp-mcu-v3 from [drive](https://drive.google.com/file/d/1J3hv7t25zKeHH64AVqrNvAQnffeWxQjt/view?usp=sharing)

```
weld/
├── src/
│   └── raspberry_pi_deploy.py        # Edge deployment script
├── weld-cpp-mcu-v3/                  # Edge Impulse C++ library for MCU
│   ├── edge-impulse-sdk/             # Inference SDK
│   ├── tflite-model/                 # Trained TFLite model
│   ├── model-parameters/             # Model configuration
│   └── CMakeLists.txt                # Build configuration
├── raspberry_pi_setup.sh             # Automated Raspberry Pi setup
├── data.zip                          # Dataset for training
```



### Raspberry Pi / Edge Devices
- **Raspberry Pi 4** (2GB+ RAM recommended, 4GB for best performance)
- Raspberry Pi OS (64-bit recommended)
- FLIR Lepton 3.5 thermal camera
- **MCU Deployment** (using Edge Impulse C++ library):
  - ESP32 / STM32 / Arduino boards
  
## Installation

### 1. Clone Repository


### 2. Setup Python Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# .venv\Scripts\activate  # On Windows

pip install --upgrade pip
pip install tensorflow opencv-python albumentations pillow numpy pandas
```

## Usage

### Step 1: Train the Model on Edge Impulse

1. **Extract and upload dataset:**
   ```bash
   unzip data.zip
   ```

2. **Upload to Edge Impulse Studio:**
   - Go to https://studio.edgeimpulse.com/
   - Create a new project or open existing project (ID: 839037)
   - Upload images from extracted `data/` folder

3. **Train with YOLO:**
   - Use **YOLO Pro** learning block
   - Recommended: **Nano or Medium model** for MCU deployment
   - Train for at least **20-30 epochs**
   -  We got 85.4% Precision Score

4. **Download C++ library:**
   - After training, go to **Deployment** tab
   - Select **C++ library**
   - Click **Build** to generate `weld-cpp-mcu-v3.zip`
   - Extract to project root (already included as `weld-cpp-mcu-v3/`)

### Step 2: Deploy on MCU (Arduino)

Using the Edge Impulse C++ library for microcontroller deployment:

#### Option A: Arduino IDE

1. **Add library to Arduino:**
   ```bash
   # Copy the library to Arduino libraries folder
   cp -r weld-cpp-mcu-v3 ~/Arduino/libraries/weld_inferencing
   ```

2. **Create Arduino sketch:**
   ```cpp
   #include <weld_inferencing.h>
   
   // Camera setup (e.g., ESP32-CAM)
   void setup() {
       Serial.begin(115200);
       // Initialize camera
       // Initialize FLIR Lepton if using thermal imaging
   }
   
   void loop() {
       ei_impulse_result_t result;
       signal_t signal;
       
       // Capture image and preprocess
       // ...
       
       // Run inference
       EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false);
       
       if (res == EI_IMPULSE_OK) {
           // Process detections
           for (size_t i = 0; i < result.bounding_boxes_count; i++) {
               ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];
               if (bb.value > 0.5) {  // Confidence threshold
                   Serial.printf("%s (%.2f): x=%d y=%d w=%d h=%d\n",
                       bb.label, bb.value, bb.x, bb.y, bb.width, bb.height);
               }
           }
       }
       delay(100);
   }
   ```

3. **Compile and upload:**
   - Select your board (ESP32, STM32, etc.)
   - Compile and upload to device

### Step 3: Deploy on Raspberry Pi (Python/TFLite)

#### Quick Setup (Automated)

Transfer files to Raspberry Pi:
```bash
# From your development machine
scp -r models/ raspberry_pi_setup.sh src/ pi@raspberrypi.local:~/weld_detection/
```

SSH into Raspberry Pi and run setup:
```bash
ssh pi@raspberrypi.local
cd ~/weld_detection
chmod +x raspberry_pi_setup.sh
./raspberry_pi_setup.sh
```

## Edge Impulse C++ Library API

Key functions from `weld-cpp-mcu-v3/` library:

### Run Inference
```cpp
#include <weld_inferencing.h>

// Initialize classifier
ei_impulse_result_t result = { 0 };
signal_t signal;
signal.total_length = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT;
signal.get_data = &get_signal_data;

// Run classifier
EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false);

// Check results
if (res == EI_IMPULSE_OK) {
    ei_printf("Predictions (DSP: %d ms., Classification: %d ms.)\n",
        result.timing.dsp, result.timing.classification);
    
    // Object detection results
    for (size_t ix = 0; ix < result.bounding_boxes_count; ix++) {
        auto bb = result.bounding_boxes[ix];
        if (bb.value == 0) continue;
        
        ei_printf("  %s (%.5f) [ x: %u, y: %u, width: %u, height: %u ]\n",
            bb.label, bb.value, bb.x, bb.y, bb.width, bb.height);
    }
}
```
