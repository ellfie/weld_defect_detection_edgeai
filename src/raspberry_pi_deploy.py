"""
Weld Defect Detection - Raspberry Pi Deployment
Runs TFLite INT8 quantized model on Raspberry Pi for real-time inference
"""

import numpy as np
import cv2
import json
import time
from pathlib import Path
import argparse

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# Configuration
MODEL_PATH = "models/model_int8.tflite"
IMAGE_SIZE = (128, 128)
CONFIDENCE_THRESHOLD = 0.5

# Load category names
CATEGORIES = [
    "concavity",
    "convcavity", 
    "excessive reinforcement",
    "misalignment",
    "spatter"
]


class WeldDefectDetector:
    """Weld defect detector using TFLite model"""
    
    def __init__(self, model_path, image_size=IMAGE_SIZE, use_edgetpu=False):
        """Initialize the detector
        
        Args:
            model_path: Path to TFLite model
            image_size: Input image size (width, height)
            use_edgetpu: Use Coral Edge TPU if available
        """
        self.image_size = image_size
        self.use_edgetpu = use_edgetpu
        
        # Load TFLite model
        print(f"Loading model from {model_path}...")
        
        if use_edgetpu:
            try:
                from tflite_runtime.interpreter import Interpreter, load_delegate
                self.interpreter = Interpreter(
                    model_path=str(model_path),
                    experimental_delegates=[load_delegate('libedgetpu.so.1')]
                )
                print("✓ Using Coral Edge TPU acceleration")
            except Exception as e:
                print(f"Warning: Could not load Edge TPU: {e}")
                print("Falling back to CPU...")
                self.interpreter = tflite.Interpreter(model_path=str(model_path))
        else:
            self.interpreter = tflite.Interpreter(model_path=str(model_path))
            print("✓ Using CPU inference")
        
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Input dtype: {self.input_details[0]['dtype']}")
        print(f"Number of outputs: {len(self.output_details)}")
        
        # Check if model expects uint8 (quantized)
        self.is_quantized = self.input_details[0]['dtype'] == np.uint8
        
        if self.is_quantized:
            print("✓ Model is quantized (INT8)")
        else:
            print("✓ Model is floating point")
    
    def preprocess_image(self, image):
        """Preprocess image for inference
        
        Args:
            image: Input image (BGR format from cv2)
            
        Returns:
            Preprocessed image ready for inference
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image_resized = cv2.resize(image_rgb, self.image_size)
        
        # Normalize and convert to expected format
        if self.is_quantized:
            # For INT8 quantized models, input should be uint8 [0, 255]
            input_data = image_resized.astype(np.uint8)
        else:
            # For FP32 models, normalize to [0, 1]
            input_data = image_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        input_data = np.expand_dims(input_data, axis=0)
        
        return input_data
    
    def detect(self, image):
        """Run detection on image
        
        Args:
            image: Input image (BGR format from cv2)
            
        Returns:
            Dictionary with detection results:
            - bbox: [cx, cy, w, h] normalized coordinates
            - class_id: Predicted class ID
            - class_name: Predicted class name
            - class_confidence: Classification confidence
            - detection_confidence: Overall detection confidence
        """
        # Preprocess
        input_data = self.preprocess_image(image)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        start_time = time.time()
        self.interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Get outputs
        bbox = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        class_probs = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        confidence = self.interpreter.get_tensor(self.output_details[2]['index'])[0][0]
        
        # Dequantize if needed
        if self.is_quantized:
            # Dequantize outputs
            bbox = self.dequantize_output(bbox, self.output_details[0])
            class_probs = self.dequantize_output(class_probs, self.output_details[1])
            confidence = self.dequantize_output(confidence, self.output_details[2])
        
        # Get predicted class
        class_id = np.argmax(class_probs)
        class_confidence = class_probs[class_id]
        class_name = CATEGORIES[class_id] if class_id < len(CATEGORIES) else f"Class {class_id}"
        
        return {
            'bbox': bbox.tolist(),
            'class_id': int(class_id),
            'class_name': class_name,
            'class_confidence': float(class_confidence),
            'detection_confidence': float(confidence),
            'inference_time_ms': inference_time
        }
    
    def dequantize_output(self, quantized_output, output_details):
        """Dequantize INT8 output to float32"""
        scale = output_details['quantization_parameters']['scales'][0]
        zero_point = output_details['quantization_parameters']['zero_points'][0]
        return (quantized_output.astype(np.float32) - zero_point) * scale
    
    def draw_detection(self, image, detection, show_text=True):
        """Draw detection on image
        
        Args:
            image: Input image (BGR format)
            detection: Detection dictionary from detect()
            show_text: Whether to show text labels
            
        Returns:
            Image with detection drawn
        """
        img_h, img_w = image.shape[:2]
        
        # Get bbox in pixel coordinates
        cx, cy, w, h = detection['bbox']
        x1 = int((cx - w/2) * img_w)
        y1 = int((cy - h/2) * img_h)
        x2 = int((cx + w/2) * img_w)
        y2 = int((cy + h/2) * img_h)
        
        # Determine color based on confidence
        conf = detection['detection_confidence']
        if conf > 0.8:
            color = (0, 255, 0)  # Green - high confidence
        elif conf > 0.5:
            color = (0, 255, 255)  # Yellow - medium confidence
        else:
            color = (0, 0, 255)  # Red - low confidence
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        if show_text:
            # Prepare text
            label = f"{detection['class_name']}: {detection['class_confidence']:.2f}"
            det_conf = f"Det: {detection['detection_confidence']:.2f}"
            
            # Draw background for text
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - text_h - 25), (x1 + max(text_w, 100), y1), color, -1)
            
            # Draw text
            cv2.putText(image, label, (x1 + 2, y1 - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image, det_conf, (x1 + 2, y1 - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return image


def detect_from_image(detector, image_path, output_path=None, show=False):
    """Run detection on a single image
    
    Args:
        detector: WeldDefectDetector instance
        image_path: Path to input image
        output_path: Path to save output (optional)
        show: Whether to display image
    """
    print(f"\nProcessing: {image_path}")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Run detection
    detection = detector.detect(image)
    
    # Print results
    print(f"  Class: {detection['class_name']}")
    print(f"  Classification confidence: {detection['class_confidence']:.3f}")
    print(f"  Detection confidence: {detection['detection_confidence']:.3f}")
    print(f"  Inference time: {detection['inference_time_ms']:.2f} ms")
    print(f"  FPS: {1000/detection['inference_time_ms']:.1f}")
    
    # Draw detection
    output_image = detector.draw_detection(image.copy(), detection)
    
    # Save if output path provided
    if output_path:
        cv2.imwrite(str(output_path), output_image)
        print(f"  Saved to: {output_path}")
    
    # Show if requested
    if show:
        cv2.imshow('Detection', output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def detect_from_camera(detector, camera_id=0, save_video=None):
    """Run detection on camera stream
    
    Args:
        detector: WeldDefectDetector instance
        camera_id: Camera device ID
        save_video: Path to save output video (optional)
    """
    print(f"\nOpening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Camera: {width}x{height} @ {fps} FPS")
    
    # Video writer
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(save_video), fourcc, fps, (width, height))
        print(f"Recording to: {save_video}")
    
    print("\nStarting detection... Press 'q' to quit")
    
    # FPS calculation
    frame_count = 0
    start_time = time.time()
    inference_times = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            detection = detector.detect(frame)
            inference_times.append(detection['inference_time_ms'])
            
            # Draw detection
            output_frame = detector.draw_detection(frame.copy(), detection)
            
            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed > 0:
                display_fps = frame_count / elapsed
            else:
                display_fps = 0
            
            # Draw FPS on frame
            fps_text = f"FPS: {display_fps:.1f} | Inference: {detection['inference_time_ms']:.1f}ms"
            cv2.putText(output_frame, fps_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Weld Defect Detection', output_frame)
            
            # Save frame
            if writer:
                writer.write(output_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        print("\n" + "="*60)
        print("STATISTICS")
        print("="*60)
        print(f"Total frames: {frame_count}")
        print(f"Average FPS: {frame_count / elapsed:.1f}")
        print(f"Average inference time: {np.mean(inference_times):.2f} ms")
        print(f"Min/Max inference time: {np.min(inference_times):.2f} / {np.max(inference_times):.2f} ms")
        print("="*60)


def benchmark_model(detector, num_runs=100):
    """Benchmark model performance
    
    Args:
        detector: WeldDefectDetector instance
        num_runs: Number of inference runs
    """
    print(f"\nBenchmarking model ({num_runs} runs)...")
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Warm-up
    for _ in range(10):
        _ = detector.detect(dummy_image)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        result = detector.detect(dummy_image)
        times.append(result['inference_time_ms'])
    
    # Statistics
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Number of runs: {num_runs}")
    print(f"Average: {np.mean(times):.2f} ms (±{np.std(times):.2f} ms)")
    print(f"Min/Max: {np.min(times):.2f} / {np.max(times):.2f} ms")
    print(f"FPS: {1000/np.mean(times):.1f}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Weld Defect Detection on Raspberry Pi")
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                       help='Path to TFLite model')
    parser.add_argument('--mode', type=str, default='camera',
                       choices=['image', 'camera', 'benchmark'],
                       help='Detection mode')
    parser.add_argument('--image', type=str, help='Path to input image (for image mode)')
    parser.add_argument('--output', type=str, help='Path to output image/video')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--edgetpu', action='store_true', help='Use Coral Edge TPU')
    parser.add_argument('--benchmark', type=int, default=100,
                       help='Number of benchmark runs')
    
    args = parser.parse_args()
    
    # Initialize detector
    print("="*60)
    print("WELD DEFECT DETECTION - RASPBERRY PI")
    print("="*60)
    
    detector = WeldDefectDetector(
        model_path=args.model,
        use_edgetpu=args.edgetpu
    )
    
    # Run selected mode
    if args.mode == 'image':
        if not args.image:
            print("Error: --image required for image mode")
            return
        detect_from_image(detector, args.image, args.output, show=True)
    
    elif args.mode == 'camera':
        detect_from_camera(detector, args.camera, args.output)
    
    elif args.mode == 'benchmark':
        benchmark_model(detector, args.benchmark)
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
