import os
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
import torchvision.ops as ops

# ===============================
# CONFIGURATION
# ===============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# COCO Dataset Classes (80 classes) - includes hand-held objects
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Hand-held / portable objects (lower confidence threshold for these)
HANDHELD_OBJECTS = {
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'cell phone', 'remote',
    'book', 'scissors', 'teddy bear', 'handbag', 'backpack', 'umbrella',
    'tie', 'suitcase', 'frisbee', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'tennis racket', 'mouse', 'keyboard',
    'clock', 'vase', 'hair drier', 'toothbrush'
}

# Color palette
COLORS = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype=np.uint8)
# Make handheld objects more visible (brighter colors)
for idx, cls in enumerate(COCO_CLASSES):
    if cls in HANDHELD_OBJECTS:
        COLORS[idx] = np.random.randint(150, 255, size=3, dtype=np.uint8)


# ===============================
# LOAD OPTIMIZED MODEL
# ===============================
def load_model():
    """Load pre-trained SSD300 model optimized for all objects"""
    print("\n📥 Loading optimized SSD300 model...")
    
    try:
        model = ssd300_vgg16(weights=SSD300_VGG16_Weights.COCO_V1)
    except:
        model = models.detection.ssd300_vgg16(pretrained=True)
    
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")
    print(f"📊 Detecting {len(COCO_CLASSES)-1} object classes")
    print(f"🖐️  Optimized for hand-held objects: {len(HANDHELD_OBJECTS)} classes")
    
    return model


# ===============================
# ENHANCED DETECTION WITH NMS
# ===============================
def detect_objects_optimized(model, image_path, conf_threshold=0.5, 
                             handheld_conf=0.25, iou_threshold=0.45,
                             enable_multi_scale=True):
    """
    Enhanced object detection with optimizations for small/hand-held objects
    
    Args:
        model: Pre-trained SSD model
        image_path: Path to input image
        conf_threshold: Confidence threshold for regular objects
        handheld_conf: Lower confidence threshold for hand-held objects
        iou_threshold: IoU threshold for NMS
        enable_multi_scale: Enable multi-scale detection
    
    Returns:
        boxes, labels, scores
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image not found: {image_path}")
    
    print(f"\n🔍 Detecting objects in: {os.path.basename(image_path)}")
    print(f"📊 Confidence: {conf_threshold} (regular) / {handheld_conf} (hand-held)")
    
    # Load image
    img_pil = Image.open(image_path).convert("RGB")
    original_size = img_pil.size
    
    all_boxes = []
    all_labels = []
    all_scores = []
    
    # Multi-scale detection for better small object detection
    if enable_multi_scale:
        scales = [1.0, 1.2, 0.8]  # Original, zoom in, zoom out
        print("🔍 Using multi-scale detection for better accuracy...")
    else:
        scales = [1.0]
    
    for scale in scales:
        # Resize image
        new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
        img_scaled = img_pil.resize(new_size, Image.BILINEAR)
        
        # Preprocess
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(img_scaled).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            predictions = model(img_tensor)
        
        # Extract predictions
        pred = predictions[0]
        boxes = pred['boxes'].cpu()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        
        # Scale boxes back to original size
        boxes[:, [0, 2]] *= (original_size[0] / new_size[0])
        boxes[:, [1, 3]] *= (original_size[1] / new_size[1])
        
        # Apply adaptive confidence threshold
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else ''
            
            # Use lower threshold for hand-held objects
            threshold = handheld_conf if class_name in HANDHELD_OBJECTS else conf_threshold
            
            if score >= threshold:
                all_boxes.append(box.numpy())
                all_labels.append(label)
                all_scores.append(score)
    
    if len(all_boxes) == 0:
        print("⚠️  No objects detected")
        return np.array([]), np.array([]), np.array([])
    
    # Convert to tensors
    all_boxes = torch.tensor(np.array(all_boxes))
    all_labels = torch.tensor(np.array(all_labels))
    all_scores = torch.tensor(np.array(all_scores))
    
    # Apply Non-Maximum Suppression (NMS) to remove duplicates
    keep_indices = ops.nms(all_boxes, all_scores, iou_threshold)
    
    final_boxes = all_boxes[keep_indices].numpy()
    final_labels = all_labels[keep_indices].numpy()
    final_scores = all_scores[keep_indices].numpy()
    
    print(f"✅ Found {len(final_boxes)} unique objects")
    
    # Count by category
    regular_count = sum(1 for l in final_labels if COCO_CLASSES[l] not in HANDHELD_OBJECTS)
    handheld_count = sum(1 for l in final_labels if COCO_CLASSES[l] in HANDHELD_OBJECTS)
    print(f"   📦 Regular objects: {regular_count}")
    print(f"   🖐️  Hand-held objects: {handheld_count}")
    
    return final_boxes, final_labels, final_scores


# ===============================
# ENHANCED VISUALIZATION
# ===============================
def visualize_detections_enhanced(image_path, boxes, labels, scores, 
                                 output_path=None, show=True, 
                                 highlight_handheld=True):
    """
    Enhanced visualization with special highlighting for hand-held objects
    
    Args:
        image_path: Path to input image
        boxes: Detection bounding boxes
        labels: Class labels
        scores: Confidence scores
        output_path: Path to save annotated image
        show: Whether to display image
        highlight_handheld: Highlight hand-held objects with thicker borders
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return None
    
    height, width = img.shape[:2]
    
    print("\n" + "="*80)
    print("DETECTION RESULTS (OPTIMIZED FOR ALL OBJECTS)")
    print("="*80)
    
    # Group detections by type
    regular_objs = []
    handheld_objs = []
    
    for idx, (box, label, score) in enumerate(zip(boxes, labels, scores), 1):
        x1, y1, x2, y2 = map(int, box)
        
        # Ensure coordinates are within bounds
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        
        # Get class info
        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f'class_{label}'
        is_handheld = class_name in HANDHELD_OBJECTS
        color = tuple(map(int, COLORS[label % len(COLORS)]))
        
        # Draw bounding box (thicker for hand-held objects)
        thickness = 3 if (is_handheld and highlight_handheld) else 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label
        label_text = f"{class_name}: {score:.2f}"
        if is_handheld:
            label_text = f"🖐️ {label_text}"
        
        # Calculate text size
        font_scale = 0.6
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text.replace('🖐️ ', ''),  # Remove emoji for size calc
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            2
        )
        
        # Draw background for text
        padding = 5
        cv2.rectangle(
            img,
            (x1, y1 - text_height - baseline - padding*2),
            (x1 + text_width + padding*2, y1),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            img,
            label_text.replace('🖐️ ', ''),
            (x1 + padding, y1 - padding - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            2
        )
        
        # Add icon indicator for hand-held objects
        if is_handheld and highlight_handheld:
            cv2.circle(img, (x1 + 10, y1 + 10), 5, color, -1)
        
        # Categorize for display
        obj_info = f"{class_name:25s} | Conf: {score:.3f} ({score*100:.1f}%) | Box: [{x1:4d}, {y1:4d}, {x2:4d}, {y2:4d}]"
        if is_handheld:
            handheld_objs.append((idx, obj_info))
        else:
            regular_objs.append((idx, obj_info))
    
    # Print categorized results
    if regular_objs:
        print("\n📦 REGULAR OBJECTS:")
        for idx, info in regular_objs:
            print(f"  {idx:2d}. {info}")
    
    if handheld_objs:
        print("\n🖐️  HAND-HELD / PORTABLE OBJECTS:")
        for idx, info in handheld_objs:
            print(f"  {idx:2d}. {info}")
    
    print("="*80)
    
    # Add summary overlay on image
    summary_y = 30
    cv2.putText(img, f"Total: {len(boxes)} objects", (10, summary_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if handheld_objs:
        cv2.putText(img, f"Hand-held: {len(handheld_objs)}", (10, summary_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
    
    # Save result
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"\n💾 Saved result to: {output_path}")
    
    # Display
    if show:
        display_img = img.copy()
        max_display_height = 900
        if height > max_display_height:
            scale = max_display_height / height
            new_width = int(width * scale)
            display_img = cv2.resize(display_img, (new_width, max_display_height))
        
        cv2.imshow("SSD Object Detection (Optimized)", display_img)
        print("\n📸 Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return img


# ===============================
# BATCH DETECTION (OPTIMIZED)
# ===============================
def detect_in_folder_optimized(model, folder_path, conf_threshold=0.5, 
                               handheld_conf=0.25, output_folder="detections"):
    """Optimized batch detection for folder"""
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return
    
    os.makedirs(output_folder, exist_ok=True)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f.lower())[1] in image_extensions
    ]
    
    if not image_files:
        print(f"❌ No images found in {folder_path}")
        return
    
    print(f"\n📁 Processing {len(image_files)} images from {folder_path}")
    print("="*80)
    
    total_detections = 0
    total_handheld = 0
    
    for idx, img_file in enumerate(image_files, 1):
        img_path = os.path.join(folder_path, img_file)
        output_path = os.path.join(output_folder, f"detected_{img_file}")
        
        print(f"\n[{idx}/{len(image_files)}] Processing: {img_file}")
        
        try:
            boxes, labels, scores = detect_objects_optimized(
                model, img_path, conf_threshold, handheld_conf
            )
            
            total_detections += len(boxes)
            handheld_count = sum(1 for l in labels if COCO_CLASSES[l] in HANDHELD_OBJECTS)
            total_handheld += handheld_count
            
            if len(boxes) > 0:
                visualize_detections_enhanced(
                    img_path, boxes, labels, scores, output_path, show=False
                )
            else:
                print("⚠️  No objects detected")
        
        except Exception as e:
            print(f"❌ Error processing {img_file}: {e}")
    
    print("\n" + "="*80)
    print(f"✅ Batch processing complete!")
    print(f"📊 Total detections: {total_detections}")
    print(f"🖐️  Hand-held objects: {total_handheld}")
    print(f"📁 Results saved to: {output_folder}")
    print("="*80)


# ===============================
# WEBCAM WITH OPTIMIZED DETECTION
# ===============================
def detect_from_webcam_optimized(model, conf_threshold=0.5, handheld_conf=0.25):
    """Real-time webcam detection optimized for all objects"""
    print("\n📹 Starting optimized webcam detection...")
    print("Press 'q' to quit, 's' to save screenshot")
    print("🖐️  Optimized to detect hand-held objects")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    transform = transforms.Compose([transforms.ToTensor()])
    screenshot_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break
        
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            predictions = model(img_tensor)
        
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        
        # Adaptive filtering
        kept_boxes = []
        kept_labels = []
        kept_scores = []
        
        for box, label, score in zip(boxes, labels, scores):
            class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else ''
            threshold = handheld_conf if class_name in HANDHELD_OBJECTS else conf_threshold
            
            if score >= threshold:
                kept_boxes.append(box)
                kept_labels.append(label)
                kept_scores.append(score)
        
        # Draw detections
        handheld_count = 0
        for box, label, score in zip(kept_boxes, kept_labels, kept_scores):
            x1, y1, x2, y2 = map(int, box)
            class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f'class_{label}'
            is_handheld = class_name in HANDHELD_OBJECTS
            color = tuple(map(int, COLORS[label % len(COLORS)]))
            
            thickness = 3 if is_handheld else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            label_text = f"{class_name}: {score:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if is_handheld:
                handheld_count += 1
                cv2.circle(frame, (x1 + 10, y1 + 10), 5, color, -1)
        
        # Display stats
        cv2.putText(frame, f"Objects: {len(kept_boxes)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Hand-held: {handheld_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
        
        cv2.imshow('SSD Webcam Detection (Optimized)', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_count += 1
            filename = f"webcam_detection_{screenshot_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Screenshot saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Webcam detection stopped")


# ===============================
# MAIN MENU
# ===============================
def main():
    print("""
    ══════════════════════════════════════════════════════════════
            SSD OBJECT DETECTION (OPTIMIZED)
    ══════════════════════════════════════════════════════════════
    ✨ Multi-scale detection for better accuracy
    🖐️  Optimized for hand-held and small objects
    📦 Detects 80 object classes including portable items
    🎯 Adaptive confidence thresholds per object type
    ══════════════════════════════════════════════════════════════
    """)
    
    model = load_model()
    
    while True:
        print("""
    ══════════════════════════════════════════════════════════════
    1️⃣  Detect objects in single image (optimized)
    2️⃣  Detect objects in folder (batch, optimized)
    3️⃣  Real-time webcam detection (optimized)
    4️⃣  Show hand-held object classes
    5️⃣  Show all available classes
    6️⃣  Exit
    ══════════════════════════════════════════════════════════════
        """)
        
        choice = input("Enter choice (1-6): ").strip()
        
        if choice == '1':
            image_path = input("\nEnter image path: ").strip()
            if not image_path:
                print("❌ No path provided")
                continue
            
            conf = input("Regular objects confidence (default 0.5): ").strip()
            conf = float(conf) if conf else 0.5
            
            hand_conf = input("Hand-held objects confidence (default 0.25): ").strip()
            hand_conf = float(hand_conf) if hand_conf else 0.25
            
            multi_scale = input("Enable multi-scale detection? (y/n, default y): ").strip().lower()
            multi_scale = multi_scale != 'n'
            
            try:
                boxes, labels, scores = detect_objects_optimized(
                    model, image_path, conf, hand_conf, enable_multi_scale=multi_scale
                )
                
                if len(boxes) > 0:
                    output_path = "detection_result_optimized.jpg"
                    visualize_detections_enhanced(
                        image_path, boxes, labels, scores, output_path, show=True
                    )
                else:
                    print("⚠️  No objects detected. Try lowering confidence thresholds.")
            
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
        
        elif choice == '2':
            folder_path = input("\nEnter folder path: ").strip()
            if not folder_path:
                print("❌ No path provided")
                continue
            
            conf = input("Regular objects confidence (default 0.5): ").strip()
            conf = float(conf) if conf else 0.5
            
            hand_conf = input("Hand-held objects confidence (default 0.25): ").strip()
            hand_conf = float(hand_conf) if hand_conf else 0.25
            
            output_folder = input("Output folder (default 'detections'): ").strip()
            output_folder = output_folder if output_folder else "detections"
            
            try:
                detect_in_folder_optimized(model, folder_path, conf, hand_conf, output_folder)
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == '3':
            conf = input("Regular objects confidence (default 0.5): ").strip()
            conf = float(conf) if conf else 0.5
            
            hand_conf = input("Hand-held objects confidence (default 0.25): ").strip()
            hand_conf = float(hand_conf) if hand_conf else 0.25
            
            try:
                detect_from_webcam_optimized(model, conf, hand_conf)
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == '4':
            print("\n" + "="*80)
            print("HAND-HELD / PORTABLE OBJECT CLASSES")
            print("="*80)
            sorted_handheld = sorted(HANDHELD_OBJECTS)
            for idx, cls in enumerate(sorted_handheld, 1):
                print(f"{idx:2d}. {cls}")
            print(f"\nTotal: {len(HANDHELD_OBJECTS)} hand-held object classes")
            print("="*80)
        
        elif choice == '5':
            print("\n" + "="*80)
            print("ALL AVAILABLE OBJECT CLASSES (80 total)")
            print("="*80)
            for idx, cls in enumerate(COCO_CLASSES[1:], 1):
                if cls != 'N/A':
                    indicator = "🖐️ " if cls in HANDHELD_OBJECTS else "   "
                    print(f"{indicator}{idx:2d}. {cls}")
            print("\n🖐️  = Hand-held/portable object")
            print("="*80)
        
        elif choice == '6':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice!")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()