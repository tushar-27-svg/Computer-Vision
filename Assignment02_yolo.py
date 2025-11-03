import os
from ultralytics import YOLO
import yaml
from natsort import natsorted
import cv2

def find_any_model(base_dir):
    """Find any .pt model file"""
    search_paths = [
        os.path.join(base_dir, "runs", "train"),
        os.path.join(base_dir, "runs", "detect"),
    ]
    
    found_models = []
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
        for root, dirs, files in os.walk(search_path):
            for file in files:
                if file in ["best.pt", "last.pt"]:
                    found_models.append(os.path.join(root, file))
    return found_models


def infer_with_model(image_path, model_choice="pretrained", custom_model_path=None, conf=0.25):
    """Run inference with either pre-trained or custom model"""
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print("\n" + "="*60)
    
    # Select model
    if model_choice == "pretrained":
        print("📥 Using Pre-trained YOLOv8 (COCO dataset - 80 classes)")
        model = YOLO("yolov8n.pt")
    else:
        if not custom_model_path or not os.path.exists(custom_model_path):
            print("❌ Custom model not found!")
            return
        print(f"📥 Using Custom Model: {custom_model_path}")
        model = YOLO(custom_model_path)
    
    print(f"🖼️  Image: {image_path}")
    print(f"📊 Confidence: {conf}")
    print(f"🏷️  Classes: {list(model.names.values())}")
    print("="*60)
    
    # Run detection
    print("\n🔍 Running detection...")
    results = model(image_path, save=True, conf=conf, verbose=False)
    
    # Parse results
    print("\n" + "="*60)
    print("DETECTION RESULTS")
    print("="*60)
    
    for r in results:
        if len(r.boxes) == 0:
            print("⚠️  No objects detected!")
            print(f"\n💡 Try:")
            print(f"   - Lower confidence (current: {conf})")
            print(f"   - Use pre-trained model if detecting common objects")
            print(f"   - Check if model is trained on this image type")
        else:
            print(f"✅ Detected {len(r.boxes)} object(s):\n")
            
            for i, box in enumerate(r.boxes, 1):
                cls = int(box.cls[0])
                conf_val = float(box.conf[0])
                class_name = model.names[cls]
                coords = box.xyxy[0].tolist()
                
                print(f"   {i}. {class_name}")
                print(f"      Confidence: {conf_val:.3f} ({conf_val*100:.1f}%)")
                print(f"      Box: [{coords[0]:.0f}, {coords[1]:.0f}, {coords[2]:.0f}, {coords[3]:.0f}]")
                print()
    
    print("="*60)
    
    # Show result
    pred_folders = [d for d in os.listdir("runs/detect") if d.startswith("predict")]
    if pred_folders:
        latest = natsorted(pred_folders)[-1]
        pred_path = os.path.join("runs/detect", latest)
        
        for file in os.listdir(pred_path):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(pred_path, file)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # Resize for display
                    h, w = img.shape[:2]
                    if h > 900:
                        ratio = 900 / h
                        img = cv2.resize(img, (int(w * ratio), 900))
                    
                    cv2.imshow("YOLOv8 Detection", img)
                    print(f"\n📸 Press any key to close...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                break


def train_yolo(data_yaml, epochs=10, base_dir=None):
    """Train YOLOv8"""
    if not os.path.exists(data_yaml):
        print(f"❌ data.yaml not found: {data_yaml}")
        return
    
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"📊 Classes: {data.get('nc')} - {data.get('names')}")
    print(f"📁 Train: {data.get('train')}")
    print(f"📁 Val: {data.get('val')}")
    print(f"⏱️  Epochs: {epochs}")
    print("="*60)
    
    proceed = input("\n🚀 Start training? (y/n): ").lower()
    if proceed != 'y':
        return
    
    model = YOLO("yolov8n.pt")
    
    project_dir = os.path.join(base_dir, "runs", "train") if base_dir else "runs/train"
    
    print(f"\n📥 Training started...")
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        project=project_dir,
        name="yolo_custom",
        patience=50,
        save=True,
        plots=True
    )
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    base_dir = r"C:\Users\ts255\Downloads\CVIP\CVIP2"
    
    print("""
    ══════════════════════════════════════
            YOLOv8 DETECTION
    ══════════════════════════════════════
    1️⃣  Train Custom Model
    2️⃣  Detect with Pre-trained Model (COCO)
    3️⃣  Detect with Custom Model
    4️⃣  Detect with Custom Confidence
    ══════════════════════════════════════
    """)
    
    choice = input("Choice (1-4): ").strip()
    
    if choice == '1':
        # Train
        yaml_path = os.path.join(base_dir, "data.yaml")
        epochs = input("Epochs (default 10): ").strip()
        epochs = int(epochs) if epochs.isdigit() else 10
        train_yolo(yaml_path, epochs, base_dir)
    
    elif choice == '2':
        # Pre-trained detection
        img = input("Image path (Enter for default): ").strip()
        if not img:
            img = os.path.join(base_dir, "dataset", "images", "val", "img6.jpg")
        
        conf = input("Confidence (default 0.25): ").strip()
        conf = float(conf) if conf else 0.25
        
        infer_with_model(img, model_choice="pretrained", conf=conf)
    
    elif choice == '3':
        # Custom model detection
        models = find_any_model(base_dir)
        
        if not models:
            print("❌ No custom models found! Train one first (option 1)")
        else:
            print("\nAvailable models:")
            for i, m in enumerate(models, 1):
                print(f"{i}. {os.path.relpath(m, base_dir)}")
            
            idx = input(f"\nSelect (1-{len(models)}): ").strip()
            idx = int(idx) - 1 if idx.isdigit() else 0
            
            img = input("Image path (Enter for default): ").strip()
            if not img:
                img = os.path.join(base_dir, "dataset", "images", "val", "img6.jpg")
            
            infer_with_model(img, model_choice="custom", custom_model_path=models[idx])
    
    elif choice == '4':
        # Custom confidence
        img = input("Image path (Enter for default): ").strip()
        if not img:
            img = os.path.join(base_dir, "dataset", "images", "val", "img6.jpg")
        
        conf = input("Confidence (0.01-1.0): ").strip()
        conf = float(conf) if conf else 0.25
        
        use_pretrained = input("Use pre-trained model? (y/n): ").lower() == 'y'
        
        if use_pretrained:
            infer_with_model(img, model_choice="pretrained", conf=conf)
        else:
            models = find_any_model(base_dir)
            if models:
                infer_with_model(img, model_choice="custom", custom_model_path=models[0], conf=conf)
            else:
                print("No custom models found, using pre-trained")
                infer_with_model(img, model_choice="pretrained", conf=conf)