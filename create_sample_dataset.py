import os
from PIL import Image, ImageDraw
import random
from pathlib import Path
import yaml

# Root dataset directory
root = Path("./dataset")

# YOLO structure
dirs = [
    root / "images/train",
    root / "images/val",
    root / "labels/train",
    root / "labels/val",
    root / "JPEGImages",
    root / "Annotations"
]

for d in dirs:
    d.mkdir(parents=True, exist_ok=True)

# Label map
classes = ["person", "car"]

def create_image(idx, split):
    """Creates a random image with rectangles simulating person/car."""
    img = Image.new("RGB", (640, 480), color=(random.randint(150,255),)*3)
    draw = ImageDraw.Draw(img)

    objs = []
    for c in classes:
        x1, y1 = random.randint(50, 200), random.randint(50, 200)
        x2, y2 = x1 + random.randint(100, 200), y1 + random.randint(100, 200)
        draw.rectangle([x1, y1, x2, y2], outline="red" if c == "person" else "blue", width=3)
        objs.append((c, x1, y1, x2, y2))

    # Save image
    img_path = root / "images" / split / f"img{idx}.jpg"
    img.save(img_path)

    # Save YOLO label
    label_path = root / "labels" / split / f"img{idx}.txt"
    with open(label_path, "w") as f:
        for obj in objs:
            cls = classes.index(obj[0])
            x_center = ((obj[1] + obj[3]) / 2) / 640
            y_center = ((obj[2] + obj[4]) / 2) / 480
            w = (obj[3] - obj[1]) / 640
            h = (obj[4] - obj[2]) / 480
            f.write(f"{cls} {x_center:.4f} {y_center:.4f} {w:.4f} {h:.4f}\n")

    # Save VOC annotation (for SSD)
    xml_path = root / "Annotations" / f"img{idx}.xml"
    with open(xml_path, "w") as f:
        f.write(f"""<annotation>
    <filename>img{idx}.jpg</filename>
    {"".join([
        f"<object><name>{obj[0]}</name><bndbox><xmin>{obj[1]}</xmin><ymin>{obj[2]}</ymin><xmax>{obj[3]}</xmax><ymax>{obj[4]}</ymax></bndbox></object>"
        for obj in objs
    ])}
</annotation>""")

    # Copy for JPEGImages (SSD format)
    img.save(root / "JPEGImages" / f"img{idx}.jpg")

# Generate 3 train + 2 val images
for i in range(1, 4):
    create_image(i, "train")
for i in range(4, 6):
    create_image(i, "val")

# Save data.yaml
yaml_content = {
    "train": "./dataset/images/train",
    "val": "./dataset/images/val",
    "nc": len(classes),
    "names": classes
}
with open("data.yaml", "w") as f:
    yaml.dump(yaml_content)

print("✅ Sample dataset generated successfully!")
print("Structure:")
for d in dirs:
    print(" -", d)
