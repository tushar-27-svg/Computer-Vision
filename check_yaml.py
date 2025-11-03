import yaml

path = r"C:\Users\ts255\Downloads\CVIP\CVIP2\data.yaml"

try:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    print("✅ YAML loaded successfully:")
    print(data)
except Exception as e:
    print("❌ Error reading YAML:", e)
