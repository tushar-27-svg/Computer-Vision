import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import torch
from torchvision import models, transforms

st.set_page_config(page_title="🧠 Image Segmentation Studio", layout="wide", page_icon="🧩")

st.title("🧠 Image Segmentation Studio")
st.markdown("### Explore multiple segmentation techniques with a clean, interactive interface.")

st.sidebar.header("⚙️ Controls")
technique = st.sidebar.selectbox(
    "Choose Segmentation Technique",
    ["Semantic Segmentation", "Threshold Segmentation", "Edge Detection", "Watershed Transform"],
)

st.sidebar.markdown("---")
st.sidebar.info("Upload an image, choose a method, tune parameters, and hit **Run Segmentation**.")

uploaded_file = st.sidebar.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
    with col2:
        st.markdown("### 🎚️ Parameter Tuning")

        if technique == "Threshold Segmentation":
            thresh = st.slider("Threshold Value", 0, 255, 127)
        elif technique == "Edge Detection":
            low = st.slider("Canny Low Threshold", 0, 255, 50)
            high = st.slider("Canny High Threshold", 0, 255, 150)
        elif technique == "Watershed Transform":
            blur = st.slider("Gaussian Blur Kernel Size", 1, 15, 5, step=2)

    st.markdown("---")
    run = st.button("🚀 Run Segmentation", use_container_width=True)

    if run:
        np_img = np.array(image.convert("RGB"))

        if technique == "Threshold Segmentation":
            gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
            _, result = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

        elif technique == "Edge Detection":
            gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
            result = cv2.Canny(gray, low, high)

        elif technique == "Watershed Transform":
            gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
            blur_img = cv2.GaussianBlur(gray, (blur, blur), 0)
            ret, thresh_img = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((3,3), np.uint8)
            sure_bg = cv2.dilate(thresh_img, kernel, iterations=3)
            dist_transform = cv2.distanceTransform(thresh_img, cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            ret, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            result = cv2.watershed(np_img, markers)
            np_img[result == -1] = [255, 0, 0]
            result = np_img

        elif technique == "Semantic Segmentation":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = models.segmentation.deeplabv3_resnet50(pretrained=True).to(device).eval()
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            if np_img.shape[2] == 4:
                np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)

            input_tensor = transform(np_img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)["out"][0]
            output_predictions = output.argmax(0).byte().cpu().numpy()
            colored_mask = cv2.applyColorMap((output_predictions * 10).astype(np.uint8), cv2.COLORMAP_JET)
            result = cv2.addWeighted(np_img, 0.6, colored_mask, 0.4, 0)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="🖼️ Original Image", use_container_width=True)
        with col2:
            st.image(result, caption=f"🎯 {technique} Result", use_container_width=True)

        st.markdown("---")
        result_pil = Image.fromarray(result.astype(np.uint8))
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        st.download_button(
            label="💾 Download Result",
            data=buf.getvalue(),
            file_name=f"{technique.replace(' ', '_').lower()}_result.png",
            mime="image/png",
        )

else:
    st.info("👈 Please upload an image to begin segmentation.")