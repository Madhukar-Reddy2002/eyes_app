import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from io import BytesIO
import cv2

def apply_gray(image):
    return ImageOps.grayscale(image)

def apply_sepia(image):
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_img = np.array(image).dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return Image.fromarray(np.uint8(sepia_img * 255))

def apply_blur(image):
    return image.filter(ImageFilter.BLUR)

def apply_edge_detection(image):
    return image.filter(ImageFilter.FIND_EDGES)

def apply_emboss(image):
    return image.filter(ImageFilter.EMBOSS)

def apply_sharpen(image):
    return image.filter(ImageFilter.SHARPEN)

def apply_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def apply_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def process_image(image, effect, brightness, contrast):
    processed_image = image.copy()

    if effect == 'Gray':
        processed_image = apply_gray(processed_image)
    elif effect == 'Sepia':
        processed_image = apply_sepia(processed_image)
    elif effect == 'Blur':
        processed_image = apply_blur(processed_image)
    elif effect == 'Edge Detection':
        processed_image = apply_edge_detection(processed_image)
    elif effect == 'Emboss':
        processed_image = apply_emboss(processed_image)
    elif effect == 'Sharpen':
        processed_image = apply_sharpen(processed_image)

    processed_image = apply_brightness(processed_image, brightness)
    processed_image = apply_contrast(processed_image, contrast)

    return processed_image

# Set up Streamlit app
st.set_page_config(page_title="Image Processing App", page_icon=":camera:", layout="wide")

# Sidebar for controls
st.sidebar.title("Image Processing App")

# Input method selection
input_method = st.sidebar.radio("Select Input Method", ("Upload Image", "Use Webcam"))

# Effect selection
effect = st.sidebar.selectbox(
    'Select Effect',
    ('None', 'Gray', 'Sepia', 'Blur', 'Edge Detection', 'Emboss', 'Sharpen')
)

# Brightness and Contrast sliders
brightness = st.sidebar.slider('Brightness', 0.0, 2.0, 1.0, 0.1)
contrast = st.sidebar.slider('Contrast', 0.0, 2.0, 1.0, 0.1)

# Main area for displaying images
st.title("Image Processing App")

if input_method == "Upload Image":
    st.write("Upload an image and apply various effects using the sidebar controls.")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        processed_image = process_image(image, effect, brightness, contrast)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        with col2:
            st.subheader("Processed Image")
            st.image(processed_image, use_column_width=True)

        buffered = BytesIO()
        processed_image.save(buffered, format="PNG")
        st.download_button(
            label="Download processed image",
            data=buffered.getvalue(),
            file_name="processed_image.png",
            mime="image/png"
        )
    else:
        st.write("Please upload an image to begin processing.")

else:  # Use Webcam
    st.write("Capture an image from your webcam and apply effects in real-time.")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not open webcam. Please make sure your webcam is connected and not being used by another application.")
    else:
        # Create a button to capture image
        if st.button("Capture Image"):
            ret, frame = cap.read()
            if ret:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                processed_image = process_image(image, effect, brightness, contrast)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Captured Image")
                    st.image(image, use_column_width=True)
                with col2:
                    st.subheader("Processed Image")
                    st.image(processed_image, use_column_width=True)

                buffered = BytesIO()
                processed_image.save(buffered, format="PNG")
                st.download_button(
                    label="Download processed image",
                    data=buffered.getvalue(),
                    file_name="processed_image.png",
                    mime="image/png"
                )
            else:
                st.error("Error: Failed to capture image from webcam.")

        # Release the webcam when the app is closed
        cap.release()

# Add some information about the app
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info("This app allows you to apply various image processing effects to your uploaded images or webcam captures. "
                "Select an effect from the dropdown menu and adjust brightness and contrast as needed.")
st.sidebar.markdown("Created by [Your Name/Organization]")
st.sidebar.markdown("[Source Code](https://github.com/yourusername/your-repo)")