import cv2
import streamlit as st
import numpy as np
from PIL import Image

# Load the pre-trained face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 100), 2)
    return frame

def apply_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def apply_hsv(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

def apply_smoothing(frame):
    return cv2.GaussianBlur(frame, (15, 15), 0)

def apply_edge_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 100, 200)

def apply_cartoon(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(frame, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def apply_noise_reduction(frame):
    return cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

# Set up Streamlit app
st.set_page_config(page_title="Image Processing App", page_icon=":camera:", layout="wide")

# Sidebar for navigation and controls
st.sidebar.title("Image Processing App")

# Initialize session state
if 'run_camera' not in st.session_state:
    st.session_state.run_camera = False
if 'effect' not in st.session_state:
    st.session_state.effect = 'None'
if 'face_detection' not in st.session_state:
    st.session_state.face_detection = False

# Camera control
camera_control = st.sidebar.empty()
if not st.session_state.run_camera:
    if camera_control.button('Start Camera'):
        st.session_state.run_camera = True
else:
    if camera_control.button('Stop Camera'):
        st.session_state.run_camera = False

# Effect selection
effect = st.sidebar.selectbox(
    'Select Effect',
    ('None', 'Gray', 'HSV', 'Smoothing', 'Edge Detection', 'Cartoon', 'Noise Reduction'),
    key='effect'
)

# Face detection toggle
face_detection = st.sidebar.checkbox('Enable Face Detection', key='face_detection')

# Main area for displaying video
st.title("Image Processing App")
st.write("Use the sidebar to control the camera, apply effects, and toggle face detection.")

# Placeholder for video stream
stframe = st.empty()

if st.session_state.run_camera:
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not open webcam")
    else:
        while st.session_state.run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Failed to capture video frame")
                break

            # Apply selected image processing effect
            if effect == 'Gray':
                frame = apply_gray(frame)
            elif effect == 'HSV':
                frame = apply_hsv(frame)
            elif effect == 'Smoothing':
                frame = apply_smoothing(frame)
            elif effect == 'Edge Detection':
                frame = apply_edge_detection(frame)
            elif effect == 'Cartoon':
                frame = apply_cartoon(frame)
            elif effect == 'Noise Reduction':
                frame = apply_noise_reduction(frame)
            
            # Apply face detection if enabled
            if face_detection:
                frame = detect_faces(frame)

            # Convert to RGB for display
            if effect != 'HSV':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the processed frame
            stframe.image(frame, channels="RGB", use_column_width=True)

        # Release the webcam when the loop ends
        cap.release()

else:
    st.write("Camera is stopped. Click 'Start Camera' to begin.")

# Add a file uploader for processing static images
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image file
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Convert RGB to BGR (OpenCV uses BGR)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Apply selected effect
    if effect == 'Gray':
        img_array = apply_gray(img_array)
    elif effect == 'HSV':
        img_array = apply_hsv(img_array)
    elif effect == 'Smoothing':
        img_array = apply_smoothing(img_array)
    elif effect == 'Edge Detection':
        img_array = apply_edge_detection(img_array)
    elif effect == 'Cartoon':
        img_array = apply_cartoon(img_array)
    elif effect == 'Noise Reduction':
        img_array = apply_noise_reduction(img_array)

    # Apply face detection if enabled
    if face_detection:
        img_array = detect_faces(img_array)

    # Convert back to RGB for display
    if effect != 'HSV':
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    # Display the processed image
    st.image(img_array, caption='Processed Image', use_column_width=True)