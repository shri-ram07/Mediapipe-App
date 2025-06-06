import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import tempfile
from PIL import ImageColor, Image
import requests
from io import BytesIO

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Streamlit page configuration
st.set_page_config(page_title="Video Keypoint Editor", layout="wide", page_icon="🎥")

# Custom CSS for modern UI
st.markdown("""
<style>
body {
    background-color: #f0f2f6;
}
.stApp {
    max-width: 1200px;
    margin: 0 auto;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #45a049;
}
.sidebar .sidebar-content {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
h1 {
    color: #2c3e50;
    text-align: center;
    font-family: 'Arial', sans-serif;
}
.stSlider > div > div > div > div {
    background-color: #4CAF50;
}
</style>
""", unsafe_allow_html=True)

# App title
st.title("🎥 Video Keypoint Editor with MediaPipe")

# Sidebar for customization options
st.sidebar.header("Customization Options")

# Keypoint selection
keypoint_options = {
    "Nose": 0, "Left Eye": 2, "Right Eye": 5, "Left Shoulder": 11, "Right Shoulder": 12,
    "Left Elbow": 13, "Right Elbow": 14, "Left Wrist": 15, "Right Wrist": 16,
    "Left Hip": 23, "Right Hip": 24, "Left Knee": 25, "Right Knee": 26,
    "Left Ankle": 27, "Right Ankle": 28
}
selected_keypoints = st.sidebar.multiselect(
    "Select Keypoints to Display", list(keypoint_options.keys()), default=list(keypoint_options.keys())
)

# Line color selection
line_color = st.sidebar.color_picker("Pick Line Color", "#FF0000")
line_color_rgb = ImageColor.getcolor(line_color, "RGB")
line_thickness = st.sidebar.slider("Line Thickness", 1, 10, 2)

# Keypoint color selection
keypoint_color = st.sidebar.color_picker("Pick Keypoint Color", "#00FF00")
keypoint_color_rgb = ImageColor.getcolor(keypoint_color, "RGB")
keypoint_size = st.sidebar.slider("Keypoint Size", 1, 20, 5)

# Demo image preview
st.subheader("Demo Preview (Keypoint Customization)")
image_url = "https://chatgpt.com/backend-api/public_content/enc/eyJpZCI6Im1fNjg0MDUxMjQxMDBjODE5MWJhNjI0ODRkNDllZGIzZDQ6ZmlsZV8wMDAwMDAwMDA4NDA2MWY3OWEzYzE0NGQ1MzQzNzNmOCIsInRzIjoiNDg1ODQ1IiwicCI6InB5aSIsInNpZyI6IjdmNWY5MTU1ODU1MWJkODIxZTA2ZTc3ZjgwYjExMDY1NjMzOTVhMTUwN2VhMTZmNzE5NWY4MWNjNTcyNjNlYWMiLCJ2IjoiMCIsImdpem1vX2lkIjpudWxsfQ=="
response = requests.get(image_url)
img = Image.open(BytesIO(response.content))
img_array = np.array(img)

# Process demo image with MediaPipe
demo_pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
demo_frame_rgb = img_array.copy()
demo_results = demo_pose.process(demo_frame_rgb)

# Prepare demo output image
demo_annotated_frame = demo_frame_rgb.copy()
if demo_results.pose_landmarks:
    # Custom drawing specifications
    custom_landmark_style = mp_drawing.DrawingSpec(color=keypoint_color_rgb, thickness=keypoint_size, circle_radius=keypoint_size)
    custom_connection_style = mp_drawing.DrawingSpec(color=line_color_rgb, thickness=line_thickness)

    # Filter connections based on selected keypoints
    selected_indices = [keypoint_options[kp] for kp in selected_keypoints]
    connections = [
        (start, end) for start, end in mp_pose.POSE_CONNECTIONS
        if start in selected_indices and end in selected_indices
    ]

    # Draw keypoints and connections
    mp_drawing.draw_landmarks(
        demo_annotated_frame,
        demo_results.pose_landmarks,
        connections=connections,
        landmark_drawing_spec=custom_landmark_style,
        connection_drawing_spec=custom_connection_style
    )

# Display demo preview
st.image(demo_annotated_frame, channels="RGB", use_container_width=True, caption="Demo Image with Customized Keypoints")
demo_pose.close()

# Video upload
st.subheader("Upload Video to Apply Keypoints")
uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded video to temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()

    # Display original video
    st.subheader("Original Video")
    st.video(tfile.name)

    # Process video without live preview
    st.subheader("Processing Video...")
    cap = cv2.VideoCapture(tfile.name)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare output video
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = output_file.name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Custom drawing specifications
    custom_landmark_style = mp_drawing.DrawingSpec(color=keypoint_color_rgb, thickness=keypoint_size, circle_radius=keypoint_size)
    custom_connection_style = mp_drawing.DrawingSpec(color=line_color_rgb, thickness=line_thickness)

    # Filter connections based on selected keypoints
    selected_indices = [keypoint_options[kp] for kp in selected_keypoints]
    connections = [
        (start, end) for start, end in mp_pose.POSE_CONNECTIONS
        if start in selected_indices and end in selected_indices
    ]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Create a blank image for drawing
            annotated_frame = frame.copy()
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                connections=connections,
                landmark_drawing_spec=custom_landmark_style,
                connection_drawing_spec=custom_connection_style
            )

            # Convert back to BGR for OpenCV
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            out.write(annotated_frame)

    cap.release()
    out.release()

    # Provide download button
    st.subheader("Download Processed Video")
    with open(output_path, "rb") as file:
        video_bytes = file.read()
        st.download_button(
            label="Download Video",
            data=video_bytes,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )

    # Clean up temporary files
    os.unlink(tfile.name)
    os.unlink(output_path)

else:
    st.info("Please upload a video to apply the customized keypoints.")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7f8c8d;'>Powered by Streamlit & MediaPipe</p>", unsafe_allow_html=True)