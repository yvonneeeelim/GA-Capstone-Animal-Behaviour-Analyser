
import streamlit as st
import os
os.environ['CV2_CUDNN_STREAM'] = '1'
import cv2
import numpy as np
import pickle
import tempfile

import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase


# opening the image
#image = open('banner_image.jpeg', 'rb').read()
#st.image(image, use_column_width=True)

st.divider()


# Custom CSS to style the title and subheader
st.markdown(
    """
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 10px;
    }
    .subheader {
        font-size: 18px;
        font-style: italic;
        color: #777;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and subheader with custom styles
st.markdown('<p class="title">Animal Signal</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Know your pet! Uuncover its mood, and decipher its needs. </p>', unsafe_allow_html=True)


st.divider()

st.markdown("**Choose from below options:**")
tab1, tab2, tab3, tab4 = st.tabs(["Upload your Pet Video", "Upload an image", "Search Keywords", "Find Healthy Snack"])
# Add a short liner above the tabs

with tab1:
    st.header("Upload your Pet Video")

    # Load the trained model
    model = tf.keras.models.load_model('cnn_model.keras')

    def preprocess_frames(frames):
        processed_frames = []
        for frame in frames:
            frame = frame.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
            processed_frames.append(frame)
        return np.array(processed_frames)

    def annotate_video(input_video_path, output_video_path, model):
        frames = []
        cap = cv2.VideoCapture(input_video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

        class_labels = ['Happy', 'Sad', 'Relaxed', 'Curious', 'Angry']

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame to match the model's input shape (32x32)
            resized_frame = cv2.resize(frame, (32, 32))
            frames.append(resized_frame)

            # Make prediction
            processed_frame = preprocess_frames(np.array(frames))
            predictions = model.predict(processed_frame)
            predicted_class_index = np.argmax(predictions)
            behaviour = class_labels[predicted_class_index]

            # Add text annotation to the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (255, 0, 0)  # Red color for annotation
            cv2.putText(frame, behaviour, (50, 50), font, font_scale, color, 2, cv2.LINE_AA)
            out.write(frame)

            frames.pop(0)  # Remove the oldest frame from the list

        # Close the temporary file
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        return output_video_path


    st.title('Animal Behaviour Analyser')
    st.write('Upload a video of your pet to understand its behaviour')
            
    # Get user input for video upload
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])
            
    # Process the uploaded video if it exists
    if uploaded_file is not None:
        st.write("Analyzing video...")
        temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(temp_video_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

            output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            annotated_video_path = annotate_video(temp_video_path, output_video_path, model)

            with open(annotated_video_path, "rb") as video_file:
                video_bytes = video_file.read()
                st.write(annotated_video_path)
                #st.video(video_bytes)
                
                
                # Create an auto-download button
                st.download_button(
                    label="Download File",
                    data=video_bytes,
                    file_name='Annotated Video.mp4',
                )
                
        # Remove the temporary files (only after all the previous code has completed running)
        try:
            os.remove(temp_video_path)
            os.remove(output_video_path)
        finally:
            pass





       
