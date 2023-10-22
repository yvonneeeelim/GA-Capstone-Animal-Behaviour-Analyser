
import streamlit as st
import os
os.environ['CV2_CUDNN_STREAM'] = '1'
import cv2
import numpy as np
import pickle
from tempfile import NamedTemporaryFile

import tensorflow as tf
from tensorflow.keras.models import load_model


# opening the image
st.image('https://cf.ltkcdn.net/life-with-pets/fun-with-pets…std-sm/331988-423x285-gettyimages-1128501986.webp', use_column_width=True)

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

    # Setup function to convert video to frames
    def process_video(video_file):
        frames = []
        cam = cv2.VideoCapture(video_file)
        frameno = 0

        while True:
            ret, frame = cam.read()
            if ret:
                # if video is still left continue creating images
                name = str(frameno) + '.jpg'

                cv2.imwrite(name, frame)
                frameno += 1
            else:
                break

        cam.release()
        cv2.destroyAllWindows()

        # Load the trained model
        with open("cnn_model.pkl", 'rb') as model_file:
            model = pickle.load(model_file)

        # Process frame and make prediction
        predictions = model.predict(np.array(frames))

        # Interpret the prediction
        behaviour_results = []
        for prediction in predictions:
            if prediction > 0.5:
                behaviour_results.append('Sit Down')
            else:
                behaviour_results.append('Wag Tail')
        return behaviour_results
                    
    

    st.title('Animal Behaviour Analyser')
    st.write('Upload a video of your pet to understand its behaviour')
        
    # Get user input for video upload
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])
        
    # Process the uploaded video if it exists
    if uploaded_file is not None:
        # Save the uploaded video to a temporary file
        with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_file.read())
            
        # Extract frames and make predictions
        behaviour_results = process_video(temp_video.name)
            
        # Display predictions
        st.header("Your pet's behaviour analysed:")
        for idx, behaviour in enumerate(behaviour_results):
            st.image(f"Frame {idx + 1}: {behaviour}")
    
            
            
            
    
    














       
