
import streamlit as st
import cv2
import numpy as np
import pickle
import os
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
    
    # Load the trained model
    with open("model_path.pkl", 'rb') as model_file:
        model = pickle.load(model_file)

    input_size = (32,32)
    
    # Setup required functions
    def preprocess_frame(frame):
        frame = cv2.resize(frame, input_size)
        frame = frame / 255.0
        return frame.reshape((1,) + frame.shape)
    
    def extract_frames(video_file_path, save_path):
        cap = cv2.VideoCapture(video_file_path)
        frameno = 0
        frame_paths = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess frame and make prediction
            processed_frame = preprocess_frame(frame)
            prediction = model.predict(processed_frame)
            
            # Interpret the prediction
            if prediction[0][0] > 0.5:
                result = 'Wag Tail'
            
            else:
                result = 'Sit Down'
            
            # Save frame with prediction label
            name = os.path.join(save_path, f"{frameno}_{result}.jpg")
            cv2.imwrite(name, frame)
            frame_paths.append(name)
            frameno += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        return frame_paths
    
    def main():
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
            frame_paths = extract_frames(temp_video.name, './user_video')
            
            #Display predictions
            st.header("Your pet's behaviour analysed:")
            for idx, frame_path in enumerate(frame_paths):
                st.image(frame_path, caption=f"Frame {idx + 1}")
    
    if __name__ == "__main__":
        main()
            
            
            
    
    














       
