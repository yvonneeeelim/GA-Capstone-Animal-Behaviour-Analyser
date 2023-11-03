
import streamlit as st
import os
os.environ['CV2_CUDNN_STREAM'] = '1'
import cv2
import numpy as np
import pickle
import tempfile
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model


# opening the image
image = open('banner_image.jpeg', 'rb').read()
st.image(image, use_column_width=True)

st.title("Furgorithm - Know Your Pet")
st.markdown("Understanding the behavior of your pet is essential for responsible pet ownership. It ensures a fulfilling and mutually beneficial relationship between you and your animal companion.")

# Add a separator between the header and the main content
st.markdown("---")

# Upload tabs
selected_tab = st.radio("Choose from below options:", ["Upload your Pet Video", "Upload your Pet Image", "Chat with Us"])

if selected_tab == "Upload your Pet Video":
    st.header("Upload your Pet Video")

    # Load the trained model
    model = load_model("efficientnet_model.pkl")

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


    #st.title('Animal Behaviour Analyser')
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




elif selected_tab == "Upload your Pet Image":
    st.header("Upload your Pet Image")
   
    # Load the trained model
    model = load_model("cnn_model.h5")

    # Get user input for image upload
    uploaded_file = st.file_uploader('Upload an image of your pet to understand its behaviour', type=['jpg', 'jpeg', 'png'])

    # Process the uploaded image if it exists
    if uploaded_file is not None:
        # Open and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Load new images for prediction
        new_image = image
        new_image_array = np.array(new_image.resize((32, 32)))  # Resize the image to match the model's input shape
        new_image_array = np.expand_dims(new_image_array, axis=0)  # Add batch dimension
        new_image_array = new_image_array / 255.0  # Normalize the pixel values (same as during training)

        # Make predictions
        predictions = model.predict(new_image_array)

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions, axis=1)[0]

        # Indicate class names
        class_names = ['Angry', 'Curious', 'Happy', 'Relaxed', 'Sad']

        # Get the predicted class name
        predicted_class_name = class_names[predicted_class_index]

        st.write(f"Predicted Behaviour: {predicted_class_name}")


else:
    st.header("Chat with Us")


       
