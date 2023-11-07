
import streamlit as st
import os
os.environ['CV2_CUDNN_STREAM'] = '1'
import cv2
import numpy as np
import pickle
import tempfile
from PIL import Image
from efficientnet.tfkeras import EfficientNetB4
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img


# opening the image
image = open('banner_image.jpeg', 'rb').read()
st.image(image, use_column_width=True)

st.title("Furgorithm - Know Your Pet")
st.markdown("Understanding the behavior of your pet is essential for responsible pet ownership. It ensures a fulfilling and mutually beneficial relationship between you and your animal companion.")

# Add a separator between the header and the main content
st.markdown("---")

# Upload tabs
tab1, tab2, tab3 = st.tabs(["Upload your Pet Video", "Upload your Pet Image", "Understand your Pet Emotions"])

with tab1:
    st.header("Upload your Pet Video")
    
    class_labels = ['Relaxed', 'Sad']
    
    # Load the trained model
    model = load_model('efficientnet2.h5')

    def preprocess_frames(frames):
        processed_frames = []
        for frame in frames:
            frame = frame.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
            processed_frames.append(frame)
        return np.array(processed_frames)

    def annotate_video(input_video_path, output_video_path, model, video_records=None):
        frames = []
        total_frames = 0
        relaxed_count = 0
        sad_count = 0
        cap = cv2.VideoCapture(input_video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame to match the model's input shape (384x384)
            resized_frame = cv2.resize(frame, (384, 384))
            frames.append(resized_frame)

            # Make prediction
            processed_frame = preprocess_frames(np.array(frames))
            predictions = model.predict(np.array(processed_frame))
            predicted_class_index = np.argmax(predictions)
            behaviour = class_labels[predicted_class_index]

            if behaviour == 'Relaxed' or 'Sad':
                total_frames += 1

            if behaviour == 'Relaxed':
                relaxed_count += 1
            elif behaviour == 'Sad':
                sad_count += 1

            # Add text annotation to the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            font_thickness = 5
            color = (255, 0, 0)  # Red color for annotation
            text_size = cv2.getTextSize(behaviour, font, font_scale, font_thickness)[0]
            text_x = 30  # Horizontal position from the left edge
            text_y = frame_height - 30  # Vertical position from the bottom edge
            cv2.putText(frame, behaviour, (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)
            out.write(frame)

            frames.pop(0)  # Remove the oldest frame from the list

        # Close the temporary file
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Calculate percentages
        relaxed_percentage = round((relaxed_count / total_frames) * 100)
        sad_percentage = round((sad_count / total_frames) * 100)

        # Display counts of frames predicted as 'Relaxed' and 'Sad'
        st.write(f"Your Pet is 'Relaxed': {relaxed_percentage}%")
        st.write(f"Your Pet is 'Sad': {sad_percentage}%")

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
                #st.video(video_bytes)
                st.write(annotated_video_path)


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


with tab2:
    st.header("Upload your Pet Image")
   
    # Load the trained model
    model = load_model('efficientnet2.h5')

    # Get user input for image upload
    uploaded_file = st.file_uploader('Upload an image of your pet to understand its behaviour', type=['jpg', 'jpeg', 'png'])

    # Process the uploaded image if it exists
    if uploaded_file is not None:
        # Open and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Load new images for prediction
        new_image = image
        new_image_array = np.array(new_image.resize((384, 384)))  # Resize the image to match the model's input shape
        new_image_array = np.expand_dims(new_image_array, axis=0)  # Add batch dimension
        new_image_array = new_image_array / 255.0  # Normalize the pixel values (same as during training)

        # Make predictions
        predictions = model.predict(new_image_array)

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions, axis=1)[0]

        # Indicate class names
        class_names = ['Relaxed', 'Sad']

        # Get the predicted class name
        predicted_class_name = class_names[predicted_class_index]

        st.write(f"Your Pet is feeling: {predicted_class_name}")


with tab3:
    st.header("Understand your Pet Emotions")

    # Dog images and descriptions
    dog_images = {
        "Happy": "./images/happy_dog.jpg",
        "Angry": "./images/angry_dog.jpg",
        "Relaxed": "./images/relaxed_dog.jpg",
        "Sad": "./images/sad_dog.jpg"
    }

    dog_descriptions = {
        "Happy": "When your dog is happy, reward them with treats and playtime. Positive reinforcement helps reinforce good behavior.",
        "Angry": "If your dog is showing signs of aggression, give them space and avoid confrontations. Consult a professional dog trainer if needed.",
        "Relaxed": "A relaxed dog is content. Provide a comfortable environment, a cozy bed, and soft toys. Spend quality time together.",
        "Sad": "If your dog seems sad, offer extra attention and love. Take them for a walk, engage in activities, and offer their favorite treats."
    }

    for emotion, image_path in dog_images.items():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(image_path, caption=emotion, use_column_width=True)
        with col2:
            st.write(f"**{emotion} Dog:**")
            st.write(dog_descriptions[emotion])

    # Footer
    st.markdown("---")
    st.write("🐶 For more information and tips, consult a professional veterinarian or dog behaviorist.")

    # Additional styling
    st.markdown(
        """
        <style>
        .css-1g4b3f5 {
            background-color: #f0f5f9;
        }
        .css-1fa8wjc {
            background-color: #e0f3ff;
        }
        .css-1aumxhk {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )



       
