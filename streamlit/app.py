
import streamlit as st
import os
os.environ['CV2_CUDNN_STREAM'] = '1'
import cv2
import pandas as pd
import numpy as np
import pickle
import tempfile
from PIL import ImageFont, ImageDraw, Image
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

# Load the trained model
model = load_model('efficientnet2.h5')

with tab1:
    st.header("Upload your Pet Video")
    
    class_labels = ['Relaxed', 'Sad']

    # Set font
    font = ImageFont.truetype(font = "C:/Windows/Fonts/Arial.ttf", size = 40)

    stframe = st.empty()
    total_frames = 0
    relaxed_count = 0
    sad_count = 0

    #file uploader
    video_file_buffer = st.file_uploader("Upload an image/video", type=[ "jpeg","jpg","png","mp4", "mov",'avi','asf','m4v'])

    #temporary file name 
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if video_file_buffer:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

        #values 
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc('V','P','0','9')
        out = cv2.VideoWriter('output1.webm', codec, fps, (width, height))

        while vid.isOpened():

            ret, frame = vid.read()
            if ret == False:
                break

            #recoloring it back to BGR b/c it will rerender back to opencv
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #image.flags.writeable = True

            # Resize the frame to match the model's input shape (384x384)
            resized_image = cv2.resize(image, (384, 384))

            # Convert the resized image to an array of floating-point numbers
            image_array = np.array(resized_image.astype(np.float32)) / 255.0
            image_array = image_array.reshape((1, 384, 384, 3))

            try:
                # Make prediction
                predictions = model.predict(image_array)
                predicted_class_index = np.argmax(predictions)
                behaviour = class_labels[predicted_class_index]

                if behaviour == 'Relaxed' or 'Sad':
                    total_frames += 1

                if behaviour == 'Relaxed':
                    relaxed_count += 1
                elif behaviour == 'Sad':
                    sad_count += 1

                #st.write(behaviour)

                # setting image writeable back to true to be able process it
                image.flags.writeable = True
                pil_im = Image.fromarray(image)
                draw = ImageDraw.Draw(pil_im)

                # Print the predicted class on video frame
                draw.text((100,400), behaviour,font=font)
                image = np.array(pil_im)

            except:
                pass                


            # To display the annotated live video feed
            stframe.image(image, use_column_width=True)


        vid.release()
        out.release()
        cv2.destroyAllWindows()

        # Calculate percentages
        relaxed_percentage = round((relaxed_count / total_frames) * 100)
        sad_percentage = round((sad_count / total_frames) * 100)

        # Display counts of frames predicted as 'Relaxed' and 'Sad'
        st.write(f"Your Pet is 'Relaxed': {relaxed_percentage}%")
        st.write(f"Your Pet is 'Sad': {sad_percentage}%")



with tab2:
    st.header("Upload your Pet Image")

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
    # Set custom text font and color styles
    st.markdown(
        """
        <style>
        .header-text {
            font-size: 24px;
            color: #4CAF50; /* Green color for headers */
        }
        .subheader-text {
            font-size: 24px;
            color: #FF5733; /* Orange color for subheaders */
        }
        .emotion-text {
            font-size: 16px;
            color: #333; /* Dark gray color for emotion text */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Header section
    st.markdown('<p class="header-text">Understand your Pet Emotions</p>', unsafe_allow_html=True)
    st.write("Start practicing little habits to keep your dog healthy physically and emotionally.")

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

    # Display dog images and descriptions
    for emotion, image_path in dog_images.items():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(image_path, caption=emotion, use_column_width=True)
        with col2:
            st.markdown(f"**{emotion} Dog:**", unsafe_allow_html=True)
            st.markdown(f"<p class='emotion-text'>{dog_descriptions[emotion]}</p>", unsafe_allow_html=True)

    # Dog emotions and corresponding descriptions
    emotions = {
        "Happy": {
            "Signs": "Wagging tail, relaxed body, interest in playing.",
            "Action": "Reward with treats and playtime."
        },
        "Angry": {
            "Signs": "Baring teeth, growling, tense body.",
            "Action": "Give space, avoid confrontations. Consult a professional trainer if needed."
        },
        "Relaxed": {
            "Signs": "Loose body, gentle tail wagging, comfortable lying down.",
            "Action": "Provide a comfortable environment, cozy bed, and spend quality time together."
        },
        "Sad": {
            "Signs": "Droopy ears, tucked tail, avoiding interaction.",
            "Action": "Offer extra attention, take for a walk, engage in activities, offer favorite treats."
        }
    }

    # Dog emotions recognition section
    st.markdown('<p class="subheader-text">Recognising Your Dog\'s Emotions</p>', unsafe_allow_html=True)
    st.write("Understanding your dog's emotions is essential for their well-being. Here are some tips to recognise your dog's feelings:")

    # Display dog emotions and corresponding signs/actions
    for emotion, details in emotions.items():
        st.markdown(f"<p class='emotion-text'><strong>{emotion} Dog:</strong></p>", unsafe_allow_html=True)
        st.markdown(f"<ul class='emotion-text'><li>Signs: {details['Signs']}</li><li>Action: {details['Action']}</li></ul>", unsafe_allow_html=True)
        st.write("---")  # Divider between emotions

    # Footer
    st.write("🐶 For more information and tips, consult a professional veterinarian or dog behaviorist.")



