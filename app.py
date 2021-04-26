from main import *
import streamlit as st
from PIL import Image
import os

# Lets load the saved model
model = tf.keras.models.load_model('/home/prathamesh/Desktop/ML_Projects/Image_Captioning/model_weights/model_29.h5')


# Function to get output for maximum probablity in our model output
def maximum_likelihood_estimation(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


# Streamlit App

st.markdown("<h1 style='text-align: right; color: black;'>📷 Image Captioning 📷</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'>Made By PDP with ❤</h1>", unsafe_allow_html=True)

st.title("Welcome to Image Captioning Project 🧑🏽‍💻")
st.write("⭐ A project where you upload any image and model will generate a caption for it ⭐")
uploaded_file = st.file_uploader("🌌 🌆Choose an image...", type="jpg")

# Check if the uploaded file contains data or not
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Lets Load Inceptionv3 model for early preprocessing of image

    imodel = InceptionV3(weights='imagenet')
    model_new = Model(imodel.input, imodel.layers[-2].output)

    # encode function to make image appropriate to our model

    def encode(image):
        image = preprocess(image)  # preprocess the image
        fea_vec = model_new.predict(image)  # Get the encoding vector for the image
        fea_vec = np.reshape(fea_vec, fea_vec.shape[1])  # reshape from (1, 2048) to (2048, )
        return fea_vec

    # get image path of chosen image

    image_path = os.path.join("/home/prathamesh/Desktop/ML_Projects/Image_Captioning/flickr_data/Flickr_Data/Images",
                              uploaded_file.name)
    image = encode(image_path)
    image = image.reshape((1, 2048))
    # 600 555 377 780 666 677 955 977
    x = maximum_likelihood_estimation(image)
    st.write("👉 The caption generated:", x)


