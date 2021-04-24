from main import *
import streamlit as st
import cv2

'''
st.title('📷 Image Captioning 📷')
st.write("Made By PDP with ❤️")
st.write("Welcome to Image Captioning Project")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
print(uploaded_file)
if uploaded_file is not None:
    image = cv2.imread(str(uploaded_file.read()))
    st.image(image, width=None)

'''
model = tf.keras.models.load_model('/home/prathamesh/Desktop/ML_Projects/Image_Captioning/model_weights/model_29.h5')

# 45 7 100 25 60 85 120 36 71 250 255 290 259 555

#Lets test it out
images = "/home/prathamesh/Desktop/ML_Projects/Image_Captioning/flickr_data/Flickr_Data/Images/"
with open("/home/prathamesh/Desktop/ML_Projects/Image_Captioning/encoded_test_images.pkl","rb") as encoded_pickel:
    encoding_test = load(encoded_pickel)

# Function for maximum likelihood algo
def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

# 600 555 377 780 666 677 955 977
ap= 990
pic = list(encoding_test.keys())[ap]
image = encoding_test[pic].reshape((1,2048))
x=plt.imread(images+pic)
plt.imshow(x)
plt.show()
print("Greedy:",greedySearch(image))

