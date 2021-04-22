import pandas as pd # You know the usual stuffs
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from modules import load_description,clean_description,create_vocab,load_doc,save_descriptions
from image_preprocessing import preprocess,load_set
from pickle import dump, load
from time import time
import glob

# InceptionV3 Model Parameters
WIDTH = 299
HEIGHT = 299
OUTPUT_DIM = 2048

filename = "/home/prathamesh/Desktop/ML_Projects/Image_Captioning/flickr_data/Flickr_Data/Flickr_TextData/Flickr8k.token.txt"
doc = load_doc(filename)
descriptions = load_description(doc)
clean_description(descriptions)
vocab = create_vocab(descriptions)
save_descriptions(descriptions, 'descriptions.txt')

print(descriptions["1000268201_693b08cb0e"])
print(f'preprocessed words {len(vocab)}')


filename = "/home/prathamesh/Desktop/ML_Projects/Image_Captioning/flickr_data/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt"
train = load_set(filename)
print(f"Training Data Length {len(train)}")

# Now lets get our images
images = "/home/prathamesh/Desktop/ML_Projects/Image_Captioning/flickr_data/Flickr_Data/Images/"
# getting list of all the image name
img = glob.glob(images + '*.jpg')
print(len(img))

# Now lets get the names of all the images used for training
train_img_names = "/home/prathamesh/Desktop/ML_Projects/Image_Captioning/flickr_data/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt"
train_images = set(open(train_img_names,"r").read().strip().split("\n"))
# lets get the path for every training img in below list
train_img = []
for i in img:
    if i[len(images):] in train_images:
        train_img.append(i)


# similarly for test images
test_img_names = "/home/prathamesh/Desktop/ML_Projects/Image_Captioning/flickr_data/Flickr_Data/Flickr_TextData/Flickr_8k.testImages.txt"
test_images = set(open(test_img_names, 'r').read().strip().split('\n'))
test_img = []
for i in img:
    if i[len(images):] in test_images:
        test_img.append(i)
















'''
train = []
test = []
num = 0
for k in descriptions.keys():
    if(num<6000):
        train.append(k)
    else:
        test.append(k)
    num += 1

train_description = dict()
test_description = dict()
for img_id in train:
    train_description[img_id] = list()
    image_desc = descriptions[img_id]
    for caption in image_desc:
        c = 'startseq ' + ''.join(caption) + ' endseq'
        train_description[img_id].append(c)

for img_id in test:
    test_description[img_id] = list()
    image_desc = descriptions[img_id]
    for caption in image_desc:
        c = 'startseq ' + ''.join(caption) + ' endseq'
        test_description[img_id].append(c)

print(len(train))
print(len(train_description))
print(len(test_description))

# Now we will get InceptionV3 trained on imagenet dataset
model = InceptionV3(weights='imagenet')
# Now we will remove the last softmax layer to from InceptionV3
model_new = Model(model.input, model.layers[-2].output)


# Function to encode a given image into a vector of size (2048, )
def encode(image):
    image = preprocess(image) # preprocess the image
    fea_vec = model_new.predict(image) # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec


# Call the funtion to encode all the train images
start = time()
encoding_train = {}
for img in train_img:
    encoding_train[img[len(images):]] = encode(img)
print("Time taken in seconds =", time()-start)


'''



