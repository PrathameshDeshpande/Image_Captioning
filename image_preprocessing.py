from tensorflow.keras.preprocessing import image
import numpy as np
from numpy import array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from modules import load_doc
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


# Function to preprocess every image to make it according to InceptionV3 format
def preprocess(image_path):
    # Convert all the images to a size of 299x299 as expected by the inception v3 model
    img = image.load_img(image_path, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x


# Function to load a pre-defined list of photo identifiers
def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    # process line by line
    for line in doc.split('\n'):
        # skip empty lines
        if len(line) < 1:
            continue
        # get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)


# Generator function to load mini batches of data matrix into the memory

def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch,vocab_size):
    # getting our individual vectos in lists
    X1, X2, y = list(), list(), list()
    n=0
    # Now we will loop over every image
    while 1:
        for key, desc_list in descriptions.items():
            n+=1
            # retrieving the photo feature vector
            photo = photos[key+'.jpg']
            for desc in desc_list:
                seq = [wordtoix[word] for word in desc.split() if word in wordtoix]
                # Getting multiple X, y pairs as we need
                for i in range(1, len(seq)):
                    # Now dividing the seq in to a series of input words and just one output word
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input the sequence to get its length to max length
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # storing individual features vectors
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            # yield the batch data
            if n==num_photos_per_batch:
                # converting the list into arrays and combining them into one long vector
                yield [array(X1), array(X2)], array(y)
                # Getting ready for next batch
                X1, X2, y = list(), list(), list()
                n=0

