from imports import *
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
train_images = set(open(train_img_names, "r").read().strip().split("\n"))
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

train_descriptions = load_clean_descriptions('descriptions.txt', train)
print(f"length of training description is {len(train_descriptions)}")

# Now we will get InceptionV3 trained on imagenet dataset
model = InceptionV3(weights='imagenet')
# Now we will remove the last softmax layer to from InceptionV3
model_new = Model(model.input, model.layers[-2].output)


# Function to encode img in a vector of dim (2048,)
def encode(image):
    image = preprocess(image)  # preprocess the image
    fea_vec = model_new.predict(image)  # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])  # reshape from (1, 2048) to (2048, )
    return fea_vec

# uncomment it out to use it for first time
'''
# Encoding of train and test images will be done just once
# Lets now encode all images in training dataset
start = time()
encoding_train = {}
for img in train_img:
    encoding_train[img[len(images):]] = encode(img)
print("Time taken in seconds for training images=", time() - start)

with open("/home/prathamesh/Desktop/ML_Projects/Image_Captioning/encoded_train_images.pkl","wb") as encoded_pickle:
    pickle.dump(encoding_train, encoded_pickle)

# Similarly for test images
start = time()
encoding_test = {}
for img in test_img:
    encoding_test[img[len(images):]] = encode(img)
print("Time taken in seconds =", time() - start)

with open("/home/prathamesh/Desktop/ML_Projects/Image_Captioning/encoded_test_images.pkl",
          "wb") as encoded_pickle:
    pickle.dump(encoding_test, encoded_pickle)
'''


train_features = load(open("/home/prathamesh/Desktop/ML_Projects/Image_Captioning/encoded_train_images.pkl","rb"))
print(f"Length of training features {len(train_features)}")

test_features = load(open("/home/prathamesh/Desktop/ML_Projects/Image_Captioning/encoded_test_images.pkl","rb"))
print(f"Length of training features {len(test_features)}")


# Lets now create a list of all the training captions
all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)
print(f"The length of all training caption is {len(all_train_captions)}")

vocab = create_vocab(all_train_captions)
print(f' Length of our vocab is {len(vocab)}')

ixtoword = {}
wordtoix = {}
ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1
vocab_size = len(ixtoword)
print(f"The vocab size is {vocab_size}")

# using function in module to determine max length
max_length = max_length(train_descriptions)
print(f"Maximum length in the series {max_length}")

# Inorder to match index with word
vocab_size = vocab_size+1

# Now get it all only when running main file hence we can import global variable in different script
if __name__ == '__main__':


    glove_file = "/home/prathamesh/Desktop/ML_Projects/Image_Captioning/glove.6B.200d.txt"
    # Empty embedding dict
    embeddings_index = {}
    f = open(glove_file, encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print(f'Found {len(embeddings_index)} word vectors.')

    # Inorder to match index with word
    vocab_size = vocab_size + 1

    # Now lets get embedding matrix for our vocab
    embedding_dim = 200
    # Lets create a vectors of zeros for our embedding matrix
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in wordtoix.items():
        embedding_vector = embeddings_index.get(word)
        # Check if the word is in glove or not
        if embedding_vector is not None:
            # Chance index i of embedding matrix to the vector from glove
            embedding_matrix[i] = embedding_vector

    # Lets check the dimensions of the matrix
    print(f'Shape of embedding_matrix is : {embedding_matrix.shape}')
    # Lets build DL Model from the above vectors
    # First input layer for image vector
    input1 = Input(shape=(2048,))
    dp1 = Dropout(0.4)(input1)
    layer1 = Dense(256,activation="relu")(dp1)

    # Second input for partial captions
    input2= Input((max_length,))
    emb = Embedding(vocab_size,embedding_dim,mask_zero=True)(input2)
    dp1 = Dropout(0.4)(emb)
    layer_s1 = LSTM(256)(dp1)

    # Lets add both input layers together
    d1 = Add()([layer1,layer_s1])
    d2 = Dense(512,activation='relu')(d1)
    output = Dense(vocab_size,activation="softmax")(d2)
    model = Model(inputs = [input1, input2],outputs= output)

    model.summary()


    # Lets put the weights for embedding layer that is layer 2 in out model
    print(model.layers[2])
    model.layers[2].set_weights([embedding_matrix])
    model.layers[2].trainable = False

    # Now we are ready to compile the model
    model.compile(loss="categorical_crossentropy",optimizer="adam")


    epochs = 20
    number_of_pics_per_batch = 3
    steps = len(train_descriptions)//number_of_pics_per_batch

    for i in range(epochs):
        # Lets call generator to load images
        generator = data_generator(train_descriptions,train_features,wordtoix,max_length,number_of_pics_per_batch,vocab_size)
        # Now lets fit it into modelfrom tensorflow import keras
        model.fit(generator,epochs=1,steps_per_epoch=steps,verbose=1)
        # Lets save weights after every epoch to keep track
        model.save("/home/prathamesh/Desktop/ML_Projects/Image_Captioning/model_weights/model_" + str(i) + ".h5")



    # Now lets run our model on lower learning rate and higher batch size
    model.optimizer.lr = 0.0001
    epochs = 10
    number_of_pics_per_batch = 6
    steps = len(train_descriptions)//number_of_pics_per_batch

    # Now lets loop again for 10 epochs
    for i in range(epochs):
        generator = data_generator(train_descriptions, train_features, wordtoix, max_length, number_of_pics_per_batch,
                                   vocab_size)
        model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
        model.save("/home/prathamesh/Desktop/ML_Projects/Image_Captioning/model_weights/model_" + str(i+20) + ".h5")

    # Lets save the final weights
    model.save_weights('/home/prathamesh/Desktop/ML_Projects/Image_Captioning/model_weights/model_30.h5')

    '''
    
    # Lets load the weights for testing
    model= tf.keras.models.load_model('/home/prathamesh/Desktop/ML_Projects/Image_Captioning/model_weights/model_19.h5')
    
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
    
    
    ap=7
    pic = list(encoding_test.keys())[ap]
    image = encoding_test[pic].reshape((1,2048))
    x=plt.imread(images+pic)
    plt.imshow(x)
    plt.show()
    print("Greedy:",greedySearch(image))
    
    '''