import pickle
import pandas as pd  # You know the usual stuffs
import numpy as np
import matplotlib.pyplot as plt
import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from modules import load_description, clean_description, create_vocab, load_doc, save_descriptions,load_clean_descriptions,max_length
from image_preprocessing import preprocess, load_set, data_generator
from pickle import dump, load
from time import time
import glob
from tensorflow.keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from tensorflow import keras
from tensorflow.keras.layers import concatenate
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import Input, layers
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Add