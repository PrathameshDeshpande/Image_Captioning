import pandas as pd
from modules import load_description,clean_description,create_vocab
filename = "captions.txt"
file = open(filename,"r")
doc = file.read()
descriptions = load_description(doc)
clean_description(descriptions)
vocab = create_vocab(descriptions)
print(descriptions["3421129418_088af794f7"])
print('preprocessed words %d ' % len(vocab))
print(vocab[:25])

train = []
num = 0
for k in descriptions.keys():
    train.append(k)
    num += 1
    if(num>5):
        break

train_description = dict()
for img_id in train:
    train_description[img_id] = list()
    image_desc = descriptions[img_id]
    for caption in image_desc:
        c = 'startseq ' + ''.join(caption) + ' endseq'
        train_description[img_id].append(c)

print(len(train))
print(len(train_description))





