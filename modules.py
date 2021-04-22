import string


# function for loading file into memory
def load_doc(filename):
    # opens the file as read only
    file = open(filename, 'r')
    # reads all text
    text = file.read()
    # closes the file
    file.close()
    return text


# function to convert text into a dictionary of image_id and image_description
def load_description(text):
    mapping = dict()
    for line in text.split("\n"):
        token = line.split("#")
        if len(line) < 2:  # remove short descriptions
            continue
        img_id = token[0].split('.')[0]  # name of the image
        img_des = token[1][3:]  # description of the image
        if img_id not in mapping:
            mapping[img_id] = list()
        mapping[img_id].append(img_des)
    return mapping


# function of cleaning image description
def clean_description(desc):
    for key, des_list in desc.items():
        for i in range(len(des_list)):
            caption = des_list[i]
            caption = [ch for ch in caption if ch not in string
                .punctuation]
            caption = ''.join(caption)
            caption = caption.split(' ')
            caption = [word.lower() for word in caption if len(word) > 1 and word.isalpha()]
            caption = ' '.join(caption)
            des_list[i] = caption


# function to create vocabulary
def create_vocab(des):
    all_train_caption = []
    for k, v in des.items():
        for cap in v:
            all_train_caption.append(cap)
    # considering only words that occur more frequently
    word_count_threshold = 10
    word_count = {}
    n = 0
    for s in all_train_caption:
        n += 1
        for w in s.split():
            word_count[w] = word_count.get(w, 0) + 1

    vocab = [ws for ws in word_count if word_count[ws] >= word_count_threshold]
    return vocab


# Function to save descriptions to file
def save_descriptions(descriptions, filename):
    lines = list()
    # taking img_id,img_desc from description dict
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


# Function to load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
    # load document
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        # split id from description
        image_id, image_desc = tokens[0], tokens[1:]
        # skip images not in the set
        if image_id in dataset:
            # create list
            if image_id not in descriptions:
                descriptions[image_id] = list()
            # wrap description in tokens
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            # store
            descriptions[image_id].append(desc)
    return descriptions
