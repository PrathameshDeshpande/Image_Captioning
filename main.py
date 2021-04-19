import pandas as pd
from modules import load_description
filename = "captions.txt"
file = open(filename,"r")
doc = file.read()

descriptions = load_description(doc)
print(descriptions["3421129418_088af794f7"])