import os
import sys
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from classification_models.keras import Classifiers
from keras.applications.imagenet_utils import decode_predictions
from pyexiv2 import Image

print('WUTSDIS v1.0.0')

print('WUTSDIS loading model')
Inception, preprocess_input = Classifiers.get('inceptionresnetv2')
# load model
model = Inception(input_shape=(299,299,3), weights='imagenet', classes=1000)

def classify(path):
  # read and prepare image
  x = imread(path)
  x = resize(x, (299, 299)) * 255    # cast back to 0-255 range
  x = preprocess_input(x)
  x = np.expand_dims(x, 0)

  # processing image
  y = model.predict(x)

  # result
  predictions = [p[1] for p in decode_predictions(y)[0] if p[2] > 0.05]

  return predictions

def write_metadata(path, tags):
  i = Image(path)
  i.modify_xmp({
    'Xmp.dc.subject': tags
  })

def walk(path):
  for root, dirs, files in os.walk(path):
    path = root.split(os.sep)
    print((len(path) - 1) * '---', os.path.basename(root))
    for file in files:
      if file.rsplit('.',1)[1].lower() not in ['jpg', 'jpeg']:
        print(len(path) * '---', file, '<unclassified>')
        continue
      tags = classify(root+os.sep+file)
      write_metadata(root+os.sep+file, tags)
      print(len(path) * '---', file, tags)

print('WUTSDIS analyzing directory tree')
walk(sys.argv[1])
