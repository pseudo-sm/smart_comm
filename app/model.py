
import pandas as pd
data = pd.read_csv("app/style/style.csv")

from keras.applications import vgg16
from keras.preprocessing.image import load_img,img_to_array
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input

from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
def m(img_id):

    print(files[:4])
    original = load_img(files[int(img_id)], target_size=(imgs_model_width, imgs_model_height))
    numpy_image = img_to_array(original)

    closest_imgs = cos_similarities_df[files[int(img_id)]].sort_values(ascending=False)[1:nb_closest_images+1].index
    closest_imgs_scores = cos_similarities_df[files[int(img_id)]].sort_values(ascending=False)[1:nb_closest_images+1]
    return  (closest_imgs)



imgs_path = "app/style/"
imgs_model_width, imgs_model_height = 224, 224

nb_closest_images = 5 # number of most similar images to retrieve
# load the model
vgg_model = vgg16.VGG16(weights='imagenet')

# remove the last layers in order to get features instead of predictions
feat_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc2").output)
files = [imgs_path + x for x in os.listdir(imgs_path) if "png" in x]
importedImages = []
for f in files:
    filename = f
    original = load_img(filename, target_size=(224, 224))
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)
    importedImages.append(image_batch)

images = np.vstack(importedImages)
processed_imgs = preprocess_input(images.copy())
imgs_features = feat_extractor.predict(processed_imgs)
cosSimilarities = cosine_similarity(imgs_features)
cos_similarities_df = pd.DataFrame(cosSimilarities, columns=files, index=files)


