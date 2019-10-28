from django.shortcuts import render
from django.http import JsonResponse
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
from . import model
imgs_path = "app/style/"
imgs_model_width, imgs_model_height = 224, 224

nb_closest_images = 5 # number of most similar images to retrieve
# load the model
vgg_model = vgg16.VGG16(weights='imagenet')

# remove the last layers in order to get features instead of predictions
feat_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc2").output)

def index(request):
    context = []
    for i,item in data.iterrows():
       context.append({"id":i,"brand":item["brand_name"] ,"product":item["product_name"],"img":item["file"]})
    return render(request,"index.html",{"context":context})

def similar(request):

    img_id = request.GET.get("id")
    sim_imgs = list(model.m(img_id))
    for i,img in enumerate(sim_imgs):
        sim_imgs[i] = "static"+img[9:]
    print(sim_imgs)
    return JsonResponse({"data":sim_imgs},safe=False,content_type='json/application')
