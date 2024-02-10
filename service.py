import math
import dash
import time
import cv2 
import random
from PIL import Image
from io import BytesIO
import dash_core_components as dcc
import dash_html_components as html
import plotly
import numpy as np
import os
import torch
from networks import TextEncoder
from networks_StackGANv2 import G_NET
from utils import load_dict, get_title_wordvec, get_ingredients_wordvec, get_instructions_wordvec
from torchvision import utils as vutils
import json
from model import Model
import yaml
import recipe_loader
# Load models externally, thus we only need to load the model once
# If you want to deply model using Flask, see https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html
text_encoder = TextEncoder(
    data_dir='./models/', text_info='010', hid_dim=300,
    emb_dim=300, z_dim=1024, with_attention=2,
    ingr_enc_type='rnn').eval()
text_encoder.load_state_dict(torch.load('./models/text_encoder.model'))
food_types = ['salad', 'cookie', 'muffin']
models = {x:Model(x) for x in food_types}

# food_ingredients.json stores the ingredients for each food_type, already sorted by frequence
with open('./models/food_ingredients.json', 'r') as f:
    food_ingredients = json.load(f)
    for food_type in food_types:
        ingr2c_tuples = food_ingredients[food_type]
        # pick up the top 100 ingredients by frequence, then sorted by alphabets
        ingredients, freqs = zip(*sorted(ingr2c_tuples[:100]))
        
        ingredients_vocab = [s.replace('_', ' ') for s in ingredients]
        models[food_type].ingredients_vocab = ingredients_vocab
        
        # normalized and used as probabilities in sampling 
        frequences_list = np.array(freqs)/sum(freqs)
        models[food_type].frequences_list = frequences_list
        
        # load models for each food_type
        netG = G_NET(levels=3).eval()
        netG.load_state_dict(torch.load(f'./models/gen_{food_type}_cycleTxt1.0_e300.model'))
        models[food_type].generator = netG
# load instructions word mapping file, used for model data preprocessing 
word2i = load_dict('./models/vocab_inst.txt')
print('vocab_inst size =', len(word2i))

# load ingredients word mapping file, used for model data preprocessing 
ingr2i = load_dict('./models/vocab_ingr.txt')
print('vocab_ingr size =', len(ingr2i))

# default ingredients vocabulary
ingredients_vocab = models['salad'].ingredients_vocab

all_recipes = {}
print("loading all_recipe")
all_recipes["salad"] = recipe_loader.load_ingredients_of_recipes("recipes_withImage.json",food_type="salad")
all_recipes["cookie"] = recipe_loader.load_ingredients_of_recipes("recipes_withImage.json",food_type="cookie")
all_recipes["muffin"] = recipe_loader.load_ingredients_of_recipes("recipes_withImage.json",food_type="muffin")
print("finish loading all_recipe")
def vectorize(recipe, word2i, ingr2i):
    """data preprocessing, from recipe text to one-hot inputs

    Arguments:
        recipe {dict} -- a dictionary with 'title', 'ingredients', 'instructions'
        word2i {dict} -- word mapping for title and instructions
        ingr2i {dict} -- ingredient mapping

    Returns:
        list -- a list of three tensors [title, ingredients and instructions]
    """    
    title = get_title_wordvec(recipe, word2i) # np.int [max_len]
    ingredients = get_ingredients_wordvec(recipe, ingr2i, permute_ingrs=False) # np.int [max_len]
    instructions = get_instructions_wordvec(recipe, word2i) # np.int [max_len, max_len]
    return [torch.tensor(x).unsqueeze(0) for x in [title, ingredients, instructions]]
# It will generate images and store the image to file system
def generate_images(food_type, ingredients, batch,histogram_equalization = 0,given_noise = None) :
    title = 'dummy title'
    instructions = 'dummy instructions'
    # print('DEBUG!!!', ingredients)

    recipe = {
        'title': title,
        'ingredients': [x.replace(' ', '_') for x in ingredients],
        'instructions': instructions
    }
    title_vec, ingrs_vec, insts_vec = vectorize(recipe, word2i, ingr2i)
    # print(ingrs_vec)
    title_vec = title_vec.repeat(batch, 1)
    ingrs_vec = ingrs_vec.repeat(batch, 1)
    insts_vec = insts_vec.repeat(batch, 1, 1)
    if(given_noise is None):
        noise = torch.FloatTensor(batch, 100).normal_(0, 1)
    else:
        noise = given_noise
    # noise = torch.FloatTensor(1, 100).normal_(0, 1)
    # noise = noise.repeat(batch, 1)
    model = models[food_type]
    netG = model.generator
    with torch.no_grad():
        text_feature = text_encoder([title_vec, ingrs_vec, insts_vec])
        imgs = netG(noise, text_feature)[0]
    for i in range(batch):
        image_tensor = vutils.make_grid(imgs[2][i][np.newaxis,:], nrow=1, normalize=True, scale_each=True, padding=0)
        image_np = image_tensor.numpy().transpose(1,2,0)
        image_np = np.uint8( image_np * 255.0 )
        image_np = np.ascontiguousarray(image_np, dtype=np.uint8)
        if(histogram_equalization==1):
            image_another = image_np.copy()
            image_np = improve_contrast_image_using_clahe(image_np)
        else:
            image_another = improve_contrast_image_using_clahe(image_np)

        im = Image.fromarray(image_np)
        im_another = Image.fromarray(image_another)
        # the file name is generated using current timestamp and a random number. This is not safe when many users use the website at a same time.
        filename_time=str(int(round(time.time()*1000000+random.random()*1000)))
        im.save("static/generated_images/"+filename_time+str(histogram_equalization)+".jpg")
        im_another.save("static/generated_images/"+filename_time+str(1-histogram_equalization)+".jpg")

    return filename_time+str(histogram_equalization)+".jpg",noise.numpy()


def improve_contrast_image_using_clahe(bgr_image: np.array) -> np.array:
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    hsv_planes = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv_planes[2] = clahe.apply(hsv_planes[2])
    hsv = cv2.merge(hsv_planes)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def random_recipe(food_type):
    return random.choice(all_recipes[food_type])
def generate_images_with_random_hq(food_type,ingredients,pic_num,possibility= 0.6):
    if(random.random()>possibility):
        print("using histogram_equalization")
        image_name, noise= generate_images(food_type,ingredients,1,histogram_equalization = 1)
        return image_name, noise,1
    else:
        print("not using!")
        image_name,noise = generate_images(food_type,ingredients,1,histogram_equalization = 0)
        return image_name, noise,0

# def save_image_np(image_np,filename):
#     image = Image.fromarray(image_np)
