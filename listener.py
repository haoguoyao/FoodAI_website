from app import app
from flask import jsonify
from flask import request
from flask import Response
import DAO
import random
import service
from bson import json_util
from bson.json_util import dumps
import service
from flask import render_template
import json
import net_analyse
from model import Model
import numpy as np
import flask
import torch
from networks import TextEncoder
from networks_StackGANv2 import G_NET
import json
import yaml
with open('./static/assets/examples.yml', 'r') as f:
    examples = yaml.safe_load(f)

food_types = ['salad', 'cookie', 'muffin']
models = {x:Model(x) for x in food_types}
# food_ingredients.json stores the ingredients for each food_type, already sorted by frequence
with open('./models/food_ingredients.json', 'r') as f:
    food_ingredients = json.load(f)
    for food_type in food_types:
        ingr2c_tuples = food_ingredients[food_type]
        # pick up the top 100 ingredients by frequence, then sorted by alphabets
        #ingredients, freqs = zip(*sorted(ingr2c_tuples[:100]))
        ingredients, freqs = zip(*sorted(ingr2c_tuples[:300]))
        ingredients_vocab = [s.replace('_', ' ') for s in ingredients]
        models[food_type].ingredients_vocab = ingredients_vocab
        
        # normalized and used as probabilities in sampling 
        frequences_list = np.array(freqs)/sum(freqs)
        models[food_type].frequences_list = frequences_list
        
        # load models for each food_type
        netG = G_NET(levels=3).eval()
        netG.load_state_dict(torch.load(f'./models/gen_{food_type}_cycleTxt1.0_e300.model'))
        models[food_type].generator = netG

# return the main page
@app.route('/',methods=['GET'])
def home():
    print("Getting homepage")
    return app.send_static_file('index.html')
@app.route('/mapping_file',methods=['GET'])
def mapping_file():
    print("Getting file mapping page")
    return app.send_static_file('mapping_file.html')
@app.route('/mapping_file.html',methods=['GET'])
def mapping_file2():
    print("Getting file mapping page")
    return app.send_static_file('mapping_file.html')
# return the test page
@app.route('/testpage',methods=['GET'])
def testhome():
    print("Getting testpage")
    return app.send_static_file('test.html')

@app.route('/generate_image2', methods=['POST'])
def generate_image_temp():
    data = request.json 
    ingredients = data.get("ingredients")
    food_type = data.get("food_type")
    image_name,noise,histogram_equalization = service.generate_images_with_random_hq(food_type,ingredients,1)
    response = Response(dumps({"src":image_name}),content_type='application/json')
    return response

@app.route('/generate_image', methods=['POST'])
def index():
    if request.headers.getlist("X-Forwarded-For"):
        ip = request.headers.getlist("X-Forwarded-For")[0]
    else:
        ip = request.remote_addr

    user_agent = request.headers.get("User-Agent")
    user_agent = net_analyse.analyse_browser(user_agent)
    print(user_agent)
    data = request.json 
    ingredients = data.get("ingredients")
    food_type = data.get("food_type")
    image_name,noise,histogram_equalization = service.generate_images_with_random_hq(food_type,ingredients,1)
    DAO.generate_a_image(food_type,'/'.join([x.replace(' ', '_') for x in ingredients]),image_name,ip,user_agent,histogram_equalization,'/'.join(str(i) for i in noise[0]))
    response = Response(dumps({"src":image_name}),content_type='application/json')
    return response


@app.route('/random_recipe', methods=['POST'])
def random_recipe():
    data = request.json
    food_type = data.get("food_type")
    random_recipe = service.random_recipe(food_type)
    response = Response(dumps({"random_recipe":random_recipe}),content_type='application/json')
    return response



@app.route('/ingredients', methods=['POST'])
def get_all_ingredients():
    ingredients = {}
    for j in food_types:
        ingredients[j] = [{'label': i, 'value': i} for i in models[j].ingredients_vocab]

    response = Response(dumps(ingredients),content_type='application/json')
    return response
# rate an image using its file name
@app.route('/submit_rating', methods=['POST'])
def submit_rating():
    print("submit_rating listener")
    data = request.json 
    image_name = data.get("image_name")
    feedback = data.get("feedback")
    if(feedback!=None):
        DAO.rate_image(image_name,feedback)
        response = Response(dumps({"1":1}),content_type='application/json')
    return response

#get the total example number
@app.route('/example_number', methods=['POST'])
def example_dish():
    # Currently, the number needs to be changed manually
    response = Response(dumps({"number":9}),content_type='application/json')
    return response

# get ingredients of one example
@app.route('/one_example', methods=['POST'])
def one_example(): 
    data = request.json 
    number = data.get("number")
    response = Response(dumps({"ingredients":examples[int(number)]["example"]["ingredients"]}),content_type='application/json')
    return response
