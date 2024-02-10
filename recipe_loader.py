import json
from PIL import Image
import requests
from io import BytesIO

def load_recipes(filename, food_type='salad'):
    with open(filename, 'r') as f:
        recipes = json.load(f)
    recipes = [x for x in recipes if food_type.lower() in x['title'].lower()]
    return recipes
def load_ingredients_of_recipes(filename, food_type='salad'):
    with open(filename, 'r') as f:
        recipes = json.load(f)
    recipes = [x["ingredients"] for x in recipes if food_type.lower() in x['title'].lower()]
    return recipes

def load_image(url):
    response = requests.get(url)
    try:
        img = Image.open(BytesIO(response.content))
    except:
        print(f'can not load image from {url}')
        img = None
    return img

if __name__ == '__main__':
    import pprint
    recipes = load_recipes('recipes_withImage.json')
    print(len(recipes))
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(recipes[0])

    img = load_image(recipes[0]['images'][0]['url'])
    if img:
        img.show()