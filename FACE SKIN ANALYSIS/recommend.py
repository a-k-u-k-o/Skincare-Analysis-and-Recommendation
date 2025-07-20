import json

def recommend_products(concern):
    with open('products.json', 'r') as file:
        products = json.load(file)
    return products.get(concern, ["No recommendation found"])
