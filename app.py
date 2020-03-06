# Etapa 1: Importação das bibliotecas
import os
import requests
import numpy as np
import tensorflow as tf

from scipy.misc import imread, imsave
from tensorflow.keras.datasets import fashion_mnist
from flask import Flask, request, jsonify

print(tf.__version__)

# Etapa 2: Carregamento do modelo pré-treinado
with open("gato-cachorro.json", "r") as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)
model.load_weights("gato-cachorro.h5")
model.summary()

# Etapa 3: Criação da API em Flask
app = Flask(__name__)

# Função para classificação de imagens
@app.route("/<string:img_name>", methods = ["POST"])
def classify_image(img_name):
    upload_dir = "uploads/"
    
    classes = ["cat", "dog"]
     
    img= tf.keras.preprocessing.image.load_img(r'uploads/'+img_name, target_size=(224,224))
    img= tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img= tf.keras.applications.resnet50.preprocess_input(img)
     # [1, 28, 28] -> [1, 784]
    prediction = model.predict(img)
    print(prediction)
    
    return jsonify({"TipoDoAnimal": classes[np.argmax(prediction[0])]})

# Iniciando a aplicação Flask
app.run(port = 5000, debug = False)   































