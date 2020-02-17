# -*- coding: utf-8 -*-
# app.py

from flask import Flask, request, render_template
import pickle
import gzip
import numpy as np
import AiPoems

#start up cell -- import necessary metadata for model
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with gzip.GzipFile('predictors.npy.gz', "r") as f:
    predictors=np.load(f)
with gzip.GzipFile('label.npy.gz', "r") as f:
    label=np.load(f)
total_words=len(label[0])
max_sequence_len=len(predictors[0])+1
filename='word_model_love_poems_composite_100.h5'

#start up cell -- initialize model
model = AiPoems.initialize_model(predictors, label, max_sequence_len, 
                 total_words, device='/cpu:0')
model=AiPoems.load_model(model, filename)

text=AiPoems.generate_text_random(model, tokenizer, 10, max_sequence_len, seed_text="starttoken", top_n=10)

app = Flask(__name__)

@app.route("/example")
def get_numbers():
    #return ExampleService().supply_numbers(1,2)
    return str(1+2)
    #return ExampleModel().add_numbers(5,5)

@app.route("/")
def home():
    return render_template("home.html")
    
@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/make")
def make():        
    return render_template("make.html")

@app.route("/generatedPoem")
def generatedPoem():
    #choices=['this is string 1', 'this is string 2', 'a cat is a cat is a cat', 'the rain is spain']
    #import random
    AiPoem=AiPoems.generate_text_random(model, tokenizer, 50, max_sequence_len, seed_text="starttoken", top_n=10)
    AiPoem=AiPoem.replace('starttoken','').replace('returntoken','\n').split('endtoken2')[0]
    AiPoem=AiPoem.strip()
    #text=str(max_sequence_len)
    
    #text=random.choice(choices)
    return render_template("generatedPoem.html", text=AiPoem)


if __name__ == "__main__":
    app.run(debug=False)