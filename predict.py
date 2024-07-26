#!/usr/bin/python3

import argparse
import json
import numpy as np
from gles import Gle

par = argparse.ArgumentParser(description='Predict')
par.add_argument('input_img', default='flowers/test/1/image_06752.jpg')
par.add_argument('checkpoint', default='checkpoint.pth')
par.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
par.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
par.add_argument('--gpu', default=False, action='store_true')

ap = par.parse_args()
image_path = ap.input_img
number_of_outputs = ap.top_k
category_names = ap.category_names
hardware = "gpu" if ap.gpu else "cpu"
checkpoint_path = ap.checkpoint

print("Loading datasets...")
training_loader, testing_loader, validation_loader, _ = Gle.load_data()

print("Loading pre-trained model...")
model = Gle.load_checkpoint(checkpoint_path, hardware)

print("Running inference...")
probabilities = Gle.predict(image_path, model, 5, hardware)

print("Outputting results...")
with open(category_names, 'r') as json_file:
    cat_to_name = json.load(json_file)

labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0].cpu().numpy())]
probability = np.array(probabilities[0][0].cpu().numpy())
print("---------------")
for i in range(0, number_of_outputs):
    print("{} with a probability of {}%".format(labels[i], probability[i]*100))
print("---------------")