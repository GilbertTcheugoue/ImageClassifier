#!/usr/bin/python3

import argparse
from gles import Gle

par = argparse.ArgumentParser(description='Train')

par.add_argument('data_dir', default="flowers/")
par.add_argument('--epochs', dest="epochs", action="store", type=int, default=10)
par.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
par.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=100)
par.add_argument('--gpu', default=False, action='store_true')
par.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
par.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
par.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)

ap = par.parse_args()

data_dir = ap.data_dir
path = ap.save_dir
learning_rate = ap.learning_rate
architecture = ap.arch
dropout = ap.dropout
hidden_units = ap.hidden_units
hardware = "gpu" if ap.gpu else "cpu"
epochs = ap.epochs
print_every = 10

print("Loading datasets...")
train_loader, validation_loader, test_loader, train_dataset = Gle.load_data(data_dir)

print("Setting up model architecture...")
model, criterion, optimizer = Gle.model_setup(architecture, dropout, hidden_units, learning_rate, hardware)

print("Training model...")
Gle.train_network(train_loader, validation_loader, model, criterion, optimizer, epochs, print_every, hardware)

print("Validating testing accuracy...")
Gle.test_accuracy(model, test_loader, hardware)

print("Saving model to disk...")
Gle.save_checkpoint(model, train_dataset.class_to_idx, path, architecture, hidden_units, dropout, learning_rate)

print("Done!")
