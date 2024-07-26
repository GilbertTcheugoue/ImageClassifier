# ImageClassifier

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, we first developped code for an image classifier built with PyTorch, then converted it into a command line application.

# Part 1 - Development Notebook

All the necessary packages and modules are imported in the first cell of the notebook
From the notebook we performed :

Training data augmentation :torchvision transforms are used to augment the training data with random scaling, rotations, mirroring, and/or cropping

Data normalization:  The training, validation, and testing data is appropriately cropped and normalized

Data batching:  The data for each set is loaded with torchvision's DataLoader

Data loading:  The data for each set (train, validation, test) is loaded with torchvision's ImageFolder

Pretrained Network:  A pretrained network such as VGG16 is loaded from torchvision.models and the parameters are frozen

Feedforward Classifier:  A new feedforward network is defined for use as a classifier using the features as input

Training the network: The parameters of the feedforward classifier are appropriately trained, while the parameters of the feature network are left static

Testing Accuracy: The network's accuracy is measured on the test data

Validation Loss and Accuracy:  During training, the validation loss and accuracy are displayed

Loading checkpoints: There is a function that successfully loads a checkpoint and rebuilds the model

Saving the model:  The trained model is saved as a checkpoint along with associated hyperparameters and the class_to_idx dictionary

Image Processing:  The process_image function successfully converts a PIL image into an object that can be used as input to a trained model

Class Prediction: The predict function successfully takes the path to an image and a checkpoint, then returns the top K most probably classes for that image

Sanity Checking with matplotlib: A matplotlib figure is created displaying an image and its associated top 5 most probable classes with actual flower names

# Part 2 - Command Line Application

Now that we have built and trained a deep neural network on the flower data set, we then converted it into an application. 

1. Train

Train a new network on a data set with train.py

. Basic usage: python train.py data_directory

. Prints out training loss, validation loss, and validation accuracy as the network trains

Options: 

* Set directory to save checkpoints: python train.py data_dir --save_dir save_directory 

* Choose architecture: python train.py data_dir --arch "vgg13" 

* Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20 

* Use GPU for training: python train.py data_dir --gpu

2. Predict
   
Predict flower name from an image with predict.py along with the probability of that name. 
That is, we passed in a single image /path/to/image and returned the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint

Options: * Return top K most likely classes: 

python predict.py input checkpoint --top_k 3 

* Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json *
