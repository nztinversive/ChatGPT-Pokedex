# Pokémon Classifier

This Pokémon Classifier web application allows users to upload images of Pokémon, and the app will identify the Pokémon and provide a short bio. It uses a trained convolutional neural network (CNN) for classification and the OpenAI GPT-3.5-turbo model for generating Pokémon bios.

## Installation

1. Clone this repository

cd pokemon-classifier
Install the required packages using pip

pip install -r requirements.txt
Set your OpenAI API key as an environment variable

export OPENAI_API_KEY=your-openai-key
Run the Flask application

python main.py
The application will be accessible at http://localhost:65/.

Main Technologies and Libraries
Python
Flask
PyTorch
OpenAI GPT-3.5-turbo
Pillow
torchvision
Model
The Pokémon classifier model is a custom convolutional neural network (CNN) with 4 convolutional layers, max-pooling, and dropout layers, followed by 2 fully connected layers. The model has been trained on images of Pokémon from the Kanto region, and can recognize 151 different Pokémon.

Preprocessing
The input image is preprocessed by converting it to RGB format, resizing it to 64x64, and normalizing the pixel values.

Make sure to add your OpenAI API key to the environment variables of the deployment platform.
