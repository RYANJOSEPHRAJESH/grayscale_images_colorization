# Grayscale Image Colorization

## 📝 Project Description
This project demonstrates how to colorize grayscale images using a deep learning model. The system is built on a Convolutional Neural Network (CNN) that has been trained to predict the missing color information from a black-and-white input image.

## ✨ Key Features
Automatic Colorization: Transforms any grayscale image into a full-color one without manual input.

Deep Learning Model: Utilizes a custom-trained CNN to learn color patterns and relationships from a large dataset.

Lab Color Space: Employs the Lab color space to separate lightness from color, simplifying the learning task for the model.

## ⚙️ How It Works
The core of this project is a neural network trained on thousands of colored images. The process is as follows:

Image Conversion: A colored image is first converted into the Lab color space, splitting it into a grayscale L-channel (lightness) and two color channels (a and b).

Model Training: The model is fed the grayscale L-channel as input and is trained to predict the corresponding color channels (a and b) as output.

Inference: When a new grayscale image is provided, the trained model predicts the color channels, which are then merged with the original grayscale channel to reconstruct a final color image.
## 🚀 Prerequisites
Before you begin, ensure you have the following installed:

Python 3.x
Pip (Python package installer)

## 💻 Installation
1) Clone this repository:
```
git clone https://your-repository-link.git
cd grayscale-colorization
```
2) Install the required Python packages:
```
pip install -r requirements.txt
```
## Usage
Place the grayscale image(s) you want to colorize in a designated input folder (e.g., input_images/).

Run the colorization script from the command line:
```
python saved_model_testing_images.py
```
The colorized output image will be saved to an output folder (e.g., output_images/).
