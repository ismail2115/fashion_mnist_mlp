# 🧠 Fashion-MNIST MLP Classifier

This project implements a *simple fully connected neural network (Multilayer Perceptron)* on the *Fashion-MNIST* dataset using *PyTorch*.  
It follows all assignment requirements: dataset loading, preprocessing, model building, training, evaluation, and result visualization.

---

## 📚 Dataset Overview

*Fashion-MNIST* is a dataset of 70,000 grayscale images of clothing items (28×28 pixels each), divided into 10 categories:

T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

- Training set: 60,000 images  
- Test set: 10,000 images  
- Automatically downloaded using torchvision.datasets.FashionMNIST

---

## 🧩 Model Architecture

| Layer | Description |
|--------|--------------|
| Input | 784 features (28×28 flattened) |
| Hidden Layer 1 | 256 neurons, ReLU activation |
| Hidden Layer 2 | 128 neurons, ReLU activation |
| Output | 10 neurons (one per class) |

*Loss:* CrossEntropyLoss  
*Optimizer:* Adam (learning rate = 0.001)  
*Batch size:* 64  
*Epochs:* 10  
*Device:* CUDA if available, else CPU

---

## 🚀 How to Run

### ⿡ Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python fashion_mnist_mlp.py 
