# Plant Disease Detection using Deep Learning

This project uses a Convolutional Neural Network (CNN) to detect diseases in plants from images of their leaves. The model is trained on the PlantVillage dataset and can identify various plant diseases across different crop species.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the PlantVillage dataset and organize it in the following structure:
```
dataset/
    train/
        disease_class_1/
            image1.jpg
            image2.jpg
            ...
        disease_class_2/
            image1.jpg
            image2.jpg
            ...
        ...
```

## Training the Model

To train the model, run:
```bash
python train.py
```

This will:
- Train the CNN model on your dataset
- Save the trained model as 'plant_disease_model.h5'
- Generate training history plots
- Save class indices for prediction

## Making Predictions

To predict disease for a new plant image:
```bash
python predict.py path/to/your/image.jpg
```

## Model Architecture

The CNN architecture consists of:
- 3 Convolutional blocks with batch normalization and max pooling
- Dense layers with dropout for classification
- Input image size: 224x224x3
- Output: Probability distribution over disease classes

## Performance

The model is trained for 15 epochs with data augmentation to prevent overfitting. Training history plots are saved showing accuracy and loss curves for both training and validation sets.
