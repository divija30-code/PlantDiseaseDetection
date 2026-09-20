import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.mixed_precision import set_global_policy
import matplotlib.pyplot as plt
import os
import shutil

# Enable mixed precision for faster training
set_global_policy('mixed_float16')

# Set random seed for reproducibility
tf.random.set_seed(42)

# Model parameters
IMG_HEIGHT = 160
IMG_WIDTH = 160
BATCH_SIZE = 64
EPOCHS = 15

def create_model(num_classes):
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        
        # Second Convolutional Block
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        
        # Third Convolutional Block
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def prepare_apple_dataset():
    # Create directories for apple-only dataset
    apple_dataset_path = 'apple_dataset'
    if os.path.exists(apple_dataset_path):
        shutil.rmtree(apple_dataset_path)
    
    os.makedirs(os.path.join(apple_dataset_path, 'train'))
    os.makedirs(os.path.join(apple_dataset_path, 'val'))
    
    # Copy apple-related folders
    apple_classes = [
        'Apple___Apple_scab',
        'Apple___Black_rot',
        'Apple___Cedar_apple_rust',
        'Apple___healthy'
    ]
    
    for class_name in apple_classes:
        # Create directories
        os.makedirs(os.path.join(apple_dataset_path, 'train', class_name))
        os.makedirs(os.path.join(apple_dataset_path, 'val', class_name))
        
        # Get list of images
        source_dir = os.path.join('dataset/train', class_name)
        images = os.listdir(source_dir)
        
        # Split into train (80%) and validation (20%)
        split_idx = int(len(images) * 0.8)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Copy images
        for img in train_images:
            shutil.copy2(
                os.path.join(source_dir, img),
                os.path.join(apple_dataset_path, 'train', class_name, img)
            )
        
        for img in val_images:
            shutil.copy2(
                os.path.join(source_dir, img),
                os.path.join(apple_dataset_path, 'val', class_name, img)
            )
    
    return apple_dataset_path

def main():
    # Prepare apple-only dataset
    apple_dataset_path = prepare_apple_dataset()
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Load training data
    train_generator = train_datagen.flow_from_directory(
        os.path.join(apple_dataset_path, 'train'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # Load validation data
    validation_generator = val_datagen.flow_from_directory(
        os.path.join(apple_dataset_path, 'val'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # Create and compile model
    num_classes = len(train_generator.class_indices)
    model = create_model(num_classes)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )

    # Plot training history
    plot_training_history(history)

    # Save the model
    model.save('apple_disease_model.h5')
    
    # Save class indices
    import json
    with open('apple_class_indices.json', 'w') as f:
        json.dump(train_generator.class_indices, f)

    print("\nTraining completed!")
    print("Model saved as 'apple_disease_model.h5'")
    print("Class indices saved as 'apple_class_indices.json'")
    print("\nAvailable classes:", train_generator.class_indices)

if __name__ == '__main__':
    main()
