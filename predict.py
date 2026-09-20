import tensorflow as tf
import numpy as np
from PIL import Image
import json

class AppleDiseasePredictor:
    def __init__(self, model_path='apple_disease_model.h5', class_indices_path='apple_class_indices.json'):
        # Load the trained model
        self.model = tf.keras.models.load_model(model_path)
        
        # Load class indices
        with open(class_indices_path, 'r') as f:
            self.class_indices = json.load(f)
            
        # Reverse the class indices dictionary to map indices to class names
        self.class_names = {v: k for k, v in self.class_indices.items()}
        
        # Create a more readable mapping for display
        self.display_names = {
            'Apple___Apple_scab': 'Apple Scab',
            'Apple___Black_rot': 'Black Rot',
            'Apple___Cedar_apple_rust': 'Cedar Apple Rust',
            'Apple___healthy': 'Healthy'
        }

    def preprocess_image(self, image_path):
        """Preprocess the input image for prediction."""
        try:
            img = Image.open(image_path)
            img = img.resize((160, 160))  # Resize to match training input size
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create batch axis
            img_array = img_array / 255.  # Rescale to match training
            return img_array
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")

    def predict_disease(self, image_path):
        """Predict the disease from an apple leaf image."""
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image_path)
            
            # Make prediction
            predictions = self.model.predict(processed_image)
            predicted_class_index = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_index]
            
            # Get the class name
            class_name = self.class_names[predicted_class_index]
            display_name = self.display_names[class_name]
            
            # Get confidence scores for all classes
            all_predictions = {self.display_names[self.class_names[i]]: float(pred) 
                             for i, pred in enumerate(predictions[0])}
            
            return {
                'prediction': display_name,
                'confidence': float(confidence),
                'all_predictions': all_predictions
            }
            
        except Exception as e:
            raise Exception(f"Error making prediction: {str(e)}")

def main():
    import sys
    import json
    
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py test_images/apple_scab.jpg")
        sys.exit(1)
        
    image_path = sys.argv[1]
    
    try:
        # Initialize predictor
        predictor = AppleDiseasePredictor()
        
        # Make prediction
        result = predictor.predict_disease(image_path)
        
        # Print results
        print("\nPrediction Results:")
        print("-" * 50)
        print(f"Predicted Disease: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nAll Predictions:")
        print("-" * 50)
        
        # Sort predictions by confidence
        sorted_predictions = dict(sorted(result['all_predictions'].items(), 
                                      key=lambda x: x[1], 
                                      reverse=True))
        
        for disease, confidence in sorted_predictions.items():
            print(f"{disease}: {confidence:.2%}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
