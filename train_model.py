import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

# Load dataset
def load_dataset(images_dir):
    images = []
    labels = []
    
    # Enumerate folders: 'tuberculosis' and 'normal'
    for label, folder in enumerate(['tuberculosis', 'normal']):
        folder_path = os.path.join(images_dir, folder)
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
            image = cv2.resize(image, (128, 128))
            images.append(image)
            labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Extract features based on VGG16 model spec. in pattern recognition 
def extract_features(images):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu')
    ])
    images = preprocess_input(images)
    features = model.predict(images)
    return features

if __name__ == '__main__':
    images_dir = './images'
    images, labels = load_dataset(images_dir)
    
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    # Extract features
    train_features = extract_features(X_train)
    test_features = extract_features(X_test)

    # Build and train the classification model
    model = Sequential([
        Dense(512, activation='relu', input_shape=(train_features.shape[1],)),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dense(2, activation='softmax')  # Output dimension is 2 for two classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_features, y_train, epochs=10, batch_size=32, validation_split=0.2)
    
    # Evaluate the model on test data
    test_loss, test_accuracy = model.evaluate(test_features, y_test)
    print(f"Test Accuracy: {test_accuracy}")

    # Save the trained model
    model.save('tb_normal_model.h5')
    
    print("Model training completed and saved as 'tb_normal_model.h5'")
