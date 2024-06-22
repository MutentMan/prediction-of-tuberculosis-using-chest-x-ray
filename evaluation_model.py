import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Switch backend to 'Agg' for non-interactive plotting
import matplotlib
matplotlib.use('Agg')

# Load patient images
def load_patient_images(patient_dir):
    images = []
    image_names = []
    
    for filename in os.listdir(patient_dir):
        image_path = os.path.join(patient_dir, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (128, 128))
        images.append(image)
        image_names.append(filename)
    
    images = np.array(images)
    return images, image_names

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
    images = np.stack([cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB) for img in images])
    images = preprocess_input(images)
    features = model.predict(images)
    return features

if __name__ == '__main__':
    patient_dir = './images/patient'
    patient_images, image_names = load_patient_images(patient_dir)
    
    # Extract features for patient images
    patient_features = extract_features(patient_images)
    
    # Load the trained model
    model = load_model('tb_normal_model.h5')
    
    # Predict patient images
    patient_predictions = model.predict(patient_features)
    patient_pred_classes = np.argmax(patient_predictions, axis=-1)

    # Create directory for saving confusion matrices
    output_dir = './confusion_matrix'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize true and predicted labels
    true_labels = []
    predicted_labels = []
    class_mapping = {'tb': 0, 'normal': 1}
    inverse_class_mapping = {0: 'Tuberculosis', 1: 'Normal'}

    for i, image_name in enumerate(image_names):
        pred_class = patient_pred_classes[i]
        print(f"Image: {image_name}, Predicted Class: {inverse_class_mapping[pred_class]}")

        # Determine true label from filename
        true_label = 0 if 'tb' in image_name.lower() else 1
        true_labels.append(true_label)

        # Predicted label
        predicted_labels.append(pred_class)

        # Compute confusion matrix for individual patient image
        cm_patient = confusion_matrix([true_label], [pred_class], labels=[0, 1])
        
        # Display and save confusion matrix for individual patient image
        plt.figure()
        sns.heatmap(cm_patient, annot=True, fmt='d', cmap='Blues', xticklabels=['Tuberculosis', 'Normal'], yticklabels=['Tuberculosis', 'Normal'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for {image_name}')
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{image_name}.png'))
        plt.close()

    # Debug: Print true and predicted labels
    print(f"\nTrue Labels: {true_labels}")
    print(f"Predicted Labels: {predicted_labels}")

    # Compute evaluation metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    # Print evaluation metrics
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")