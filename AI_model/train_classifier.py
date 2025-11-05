import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib

# Load saved embeddings and labels
embeddings = np.load('.\model\embeddings.npy')
labels = np.load('.\model\labels.npy')

# Encode string labels to integers
le = LabelEncoder()
integer_labels = le.fit_transform(labels)

# Train SVM classifier
classifier = SVC(kernel='linear', probability=True)
classifier.fit(embeddings, integer_labels)

# Save classifier and label encoder
joblib.dump(classifier, '.\model\svm_classifier.joblib')
joblib.dump(le, '.\model\label_encoder.joblib')

print("Classifier training complete and models saved.")
