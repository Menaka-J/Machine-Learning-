import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from skimage.feature import hog
from skimage.color import rgb2gray
from joblib import Parallel, delayed
from tqdm import tqdm

# ========== Load CIFAR-10 ==========
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x = np.concatenate((x_train, x_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0).flatten()

# ========== Take a Subset ==========
N_SAMPLES = 5000
x = x[:N_SAMPLES]
y = y[:N_SAMPLES]

# ========== HOG Feature Extraction ==========
def extract_hog(img):
    gray = rgb2gray(img)
    return hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
               feature_vector=True, visualize=False)

print("Extracting HOG features from", N_SAMPLES, "images...")
x_hog = Parallel(n_jobs=-1)(delayed(extract_hog)(img) for img in tqdm(x))

# ========== Preprocessing ==========
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_hog)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=42)

# ========== Train & Evaluate Models ==========

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(x_train, y_train)
y_pred_nb = nb_model.predict(x_test)

# SVM
svm_model = SVC(kernel='linear')
svm_model.fit(x_train, y_train)
y_pred_svm = svm_model.predict(x_test)

# Labels
cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']

# ========== Reports ==========
print("\n=== Naive Bayes Classification Report ===")
print(classification_report(y_test, y_pred_nb, target_names=cifar10_labels))

print("\n=== SVM Classification Report ===")
print(classification_report(y_test, y_pred_svm, target_names=cifar10_labels))
