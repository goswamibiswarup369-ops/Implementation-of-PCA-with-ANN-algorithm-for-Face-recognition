📌 Project Title:-
Implementation of PCA with ANN Algorithm for Face Recognition

📖 Overview:-
This project implements a Face Recognition System using Principal Component Analysis (PCA) for feature extraction and a Multi-Layer Perceptron (ANN / MLP) classifier for face classification.
The system follows the classic Eigenfaces approach, where high-dimensional face images are converted into a low-dimensional feature space using PCA, followed by classification using an Artificial Neural Network.

🎯 Objectives:-
Convert face images into numerical feature vectors
Reduce dimensionality using PCA (Eigenfaces)
Improve class separability using LDA
Train an ANN (MLPClassifier) for face recognition
Evaluate performance using test images and classification accuracy

🧰 Technologies & Libraries Used:-
Python 3
OpenCV (cv2) – Image reading & preprocessing
NumPy – Numerical computations
Matplotlib – Visualization
Scikit-learn
PCA (Dimensionality Reduction)
Linear Discriminant Analysis (LDA)
MLPClassifier (Artificial Neural Network)
Train-test split

📂 Dataset Structure:-

dataset/
 └── faces/
     ├── Person_1/
     │    ├── img1.jpg
     │    ├── img2.jpg
     │    └── ...
     ├── Person_2/
     │    ├── img1.jpg
     │    └── ...
     └── ...
Images are resized to 300 × 300
Converted to grayscale
Flattened into 1D vectors
🔄 Methodology
1️⃣ Image Preprocessing:-

Read image using OpenCV
Convert RGB → Grayscale
Resize to 300×300
Flatten image into vector
Assign numeric labels to each person
2️⃣ Dataset Preparation:-

Total samples: 450
Feature dimension: 90,000
Train-test split: 75% training / 25% testing

3️⃣ PCA (Eigenfaces):-
Applied PCA to reduce dimensionality
Number of components (Eigenfaces): 150
Extracted and visualized the most significant eigenfaces
PCA converts face images into a compact feature representation

4️⃣ LDA (Linear Discriminant Analysis):-
Applied LDA on PCA-reduced features
Improves class discrimination
Generates final feature vectors for ANN training

5️⃣ ANN Training (MLPClassifier):-
Model: Multi-Layer Perceptron
Hidden layers: (10, 10)
Max iterations: 1000
Optimized using back-propagation
Training loss reduces gradually during iterations

6️⃣ Face Prediction:-
Test images are:
Projected onto PCA eigenfaces
Transformed using LDA
Passed to trained ANN model
Model predicts:
Person label
Prediction probability