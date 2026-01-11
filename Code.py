import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import numpy as np
import os
import cv2

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

# Directory containing face images organized by person name folders
dir_name = "dataset/faces/"
y = []
X = []
target_names = []
person_id = 0
h = w = 300  # image resize dimension
n_samples = 0
class_names = []

for person_name in os.listdir(dir_name):
    dir_path = os.path.join(dir_name, person_name)
    class_names.append(person_name)
    for image_name in os.listdir(dir_path):
        image_path = os.path.join(dir_path, image_name)
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        resized_image = cv2.resize(gray, (h, w))     # resize
        v = resized_image.flatten()                  # flatten matrix to vector
        X.append(v)
        n_samples += 1
        y.append(person_id)
    person_id += 1

y = np.array(y)
X = np.array(X)
target_names = np.array(class_names)
n_features = X.shape[1]
n_classes = len(target_names)

print("Number of samples:", n_samples)
print("Number of features:", n_features)
print("Number of classes:", n_classes)

# Split into training and testing set 60:40
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)

# Apply PCA (Eigenfaces) for dimensionality reduction
n_components = 150  # can experiment with k values
print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))

pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)

eigenfaces = pca.components_.reshape((n_components, h, w))

plot_gallery(eigenfaces, ["eigenface %d" % i for i in range(eigenfaces.shape[0])], h, w)
plt.show()

print("Projecting the input data on the eigenfaces orthonormal basis")
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print(X_train_pca.shape, X_test_pca.shape)

# Compute Fisherfaces using LDA on PCA reduced data
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_pca, y_train)
X_train_lda = lda.transform(X_train_pca)
X_test_lda = lda.transform(X_test_pca)

# Train Multi-Layer Perceptron on LDA reduced data
clf = MLPClassifier(random_state=1, hidden_layer_sizes=(10, 10), max_iter=1000, verbose=True)
clf.fit(X_train_lda, y_train)

print("Model trained.")

# Predict test faces and probabilities
y_pred = []
y_prob = []

for test_face in X_test_lda:
    prob = clf.predict_proba(test_face.reshape(1, -1))[0]
    class_id = np.argmax(prob)
    y_pred.append(class_id)
    y_prob.append(np.max(prob))

y_pred = np.array(y_pred)

# Prepare titles for visualization
prediction_titles = []
true_positive = 0

for i in range(len(y_test)):
    pred_name = target_names[y_pred[i]]
    true_name = target_names[y_test[i]]
    prob = y_prob[i]
    title = f"pred: {pred_name}, pr: {prob:.1f}\ntrue: {true_name}"
    prediction_titles.append(title)
    
    if y_pred[i] == y_test[i]:
        true_positive += 1

# Plot results
plot_gallery(X_test, prediction_titles, h, w)
plt.show()

accuracy = true_positive * 100 / y_pred.shape[0]
print("Accuracy:", accuracy)