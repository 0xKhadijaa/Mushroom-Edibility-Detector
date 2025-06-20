# Imports
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, silhouette_score)
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from xgboost import XGBClassifier
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import os
import pyttsx3 

# Load dataset
df = pd.read_csv('mushroom_4.csv')

# Replace '?' with NaN
df = df.replace('?', np.nan)

# Separate target before modifying features
y = df["class"]

# Remove 'class' from features
X = df.drop("class", axis=1)

# Fill missing values and Encode features
label_encoders = {}
for column in X.columns:
    if X[column].isna().any():
        mode_val = X[column].mode()[0]
        X[column] = X[column].fillna(mode_val)
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column].astype(str))
    label_encoders[column] = le

# Encode target (class)
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_train, palette="viridis")
plt.title("PCA Visualization of Mushroom Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Edibility (0=Edible, 1=Poisonous, 2=Less Edible, 3=Less Poisoinous)")
plt.show()

# -------------------------
# KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

print("\n-----------------")
print("KNN Performance:")
print("-----------------")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_knn, average='macro'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_knn, average='macro'):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_knn, average='macro'):.4f}")
print(classification_report(y_test, y_pred_knn))

plt.figure(figsize=(6, 6))
cm_knn = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Blues")
plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------
# K-Means Clustering
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(X_train_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o", linestyle="--")
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Clustering with K=4
kmeans = KMeans(n_clusters=4, init="k-means++", random_state=42)
kmeans.fit(X_train_scaled)
y_kmeans = kmeans.predict(X_test_scaled)

print("\n--------------------------")
print("K-Means Clustering Results:")
print("----------------------------")
print(f"Silhouette Score: {silhouette_score(X_test_scaled, y_kmeans):.4f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans.labels_, palette="viridis")
plt.title("K-Means Clustering (K=2)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# -------------------------
# Naive Bayes
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
y_pred_nb = nb.predict(X_test_scaled)

print("\n-------------------------")
print("Naive Bayes Performance:")
print("-------------------------")
print(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_nb, average='macro'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_nb, average='macro'):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_nb, average='macro'):.4f}")
print(classification_report(y_test, y_pred_nb))

plt.figure(figsize=(6, 6))
cm_nb = confusion_matrix(y_test, y_pred_nb)
sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Greens")
plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------
# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

print("\n---------------------------")
print("Random Forest Performance:")
print("---------------------------")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf, average='macro'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf, average='macro'):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_rf, average='macro'):.4f}")

feature_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_imp)
plt.title("Random Forest Feature Importance")
plt.show()

# -------------------------
# SVM
svm = SVC(kernel="rbf", random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)

print("\n-----------------")
print("SVM Performance:")
print("-----------------")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_svm, average='macro'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_svm, average='macro'):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_svm, average='macro'):.4f}")

plt.figure(figsize=(6, 6))
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Oranges")
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------
# XGBoost
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train_scaled, y_train)
y_pred_xgb = xgb.predict(X_test_scaled)

print("\n---------------------")
print("XGBoost Performance:")
print("---------------------")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_xgb, average='macro'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_xgb, average='macro'):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_xgb, average='macro'):.4f}")

plt.figure(figsize=(6, 6))
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Purples")
plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def predict_from_image():
    try:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select a Mushroom Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )

        if not file_path:
            desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
            default_folder = os.path.join(desktop_path, 'AI image', 'images')
            if os.path.exists(default_folder):
                for file in os.listdir(default_folder):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        file_path = os.path.join(default_folder, file)
                        break

        if file_path:
            img = Image.open(file_path)
            plt.figure(figsize=(4, 4))
            plt.imshow(img)
            plt.axis("off")
            plt.title("Uploaded Mushroom Image")
            plt.show()

            print("\n(Note: Real image classification requires CNN, here we simulate using most common features.)")

            user_input = {column: X[column].mode()[0] for column in X.columns}
            user_df = pd.DataFrame([user_input])
            user_scaled = scaler.transform(user_df)

            prediction = rf.predict(user_scaled)
            prediction_label = le_target.inverse_transform(prediction)[0]

            if prediction_label == 'e':
                result_text = "The mushroom is likely EDIBLE!"
                print(f"\n🔵 {result_text}")
                speak(result_text)
            else:
                result_text = "Warning! The mushroom is likely POISONOUS!"
                print(f"\n🔴 {result_text}")
                speak(result_text)
        else:
            error_text = "No file selected and no images found in default folder."
            print(error_text)
            speak(error_text)

    except Exception as e:
        error_msg = f"An error occurred: {e}"
        print(error_msg)
        speak(error_msg)

predict_from_image()