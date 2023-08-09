# Image_Forgery_Detection
Image Forgery Detection using Transfer Learning Model like Inception V3 , ResNet 50 and CNN Model.

# Lung Cancer Prediction using Chest Scan Images
This repository contains a Python notebook that demonstrates the application of deep learning models (InceptionV3 and ResNet50) for predicting different types of lung cancer from chest scan images.

# Dataset
The dataset used in this project is available on Kaggle and can be accessed here "https://www.kaggle.com/datasets/sophatvathana/casia-dataset". It consists of chest scan images for lung cancer prediction.

# Installation
To run the notebook, you need to install the required libraries. You can do this by running the following commands:

pip install opendatasets
pip install pandas
pip install matplotlib
pip install tensorflow

# Usage
You can use the provided trained models to make predictions on your own chest scan images. To do this, utilize the chestScanPrediction function provided in the notebook. Just provide the path to your image as an argument to the function, along with the desired model.

Example usage:

path = "path_to_your_chest_scan_image.png"
chestScanPrediction(path, model_incep)  # Use model_incep or model_resnet as the model argument

Please ensure to adjust the path variable to the location of your image.

# Model Comparison
The project employs two models: InceptionV3 and ResNet50. Below is a comparison of their accuracy on the test set:

algos = ['Resnet50', 'InceptionV3']
accuracy = [accuracy_resnet, accuracy_incep]
accuracy = np.floor([i * 100 for i in accuracy])

plt.figure(figsize=(6, 5))
plt.bar(algos, accuracy, color='blue', width=0.4)
plt.xlabel("Algorithms Applied")
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.show()

# Confusion Matrix
The confusion matrix showcases the performance of the model on different lung cancer types:

cm = confusion_matrix(test_data.classes, y_pred)
plot_confusion_matrix(cm, target_names, title='Confusion Matrix')

Please adjust the parameters accordingly for accurate visualization.

Feel free to explore the notebook for more details on lung cancer prediction using chest scan images.

# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Author
Dharani G

