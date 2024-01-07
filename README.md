# Lung-Cancer-Detection-using-Keras
This project aims to detect lung cancer from CT-Scan images using deep learning techniques. The dataset used in this project contains CT-Scan images of Adenocarcinoma, Large cell carcinoma, Squamous cell carcinoma, and normal cells.
# Data  Description:
The CT-Scan images are in jpg or png format to fit the model. The dataset contains four main folders:

Adenocarcinoma: contains CT-Scan images of Adenocarcinoma of the lung. Adenocarcinoma is the most common form of lung cancer, accounting for 30% of all cases overall and about 40% of all non-small cell lung cancer occurrences.

Large cell carcinoma: contains CT-Scan images of Large-cell undifferentiated carcinoma of the lung. This type of lung cancer usually accounts for 10 to 15% of all cases of NSCLC.

Squamous cell carcinoma: contains CT-Scan images of Squamous cell carcinoma of the lung. This type of lung cancer is responsible for about 30% of all non-small cell lung cancers, and is generally linked to smoking.

Normal: contains CT-Scan images of normal cells.

The dataset is divided into three sets: training, testing, and validation. The training set contains 70% of the data, the testing set contains 20% of the data, and the validation set contains 10% of the data.

# Technologies Used

Python: The primary programming language for developing the machine learning project.

TensorFlow and Keras: TensorFlow is an open-source machine learning framework, and Keras is a high-level neural networks API running on top of TensorFlow. They are used for building and training deep learning models.

NumPy: NumPy is a powerful library for numerical computations in Python. It is used for handling arrays and numerical operations.

Matplotlib and Seaborn: Matplotlib is a plotting library for creating visualizations, and Seaborn is a statistical data visualization library. They are used for creating plots and visualizing data.

os: The os module provides a way to interact with the operating system. It is used for file and directory operations.

Image Processing Libraries: PIL (Python Imaging Library) and OpenCV (Open Source Computer Vision Library) are used for image processing and manipulation.

Joblib: Joblib is used for saving and loading models as files.

Scikit-learn: Scikit-learn is a machine learning library that provides tools for data preprocessing, model selection, and evaluation.

TQDM: TQDM is a library for displaying progress bars during iterative processes.
ImageDataGenerator (from Keras): Used for real-time data augmentation during model training.

TensorFlow Addons (tfa): An add-on library for TensorFlow that includes additional functionalities. In this case, it is used for additional metrics like F1 Score.

These technologies and libraries are commonly used in machine learning and deep learning projects for tasks such as data preprocessing, model building, training, evaluation, and visualization
# Model Architechture
The deep learning model used in this project is a convolutional neural network (CNN). The model consists of several layers including convolutional, max pooling, batch normalization, and dense layers. Transfer learning is used by initializing the model with pre-trained weights from one of the above-mentioned pre-trained models.
The model was trained using the Adam optimizer with a learning rate of 0.001, a batch size of 16, and a total of 50 epochs.
