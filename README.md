
# Customer Churn Prediction with Neural Networks

This project uses a neural network model to predict customer churn in a bank based on various features like credit score, age, and more. It utilizes the Keras library for building the neural network and pandas for data preprocessing. The dataset used in this project is from the "Churn_Modelling.csv" file.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Data Preprocessing](#data-preprocessing)
- [Neural Network Model](#neural-network-model)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)

## Prerequisites

Before running the code, ensure you have the following libraries installed:

- pandas
- scikit-learn
- Keras

You can install them using pip:

```bash
pip install pandas scikit-learn keras
```

## Data Preprocessing

- The dataset is loaded using pandas, and basic information about the dataset is displayed using `dataset.info()` and `dataset.describe()`.
- Categorical variables like "Geography," "Gender," etc., are one-hot encoded for the neural network model.

## Neural Network Model

- A sequential neural network model is built using Keras.
- The model consists of multiple dense layers with the ReLU activation function.
- The output layer uses the sigmoid activation function for binary classification.
- The model is compiled using binary cross-entropy loss.

## Training and Evaluation

- The dataset is split into training and testing sets using `train_test_split` from scikit-learn.
- The model is trained on the training data using `model.fit` and evaluated on the testing data.
- Metrics such as accuracy, precision, recall, and F1-score are calculated using scikit-learn's functions.

## Results

- The precision, recall, F1-score, and accuracy of the model are displayed.
- A confusion matrix is shown to visualize the model's performance.

## Conclusion

This project demonstrates how to use a neural network to predict customer churn in a bank. By preprocessing the data and building a neural network model, you can achieve valuable insights into customer retention. You can further optimize the model and explore different architectures to improve prediction accuracy.
