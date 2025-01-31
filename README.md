# Garbage Classification dataset
### Deep learning
![](images/all.png)

### Table of Contents

<details>
  <summary>Table of Contents</summary>

  - [About The Project](#about-the-project)
  - [Built With](#built-with)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
  - [Usage](#usage)
  - [Process](#process)
  - [Demo Video](#demo-video)


</details>

## About The Project
In this project, we explore deep learning techniques to develop an effective image classification model for garbage sorting. We focus on convolutional neural networks (CNNs) to automatically categorize waste into different types, such as plastic, metal, paper, and organic material. Through data preprocessing, model architecture optimization, and hyperparameter tuning, we aim to enhance the model’s accuracy and generalization. The performance of the model is evaluated using metrics such as accuracy and loss, providing insights into its effectiveness for real-world waste classification applications.
### Error Heatmap

The following heatmap shows the confusion matrix to analyze classification errors:

![Error Heatmap](images/matrix.png)

### Titanic Deaths and Survivals by Gender

The following graph illustrates the comparison between the number of deaths and survivors based on gender:

![Titanic Deaths and Survivals](images/Capture.2PNG.PNG)
### Survived vs Died by Class

The following pie charts compare the percentages of passengers who survived and died based on their class:

![Survived vs Died by Class](images/Capture1.PNG)
### Age Distribution by Survival

The following density plot shows the age distribution of passengers who survived and did not survive:

![Age Distribution by Survival](images/Capture4.PNG)

### Built With
* Programming Languages: Python
* Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
* Techniques: Supervised learning, feature engineering, cross-validation, grid search, etc'

## Getting Started
### Prerequisites
Make sure you have the following installed:

* Python 3.x.
* Pip – Python’s package manager for installing dependencies.
* Google Colab – Access it via Google Colab, no installation required.
* Required Libraries – Install necessary packages for deep learning and image processing.
* Enable GPU for Faster Training.

## Usage
* Upload the Notebook to Google Colab – Open Google Colab and upload Final_Project_v2_Deep_Learning3.ipynb.
* Run the Notebook – Execute the cells sequentially to preprocess data, train the model, and evaluate results.
* Modify Parameters – Adjust hyperparameters, model architecture, or dataset configurations to experiment with different settings.
* Analyze Results – Review performance metrics, confusion matrices, and visualizations to refine your model.

## Process
Loading and preparing the dataset.
Performing Exploratory Data Analysis (EDA) – handling duplicates, NULL values, and visualizing data.
Testing multiple learning algorithms to establish a baseline F1 score.
Conducting 5-fold cross-validation to select the best algorithm.
Training the dataset with the best-performing model.
Creating a confusion matrix and visualizing model performance.


## Demo Video
[Click here to watch the demo video](https://www.youtube.com/watch?v=GpCbY-wfVFE&t=1s)
