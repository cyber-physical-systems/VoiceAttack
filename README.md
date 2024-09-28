
# VoiceAttack: Fingerprinting Voice Command on VPN-Protected Smart Home Speakers

## Abstract

Recently, there are growing security and privacy concerns regarding smart voice speakers, such as Amazon Alexa and Google Home. Extensive prior research has shown that it is surprisingly easy to infer Amazon Alexa voice commands over their network traffic data. To prevent these traffic analytics (TA)-based inference attacks, smart home owners are considering deploying virtual private networks (VPNs) to safeguard their smart speakers. In this work, we design a new machine learning (ML) and deep learning (DL)-powered attack framework---VoiceAttack that could still accurately fingerprint voice commands on VPN-encrypted voice speaker network traffic. We evaluate VoiceAttack under 5 different real-world settings using Amazon Alexa and Google Home. Our results show that VoiceAttack could correctly infer voice command sentences with a Matthews Correlation Coefficient (MCC) of 0.68 in a closed-world setting and infer voice command categories with an MCC of 0.84 in an open-world setting by eavesdropping VPN-encrypted network traffic rates. This presents a significant risk to user privacy and security, as it suggests that external on-path attackers could still potentially intercept and decipher users' voice commands despite the VPN encryption. We then further examine the sensitivity of voice speaker commands to VoiceAttack. We find that 134 voice speaker commands are highly vulnerable to VoiceAttack and 3 commands are less sensitive. We also present a proof-of-concept defense approach---VoiceDefense, which could effectively mitigate TA attack accuracy on Amazon Echo and Google Home by $\sim$50\%. 

## Overview
This repository contains the source code for the paper **"VoiceAttack: Fingerprinting Voice Command on VPN-Protected Smart Home Speakers"** presented at the BuildSys 2024 conference. The project aims to demonstrate that machine learning (ML) and deep learning (DL) techniques can be used to accurately fingerprint voice commands in VPN-encrypted smart speaker traffic. The code in this repository implements the different stages of the VoiceAttack framework, including data collection, preprocessing, and multiple classification models for voice command inference.

## Directory Structure
- **data_collection/**
  - `data_collector.py`: Script to capture and save network traffic data.
  - `text2speech.py`: Converts text commands into audio files for traffic collection.

- **data_preprocess/**
  - `combine_csv.py`: Combines multiple CSV files into a single dataset.
  - `data_preprocessor.py`: Preprocesses raw network traffic data.
  - `features_extraction.py`: Extracts relevant features from the processed data.
  - `traffic_filter.py`: Filters the raw traffic data based on predefined rules.
  - `traffic_resample.py`: Resamples the network traffic data at specific time intervals.
  - `traffic_trim.py`: Trims unnecessary parts of the traffic data to refine the dataset.

- **disaggregation/**
  - `disaggregate.py`: Applies disaggregation techniques to separate voice command traffic.
  - `preprocess.py`: Preprocesses data for disaggregation.

- **models/**
  - **dl/**
    - `1D_CNN.py`: Implements a 1D Convolutional Neural Network for voice command classification.
    - `LSTM.py`: Uses LSTM networks for voice command classification.
  
  - **ml/**
    - `knn.ipynb`: Implements K-Nearest Neighbor for voice command inference.
    - `LR.ipynb`: Logistic Regression-based voice command classifier.
    - `navieBayes.ipynb`: Naive Bayes classifier.
    - `svm.ipynb`: Support Vector Machine classifier for command fingerprinting.
    - **tree/**
      - `decisionTree.ipynb`: Decision Tree classifier.
      - `extraTree.ipynb`: Extra Trees classifier.
      - `randomForest.ipynb`: Random Forest classifier.
      - `Xgboost.ipynb`: XGBoost classifier.

## Requirements
- Python 3.x
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook

## Usage
1. **Data Collection**: Use the scripts in the `data_collection/` folder to capture network traffic data.
2. **Preprocessing**: Run the preprocessing scripts in `data_preprocess/` to clean and process the raw data.
3. **Feature Extraction**: Extract features using `features_extraction.py`.
4. **Model Training and Evaluation**: Choose the appropriate model from `models/` and train it using the provided datasets.

