# LiDAR-Based Steering Angle Prediction

## Overview

This project demonstrates the use of a deep neural network to predict a vehicle's steering angle based solely on raw LiDAR sensor data. The model is built with PyTorch and trained on the challenging MIT-Intel dataset, which contains synchronized odometry and laser scan information from a mobile robot.

The core of this project involves parsing raw sensor logs, aligning time-series data, preprocessing LiDAR scans, and training a regularized neural network to learn the complex relationship between environmental perception and vehicle movement.

🚀 **Core Achievement:** Successfully trained a model to predict steering commands from 2D LiDAR scans, a fundamental task in autonomous navigation.

---

## Features

-   **Data Parsing:** A custom parser to extract and synchronize odometry and LiDAR data from the raw `intel.log` file.
-   **Data Preprocessing:** LiDAR data is normalized and downsampled to create a concise and effective input for the neural network.
-   **PyTorch Model:** A `RegularizedLiDARNet` model featuring multiple linear layers, ReLU activations, and Dropout for regularization to prevent overfitting.
-   **Hyperparameter Tuning:** A systematic grid search was performed to find the optimal learning rate, batch size, and dropout rate for the best model performance.
-   **Model Export:** The final trained model is provided in both PyTorch (`.pth`) and ONNX (`.onnx`) formats for cross-platform compatibility and optimized inference.

---

## Tech Stack & Dependencies

This project is built primarily in Python and relies on the following libraries:

-   **Core Libraries:**
    -   `numpy`: For numerical operations and data manipulation.
    -   `torch`: For building and training the neural network.
    -   `scikit-learn`: For splitting the data into training and testing sets.
-   **Data Parsing & Visualization:**
    -   `matplotlib`: For visualizing the results.
-   **(Optional) For ONNX Model Inference:**
    -   `onnx`: To work with the exported ONNX model.
    -   `onnxruntime`: For running inference with the ONNX model.

You can install the primary dependencies with pip:
- pip install numpy torch scikit-learn matplotlib

---

## Project Structure
- data/
    - intel.log         # The raw MIT-Intel dataset log file
- mit-intel-ds.ipynb    # Jupyter Notebook with the full data processing and training pipeline
- lidar_model.pth       # The final trained PyTorch model weights
- final_lidar_model.onnx # The final model exported to ONNX format

---

## How It Works

The process is detailed in the `mit-intel-ds.ipynb` notebook and can be summarized in these steps:

1.  **Parsing:** The raw `intel.log` file is parsed to separate odometry (x, y, theta) and LiDAR (180 laser scan readings) data streams, each with its own timestamp.
2.  **Alignment:** The two data streams are aligned based on their timestamps to ensure that each LiDAR scan is matched with the correct steering angle (theta) from the odometry data.
3.  **Preprocessing:** To make the data suitable for the model, the LiDAR scans are processed by:
    -   Clipping the range to a maximum of 10.0 meters.
    -   Normalizing the distances to a range of [0, 1].
    -   Downsampling the 180 points to a more manageable 90 points.
4.  **Training:** The processed LiDAR data (input) and the corresponding steering angles (output) are used to train the `RegularizedLiDARNet`. A hyperparameter search was conducted to find the best model configuration, resulting in a final test loss of **2.9374**.
5.  **Evaluation:** The model's predictions on the test set are plotted against the true steering angles to visually assess its performance.

---
