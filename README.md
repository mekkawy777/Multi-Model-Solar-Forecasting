A Smart System for Predicting Solar Energy Production

This project is a comprehensive machine learning pipeline designed to forecast solar power generation using multiple regression algorithms. The system automates the process of data cleaning, feature engineering (especially for temporal data), and performance benchmarking across different solar plant datasets.

Key Features

 Multi-Model Architecture: Implements four distinct regression models: SVR, MLP, KNN, and LSTM.
 Smart Preprocessing:
 Automatic conversion of object columns to numeric.
 Advanced datetime feature engineering (extracting year, month, day, hour, etc.).
 Robust handling of missing values and data scaling.


 Scalability: Automatically switches to `LinearSVR` for large datasets to ensure computational efficiency.
 Deep Learning Integration: Features a Long Short-Term Memory (LSTM) network built with TensorFlow for capturing sequential patterns.
 Automated Evaluation: Generates detailed metrics including MSE, RMSE, MAE, and R² for all models.

 🛠️ Models Comparison

The system evaluates the following algorithms:

1. SVR (Support Vector Regression): Uses RBF kernel for small datasets and falls back to LinearSVR for efficiency on larger ones.
2. MLPRegressor (Neural Network): A multi-layer perceptron with early stopping to prevent overfitting.
3. KNN (K-Nearest Neighbors): Utilizes distance-based weighting for accurate regression.
4. LSTM (Deep Learning): A recurrent architecture designed to handle tabular data reshaped for temporal analysis.

 📊 Dataset Compatibility

The system is pre-configured to handle several standard solar datasets:

 Solar Power Plant Data
 Plant 1 & Plant 2 Generation Data
 SPG (Solar Power Generation) dataset

 💻 Installation & Usage

 Prerequisites

Ensure you have Python installed, then install the required libraries:

```
pip install numpy pandas scikit-learn tensorflow

```

 Running the System

1. Place your `.csv` data files in the same directory as the script.
2. Run the main execution script:

```
python run_4_models.py

```

 📈 Output

The system will display formatted result tables for each model and save a comprehensive report to `results_4_models.csv`.


 📝 License

This project is open-source. Feel free to use and contribute!



