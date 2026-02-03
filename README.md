Predicting Residential Electricity Consumption Using CNN-BiLSTM-Self-Attention
📌 Overview

Accurate electricity consumption forecasting is essential for efficient energy management, demand–supply balancing, and smart grid planning. Traditional statistical methods struggle to capture the complex temporal patterns and non-linear dependencies present in real-world electricity consumption data.

This project proposes a deep learning-based framework for residential electricity load prediction using Convolutional Neural Networks (CNN), Bidirectional Long Short-Term Memory (BiLSTM), and a Self-Attention mechanism. The model effectively learns both local features and long-term temporal dependencies, while selectively focusing on the most relevant time steps.

🎯 Objectives

Predict short-term residential electricity consumption accurately

Capture temporal dependencies in time-series electricity data

Reduce noise and improve generalization using attention mechanisms

Compare a baseline CNN-LSTM Autoencoder with an advanced CNN-BiLSTM-Self-Attention model

🧠 Models Implemented
1️⃣ Baseline Model: CNN-LSTM Autoencoder

The initial model uses:

CNN layers for feature extraction

LSTM encoder–decoder (autoencoder) for temporal sequence reconstruction

This model learns compressed representations of historical electricity consumption and reconstructs future values.

Limitations:

Sensitive to local fluctuations

Can overreact to noise

Limited ability to model long-range bidirectional dependencies

2️⃣ Proposed Model: CNN-BiLSTM-Self-Attention (Final Model)

The enhanced model replaces the autoencoder with:

Bidirectional LSTM (BiLSTM) to learn past and future context simultaneously

Self-Attention mechanism to emphasize important time steps and suppress noise

Architecture:
Input Sequence
 → CNN (feature extraction)
 → CNN
 → MaxPooling
 → BiLSTM (bidirectional temporal learning)
 → Self-Attention (important time-step weighting)
 → Dense layer (prediction)


Advantages:

Better temporal understanding

Improved generalization

More stable predictions

Reduced sensitivity to noisy spikes

📂 Project Structure
├── dataset/
│   └── IHEPC.csv
├── model/
│   ├── model.h5                     # CNN-LSTM Autoencoder model
│   └── model_CNN_BiLSTM_SA.h5       # CNN-BiLSTM-SA trained model
├── results/
│   ├── actual_predicted.png
│   └── actual_predicted_CNN_BiLSTM_SA.png
├── Training.py                      # Baseline CNN-LSTM Autoencoder
├── Training_CNN_BiLSTM_SA.py        # Proposed CNN-BiLSTM-SA model
├── Testing.py
└── README.md


📊 Dataset

Source: Residential electricity consumption dataset

Features used:

datetime

Global_active_power

Missing values are handled using mean imputation

Data is normalized using Min-Max scaling

🔄 Data Preprocessing

Load dataset and parse timestamps

Handle missing values

Normalize electricity consumption values

Convert time series into sliding windows

Input window size: 8 time steps

Output prediction: next 4 time steps

Split data into training and testing sets

⚙️ Model Training

Optimizer: Adam

Loss function: Mean Squared Error (MSE)

Epochs: 10

Batch size: 32

Validation split: 20%

📈 Evaluation Metrics

The models are evaluated using:

MSE (Mean Squared Error)

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

MAPE (Mean Absolute Percentage Error)

Note: MAPE is sensitive to near-zero electricity values and may not always reflect true predictive performance.

📊 Results & Comparison
🔹 CNN-LSTM Autoencoder (Baseline)

Produces sharper predictions

More sensitive to short-term fluctuations

Slightly noisier output

🔹 CNN-BiLSTM-Self-Attention (Proposed)

Produces smoother and more stable predictions

Lower RMSE and MSE

Better generalization

Attention mechanism suppresses irrelevant noise

Key Insight:
Although peak amplitudes are slightly reduced, the CNN-BiLSTM-Self-Attention model demonstrates improved robustness and lower prediction error.

🖼️ Output Visualization

The results folder contains plots comparing actual vs predicted electricity consumption for both models:

actual_predicted.png (baseline)

actual_predicted_CNN_BiLSTM_SA.png (proposed)

🧪 How to Run the Project
▶️ Train Baseline Model
python Training.py

▶️ Train Proposed CNN-BiLSTM-SA Model
python Training_CNN_BiLSTM_SA.py

🚀 Future Work

Incorporate weather and occupancy features

Extend prediction horizon

Apply transformer-based architectures

Optimize attention mechanisms for peak load prediction

🎓 Conclusion

This project demonstrates that integrating Bidirectional LSTM and Self-Attention significantly enhances electricity load forecasting performance compared to a standard CNN-LSTM autoencoder. The proposed CNN-BiLSTM-Self-Attention framework effectively captures complex temporal dependencies and offers a robust solution for real-world residential energy forecasting.

👤 Author

Annavarapu Rohith
📧 rohithannavarapu7@gmail.com
