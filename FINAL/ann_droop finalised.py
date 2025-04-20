import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# === Load and Clean Data ===
data_path = r"C:\Users\cscpr\Desktop\SEM2\PROJECTS\EEE\final\SGSMA_Competiton 2024_PMU_DATA\PMU_Data_with_Anomalies and Events\Bus39_Competition_Data.csv"
data_df = pd.read_csv(data_path)

# Filter voltage outliers
low, high = np.percentile(data_df['BUS39_IC_MAG'], [1, 99])
data_df = data_df[(data_df['BUS39_IC_MAG'] >= low) & (data_df['BUS39_IC_MAG'] <= high)].reset_index(drop=True)

# Generate synthetic inputs
data_len = len(data_df)
P = np.linspace(900, 1350, data_len)
Q = np.linspace(400, 850, data_len)

# Define Inputs (P, Q) and Outputs (Voltage, Frequency)
X_data = np.column_stack((P, Q))
V_data = data_df['BUS39_IC_MAG'].values.reshape(-1, 1)
F_data = data_df['BUS39_Freq'].values.reshape(-1, 1)

# === Train-Test Split ===
X_train, X_test, V_train, V_test, F_train, F_test = train_test_split(
    X_data, V_data, F_data, test_size=0.2, random_state=42
)

# === Scaling ===
scaler_X = MinMaxScaler(feature_range=(-1, 1))
scaler_V = MinMaxScaler(feature_range=(-1, 1))
scaler_F = MinMaxScaler(feature_range=(-1, 1))

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

V_train_scaled = scaler_V.fit_transform(V_train)
V_test_scaled = scaler_V.transform(V_test)

F_train_scaled = scaler_F.fit_transform(F_train)
F_test_scaled = scaler_F.transform(F_test)

# Combine scaled outputs
Y_train_scaled = np.hstack((V_train_scaled, F_train_scaled))
Y_test_scaled = np.hstack((V_test_scaled, F_test_scaled))

# Scaled 230V using voltage scaler only
target_voltage_scaled = scaler_V.transform([[230.0]])[0][0]
print(f"Scaled 230V value: {target_voltage_scaled:.4f}")

# === Define Model ===
model = keras.Sequential([
    Dense(512, activation='relu', input_shape=(2,)),
    Dropout(0.4),
    Dense(512, activation='relu'),
    Dropout(0.4),
    Dense(256, activation='tanh'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='linear')
])

# === Custom Loss (strong focus on 230V) ===
def custom_loss(y_true, y_pred):
    v_true, f_true = y_true[:, 0], y_true[:, 1]
    v_pred, f_pred = y_pred[:, 0], y_pred[:, 1]
    v_loss = tf.reduce_mean(tf.square(v_true - v_pred))
    f_loss = tf.reduce_mean(tf.square(f_true - f_pred))
    v_target_penalty = tf.reduce_mean(tf.square(v_pred - target_voltage_scaled))
    return 0.4 * v_loss + 0.2 * f_loss + 1.2 * v_target_penalty  # Strong voltage optimization

# === Compile ===
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss=custom_loss)

# === Train ===
history = model.fit(
    X_train_scaled, Y_train_scaled,
    epochs=300,
    batch_size=256,
    validation_data=(X_test_scaled, Y_test_scaled),
    verbose=1
)

# === Predictions ===
pred_scaled = model.predict(X_test_scaled)
v_pred = scaler_V.inverse_transform(pred_scaled[:, [0]])
f_pred = scaler_F.inverse_transform(pred_scaled[:, [1]])
v_actual = scaler_V.inverse_transform(V_test_scaled)
f_actual = scaler_F.inverse_transform(F_test_scaled)

# === Voltage Plot ===
plt.figure(figsize=(12, 6))
plt.plot(v_actual[:100], label='Actual Voltage', linewidth=2)
plt.plot(v_pred[:100], label='Predicted Voltage', linewidth=2)
plt.axhline(y=230, color='red', linestyle='--', linewidth=2, label='Target: 230V')
plt.xlabel('Sample Index')
plt.ylabel('Voltage (V)')
plt.title('Voltage Prediction')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# === Frequency Plot ===
plt.figure(figsize=(12, 6))
plt.plot(f_actual[:100], label='Actual Frequency', linewidth=2)
plt.plot(f_pred[:100], label='Predicted Frequency', linewidth=2)
plt.axhline(y=60, color='red', linestyle='--', linewidth=2, label='Target: 60Hz')
plt.xlabel('Sample Index')
plt.ylabel('Frequency (Hz)')
plt.title('Frequency Prediction')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# === Loss Plot ===
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# === Print Sample Predictions ===
print(f"\n{'P (W)':>8} {'Q (Var)':>8} {'Actual V':>10} {'Pred V':>10} {'Actual F':>10} {'Pred F':>10}")
print("-" * 60)
for i in range(10):
    print(f"{X_test[i][0]:8.2f} {X_test[i][1]:8.2f} {v_actual[i][0]:10.2f} {v_pred[i][0]:10.2f} {f_actual[i][0]:10.2f} {f_pred[i][0]:10.2f}")
