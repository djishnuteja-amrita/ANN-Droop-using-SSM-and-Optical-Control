import numpy as np
import tensorflow as tf
from scipy.signal import StateSpace, lsim
import matplotlib.pyplot as plt

# Define system parameters for microgrid
dt = 0.01  # Time step
T = 10     # Total time in seconds
time = np.arange(0, T, dt)  # Time vector

# Define system component parameters
Rf = 0.05  # Filter resistance
Lf = 1.5e-3  # Filter inductance
Cf = 1500e-6  # Filter capacitance
Rc = 0.11  # Coupling resistance
Lc = 1.25e-3  # Coupling inductance
Cc = 10e-6  # Coupling capacitance

# Define state-space matrices for microgrid dynamic model
A = np.array([
    [0, 1, 0, 0, 0, 0],
    [-1, -0.5, 0, 0, 0, 0],
    [0, 0, -Rf/Lf, -1/Lf, 0, 0],
    [0, 0, 1/Cf, 0, -1/Cf, 0],
    [0, 0, 0, 1/Lc, -Rc/Lc, -1/Lc],
    [0, 0, 0, 0, 1/Cc, 0]
])

B = np.array([
    [0],
    [1],
    [1/Lf],
    [0],
    [0],
    [0]
])

C = np.array([[1, 0, 0, 0, 0, 0]])
D = np.array([[0]])

# Create state-space system model
system = StateSpace(A, B, C, D)

# Generate input voltage signal (simulating grid fluctuations)
input_signal = np.sin(0.5 * time) + 0.1 * np.random.randn(len(time))

# Simulate system response
_, y, x = lsim(system, U=input_signal, T=time)

# Prepare training data for ANN
X_train = np.column_stack((x[:-1, 0], x[:-1, 1], input_signal[:-1]))  # Feature set
y_train = x[1:, 0]  # Target: next time step prediction

# Define ANN model for predicting state variable
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Simulate system response without ANN
_, output_voltage_without_ann, _ = lsim(system, U=input_signal, T=time)

# Simulate system response with ANN prediction
output_voltage_with_ann = []
for t in range(len(time)):
    predicted_voltage = model.predict(np.array([[output_voltage_without_ann[t-1] if t > 0 else 0, 0, input_signal[t]]]))
    output_voltage_with_ann.append(predicted_voltage[0][0])

# Compile model
model.compile(optimizer='adam', loss='mse')

# Train ANN model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Define droop control function
def droop_control(frequency, power_demand):
    """ Simulates droop control behavior based on ANN prediction. """
    predicted_state = model.predict(np.array([[frequency, power_demand, 0]]))
    droop_coefficient = -0.05  # Example coefficient
    return predicted_state[0][0] + droop_coefficient * power_demand

# Implement optical control mechanism
def optical_control(power_demand):
    """ Simulates optical communication-based control adjustment. """
    if power_demand > 0.8:
        return -0.1  # Reduce load
    elif power_demand < 0.2:
        return 0.1   # Increase load
    return 0

# Simulate microgrid operation
frequency = 50  # Initial frequency
power_demand = 0.5  # Initial power demand

# Storage arrays for results
frequency_history = []
power_demand_history = []
input_signal_history = []
state_variable_history = []

for t in range(len(time)):
    # Compute new frequency using droop control
    frequency = droop_control(frequency, power_demand)
    
    # Apply optical control adjustment
    power_adjustment = optical_control(power_demand)
    power_demand += power_adjustment
    
    # Store results
    frequency_history.append(frequency)
    power_demand_history.append(power_demand)
    input_signal_history.append(input_signal[t])
    state_variable_history.append(x[t, 0])

# Plot results
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(time, frequency_history, label='Frequency')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time, power_demand_history, label='Power Demand', color='r')
plt.xlabel('Time (s)')
plt.ylabel('Power Demand')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time, input_signal_history, label='Input Signal', color='g')
plt.xlabel('Time (s)')
plt.ylabel('Input Voltage')
plt.legend()

plt.figure(figsize=(12, 4))
plt.plot(time, state_variable_history, label='State Variable', color='m')
plt.xlabel('Time (s)')
plt.ylabel('State Variable')
plt.legend()
plt.show()
# Plot output voltage with and without ANN
plt.figure(figsize=(12, 6))
plt.plot(time, output_voltage_without_ann, label='Without ANN', linestyle='dashed', color='r')
plt.plot(time, output_voltage_with_ann, label='With ANN', linestyle='solid', color='b')
plt.xlabel('Time (s)')
plt.ylabel('Output Voltage')
plt.title('Comparison of Output Voltage with and without ANN')
plt.legend()
plt.show()

# Plot ANN training performance
history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.2)
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('ANN Training Performance')
plt.legend()
plt.show()
