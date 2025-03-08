import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.linalg import expm

# Constants
NOMINAL_FREQUENCY = 50.0  # Hz
NOMINAL_VOLTAGE = 230.0  # Volts
SIMULATION_TIME = 10.0  # seconds
TIME_STEP = 0.01  # seconds

# Droop Coefficients
KP = 0.01  # Frequency droop coefficient (Hz/W)
KQ = 0.001  # Voltage droop coefficient (V/VAr)

# Generator and Load Parameters
class Generator:
    def _init_(self, P0, Q0, kp, kq):
        self.P0 = P0  # Nominal active power (W)
        self.Q0 = Q0  # Nominal reactive power (VAr)
        self.kp = kp  # Frequency droop coefficient
        self.kq = kq  # Voltage droop coefficient
        self.P = P0  # Actual active power
        self.Q = Q0  # Actual reactive power

    def update(self, frequency, voltage):
        # Update power output based on droop control
        self.P = self.P0 - (frequency - NOMINAL_FREQUENCY) / self.kp
        self.Q = self.Q0 - (voltage - NOMINAL_VOLTAGE) / self.kq

class Load:
    def _init_(self, P_load, Q_load):
        self.P_load = P_load  # Active power demand (W)
        self.Q_load = Q_load  # Reactive power demand (VAr)

# ANN Model for Power Prediction
class ANNPowerPredictor:
    def _init_(self):
        self.model = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
        self.scaler = StandardScaler()

    def train(self, X, y):
        # Normalize data
        X_scaled = self.scaler.fit_transform(X)
        # Train the model
        self.model.fit(X_scaled, y)

    def predict(self, X):
        # Normalize data and make predictions
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

# State-Space Model for Microgrid
class StateSpaceModel:
    def _init_(self, A, B, C, D):
        self.A = A  # State matrix
        self.B = B  # Input matrix
        self.C = C  # Output matrix
        self.D = D  # Feedthrough matrix

    def update(self, x, u):
        # State-space equation: x_dot = A*x + B*u
        x_dot = np.dot(self.A, x) + np.dot(self.B, u)
        return x_dot

    def output(self, x, u):
        # Output equation: y = C*x + D*u
        y = np.dot(self.C, x) + np.dot(self.D, u)
        return y

# Power Flow Solver
def power_flow(generators, loads):
    total_P_gen = sum(gen.P for gen in generators)
    total_Q_gen = sum(gen.Q for gen in generators)
    total_P_load = sum(load.P_load for load in loads)
    total_Q_load = sum(load.Q_load for load in loads)

    # Calculate frequency and voltage deviations
    delta_P = total_P_gen - total_P_load
    delta_Q = total_Q_gen - total_Q_load

    frequency = NOMINAL_FREQUENCY - KP * delta_P
    voltage = NOMINAL_VOLTAGE - KQ * delta_Q

    return frequency, voltage

# Simulation
def simulate_microgrid(generators, loads, ann_predictor, state_space_model):
    time = np.arange(0, SIMULATION_TIME, TIME_STEP)
    frequencies = []
    voltages = []

    # State vector: [frequency, voltage, P_gen1, P_gen2, Q_gen1, Q_gen2]
    x = np.array([NOMINAL_FREQUENCY, NOMINAL_VOLTAGE, generators[0].P0, generators[1].P0, generators[0].Q0, generators[1].Q0])

    # Historical data for ANN training
    X_train = []
    y_train = []

    for t in time:
        # Update generator outputs based on droop control
        frequency, voltage = power_flow(generators, loads)
        for gen in generators:
            gen.update(frequency, voltage)

        # Record system frequency and voltage
        frequencies.append(frequency)
        voltages.append(voltage)

        # Prepare data for ANN training
        X_train.append([frequency, voltage, sum(load.P_load for load in loads), sum(load.Q_load for load in loads)])
        y_train.append([gen.P for gen in generators])

        # Update state-space model
        u = np.array([sum(load.P_load for load in loads), sum(load.Q_load for load in loads)])  # Input vector
        x_dot = state_space_model.update(x, u)
        x = x + x_dot * TIME_STEP  # Euler integration

    # Train ANN
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    ann_predictor.train(X_train, y_train)

    return time, frequencies, voltages

# Visualization
def plot_results(time, frequencies, voltages):
    plt.figure(figsize=(12, 6))

    # Plot Frequency
    plt.subplot(2, 1, 1)
    plt.plot(time, frequencies, label="System Frequency (Hz)")
    plt.axhline(y=NOMINAL_FREQUENCY, color="r", linestyle="--", label="Nominal Frequency")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Microgrid Frequency Over Time")
    plt.legend()
    plt.grid()

    # Plot Voltage
    plt.subplot(2, 1, 2)
    plt.plot(time, voltages, label="System Voltage (V)")
    plt.axhline(y=NOMINAL_VOLTAGE, color="r", linestyle="--", label="Nominal Voltage")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title("Microgrid Voltage Over Time")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# Main Function
def main():
    # Define generators and loads
    generators = [
        Generator(P0=1000, Q0=500, kp=KP, kq=KQ),
        Generator(P0=1500, Q0=800, kp=KP, kq=KQ)
    ]
    loads = [
        Load(P_load=2000, Q_load=1000),
        Load(P_load=500, Q_load=300)
    ]

    # Initialize ANN Power Predictor
    ann_predictor = ANNPowerPredictor()

    # Define state-space matrices (example values)
    A = np.array([[-0.1, 0, 0, 0, 0, 0],
                  [0, -0.1, 0, 0, 0, 0],
                  [0, 0, -0.2, 0, 0, 0],
                  [0, 0, 0, -0.2, 0, 0],
                  [0, 0, 0, 0, -0.2, 0],
                  [0, 0, 0, 0, 0, -0.2]])
    B = np.array([[1, 0],
                  [0, 1],
                  [1, 0],
                  [0, 1],
                  [1, 0],
                  [0, 1]])
    C = np.eye(6)  # Output matrix (identity for full state observation)
    D = np.zeros((6, 2))  # Feedthrough matrix

    # Initialize State-Space Model
    state_space_model = StateSpaceModel(A, B, C, D)

    # Run simulation
    time, frequencies, voltages = simulate_microgrid(generators, loads, ann_predictor, state_space_model)

    # Plot results
    plot_results(time, frequencies, voltages)

if __name__ == "__main__":
    main()