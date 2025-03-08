import numpy as np
import matplotlib.pyplot as plt

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
def simulate_microgrid(generators, loads):
    time = np.arange(0, SIMULATION_TIME, TIME_STEP)
    frequencies = []
    voltages = []

    for t in time:
        # Update generator outputs based on droop control
        frequency, voltage = power_flow(generators, loads)
        for gen in generators:
            gen.update(frequency, voltage)

        # Record system frequency and voltage
        frequencies.append(frequency)
        voltages.append(voltage)

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

    # Run simulation
    time, frequencies, voltages = simulate_microgrid(generators, loads)

    # Plot results
    plot_results(time, frequencies, voltages)

if _name_ == "_main_":
    main()