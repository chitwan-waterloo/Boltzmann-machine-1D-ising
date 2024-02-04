#main.py
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.stats import entropy
from BoltzmannMachine import BoltzmannMachine
    
def load_data(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None

    spin_configurations = []
    for line in lines:
        spin_configuration = [1 if s == '+' else -1 for s in line.strip()]
        spin_configurations.append(spin_configuration)

    if not spin_configurations:
        print("Error: No valid data found in the file.")
        return None

    return np.array(spin_configurations)

def main():
    parser = argparse.ArgumentParser(description='Fully visible Boltzmann machine for 1-D Ising model')
    parser.add_argument('data_file', type=str, help='Path to the input data file')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')
    parser.add_argument('--output_file', type=str, default='output.txt', help='Path to the output file for predictions')
    args = parser.parse_args()

    # Load data
    data = load_data(args.data_file)
    if data is None:
        return
    
    learning_rate = args.learning_rate
    epochs = args.epochs
    verbose = args.verbose

    # Initialize Boltzmann Machine with a lower learning rate
    model = BoltzmannMachine(len(data[0]), learning_rate)

    # Train the model with fewer epochs initially
    model.train(data, epochs, verbose, plot_file='kl_divergence_plot.png')
    
    # Predict couplers
    predictions = model.predict_couplers()

    # Save predictions
    BoltzmannMachine.save_predictions(predictions, args.output_file)

if __name__ == "__main__":
    main()
    