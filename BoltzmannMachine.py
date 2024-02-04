import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

class BoltzmannMachine:
    def __init__(self, num_visible, learning_rate):
        self.num_visible = num_visible
        self.learning_rate = np.float64(learning_rate)
        # Initialize couplers in the range [-0.1, 0.1]
        self.couplers = np.float64(np.random.uniform(-0.1, 0.1, size=(num_visible, num_visible)))
        self.kl_divergence_values = []

    def train(self, data, epochs, verbose=False, plot_file=None):
        num_samples, num_visible = data.shape
        best_kl_divergence = float('inf')  

        for epoch in range(epochs):
            # Inside the training loop
            for sample in data:
                positive_phase = np.outer(sample, sample)

                hidden_probabilities = self.sigmoid(np.dot(sample, self.couplers))
                hidden_states = np.sign(hidden_probabilities - 0.5)

                visible_probabilities = self.sigmoid(np.dot(self.couplers, hidden_states))
                negative_phase = np.outer(visible_probabilities, hidden_states)

                self.couplers = np.sign(self.couplers + self.learning_rate * (positive_phase - negative_phase))

                # Compute KL Divergence and track its value
                kl_divergence = self.compute_kl_divergence(data)
                self.kl_divergence_values.append(kl_divergence)

                if kl_divergence < best_kl_divergence:
                    best_kl_divergence = kl_divergence
                else:
                    print(f"Converged at epoch {epoch+1}, best KL Divergence: {best_kl_divergence}")
                    break

            # Plot and save the KL divergence values
            if verbose and plot_file is not None:
                self.plot_kl_divergence(epoch, plot_file)

    def plot_kl_divergence(self, epoch, plot_file):
        x_values = range(epoch + 1)
        y_values = self.kl_divergence_values[:epoch + 1]
        plt.plot(x_values, y_values, label='KL Divergence')
        plt.xlabel('Epoch')
        plt.ylabel('KL Divergence')
        plt.title('KL Divergence vs. Training Epochs')
        plt.legend()
        plt.savefig(plot_file)
        plt.close()

    def compute_kl_divergence(self, data):
        num_samples, num_visible = data.shape

        data_energy = -np.sum(np.dot(data, self.couplers) * data, axis=(0, 1))

        model_probabilities = self.sigmoid(np.dot(data, self.couplers.T) - np.max(np.dot(data, self.couplers.T)))
        data_probabilities = data / (np.sum(data, axis=1, keepdims=True) + 1e-8)

        # Avoid NaN values in the logarithmic terms
        model_probabilities[model_probabilities <= 0] = 1e-8
        data_probabilities[data_probabilities <= 0] = 1e-8

        kl_divergence = np.sum(data_probabilities * (np.log(data_probabilities) - np.log(model_probabilities)))
    
        return kl_divergence



    def sigmoid(self, x):
        # Use the logistic sigmoid function for stability
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def predict_couplers(self):
        coupler_predictions = {}

        for i in range(self.num_visible):
            j_positive = (i + 1) % self.num_visible  # Index of the positive coupler
            j_negative = (i - 1) % self.num_visible  # Index of the negative coupler

            coupler_predictions[(i, j_positive)] = int(np.sign(self.couplers[i, j_positive]))
            coupler_predictions[(i, j_negative)] = int(np.sign(self.couplers[i, j_negative]))

        return coupler_predictions
    
    @staticmethod
    def save_predictions(predictions, output_file):
        try:
            with open(output_file, 'w') as file:
                file.write("{\n")
                for (i, j), value in predictions.items():
                    file.write(f"  ({i}, {j}): {value},\n")
                file.write("}\n")
        except Exception as e:
            print(f"Error: Unable to save predictions to {output_file}. {e}")
