# Boltzmann Machine for 1-D Ising Model

This project implements and trains a fully visible Boltzmann machine on data generated from a 1-D classical Ising chain. The goal is to predict the model couplers in the absence of prior knowledge of the coupler values, relying solely on the model structure and the training dataset.

## Usage

To run the code, use the following command:

```bash
python main.py data/in.txt
```

Additional hyperparameters can be specified (test case used here):

```bash
python main.py data/in.txt --learning_rate 0.1 --epochs 100 --verbose
```

For a complete list of available options, use the help command:

```bash
python main.py --help
```

Certainly! Here's an overview of the functions in `main.py` and `BoltzmannMachine.py` that you can include in your `README.md` file:

### main.py

#### `load_data(file_path)`

- **Description**: Loads data from the specified file path, interpreting spin configurations from the file and converting them into a numpy array.

#### `main()`

- **Description**: The main function that orchestrates the entire process. It parses command-line arguments, loads data, initializes the Boltzmann Machine, trains the model, predicts couplers, and saves the predictions.

### BoltzmannMachine.py

#### `__init__(self, num_visible, learning_rate)`

- **Description**: Initializes the Boltzmann Machine with the given number of visible nodes and learning rate. Randomly initializes couplers in the range [-0.1, 0.1].

#### `train(self, data, epochs, verbose=False, plot_file=None)`

- **Description**: Trains the Boltzmann Machine using the training data for the specified number of epochs. Updates couplers using the positive and negative phases. Optionally, tracks and saves the KL divergence values.

#### `plot_kl_divergence(self, epoch, plot_file)`

- **Description**: Plots and saves the KL divergence values versus training epochs.

#### `compute_kl_divergence(self, data)`

- **Description**: Computes the Kullback-Leibler divergence between the data distribution and the model distribution.

#### `sigmoid(self, x)`

- **Description**: Applies the logistic sigmoid function to the given input for stability.

#### `predict_couplers(self)`

- **Description**: Predicts the coupler values for the 1-D chain and returns them in a dictionary.

#### `save_predictions(predictions, output_file)`

- **Description**: Saves the predicted coupler values to the specified output file in a readable format.

Feel free to adapt and expand these descriptions based on the specifics of your implementation. Additionally, you might want to include information on the parameters each function takes and the format of the input and output where relevant.

## Input File

The input file contains training data generated from a 1-D closed Ising chain. The spin configurations are provided in a fixed order, representing the 1-D chain structure. Each row in the file corresponds to the size of the model, with spin configurations in the order `s_0 s_1 s_2 ... s_{N-1}`. The model is a 1-D closed loop, and all coupler values are either +1 or -1.

## Output File

The program outputs its best guess of the correct values of all couplers in the 1-D chain from which the training dataset was generated. The output is a dictionary of couplers, where keys are pairs of indices, and values are the predicted coupler values.

For example:

```python
{(0, 1): -1, (1, 2): 1, (2, 3): 1, (3, 0): 1}
```

## Verbose Mode

In verbose mode, the program tracks the Kullback-Leibler (KL) divergence of the training dataset with respect to the generative model during training. A plot of KL divergence values versus training epochs is saved as `kl_divergence_plot.png`.

## Additional Remarks

- The code is designed to work with input files generated for any choice of system size `N`.
- Techniques to ensure model stability and prevent parameter divergence or NaN values have been implemented.


## Notes

- The last commit to the master branch of your repository before the deadline is used for assessment.
- Maintain a documented history of development in your branch history.
- GitHub's timestamp serves as a strict deadline for the acceptance of your work.