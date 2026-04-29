# Neural Network from Scratch in Java

This project implements a multilayer perceptron from scratch in Java and applies it to multiclass image classification on Fashion-MNIST.

The implementation does not use machine learning libraries. Core neural network components such as forward propagation, backpropagation, activation functions, weight initialization, mini-batch training, and prediction generation are implemented manually.

The model achieved `89%` accuracy on the Fashion-MNIST test set.

## Model

The network is a feed-forward neural network with the following architecture:

```text
784 input neurons → 128 hidden neurons → 64 hidden neurons → 10 output neurons
```

The input dimension corresponds to flattened `28 × 28` Fashion-MNIST images. The output layer contains 10 neurons, one for each class.

## Implementation details

The project includes:

- manual implementation of neurons, layers, and network structure,
- ReLU activation in hidden layers,
- softmax activation in the output layer,
- manually implemented backpropagation,
- mini-batch gradient descent,
- Gaussian weight initialization,
- input normalization by scaling pixel values to `[0, 1]`,
- shuffling of training data between epochs,
- prediction export for train and test data.

## Main hyperparameters

The main training parameters are configured in `Main.java`:

```text
Architecture: 784 → 128 → 64 → 10
Epochs: 8
Batch size: 32
Learning rate: 0.001
Random seed: 0
```

In the code, the learning rate is represented as `-0.001` because the update rule uses `+= learningRate * gradient`.

## Project structure

```text
.
├── README.md
└── nn_project/
    ├── pom.xml
    ├── run.sh
    ├── data/                       # expected location of Fashion-MNIST CSV files
    └── src/
        └── main/
            └── java/
                └── cz/muni/fi/pv021/impl/
                    ├── Main.java
                    ├── ActivationFunction.java
                    ├── GradientComputer.java
                    ├── NetworkTrainer.java
                    ├── NetworkTester.java
                    ├── LossFunctions/
                    │   └── CategoricalCrossEntropy.java
                    ├── Vectors/
                    │   └── Vectors.java
                    └── network/
                        ├── Network.java
                        ├── Neuron.java
                        └── NeuronLayer.java
```

## Data format

The dataset files are not included in this repository because of file-size constraints.

The code expects the Fashion-MNIST CSV files to be placed in a `data/` folder with the following filenames:

```text
data/fashion_mnist_train_vectors.csv
data/fashion_mnist_train_labels.csv
data/fashion_mnist_test_vectors.csv
data/fashion_mnist_test_labels.csv
```

Each vector file contains flattened `28 × 28` grayscale images, represented as 784 pixel values per sample. Each label file contains one integer class label per line.

Pixel values are normalized in the code by dividing by `255`.

## Usage

From the repository root, enter the project folder:

```bash
cd nn_project
```

Then run the provided script:

```bash
./run.sh
```

The script builds the Maven project and runs the main class:

```bash
mvn clean install
java -cp target/neuralNetwork-1.0-SNAPSHOT.jar cz.muni.fi.pv021.impl.Main
```

Note: `run.sh` contains `module add maven-3.9.0`, which was required in the original server environment. If the `module` command is not available on your machine, remove that line or run the Maven commands manually.

## Outputs

After training, the program writes predictions to the project root:

```text
train_predictions.csv
test_predictions.csv
```

Each prediction file contains one predicted class label per line.

## Results

The multilayer perceptron achieved `89%` accuracy on the Fashion-MNIST test set.

## Notes

This project was created to understand and implement the mechanics of neural networks from scratch.
