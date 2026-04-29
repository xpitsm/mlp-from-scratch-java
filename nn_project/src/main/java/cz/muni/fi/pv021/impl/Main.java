package cz.muni.fi.pv021.impl;

import cz.muni.fi.pv021.impl.network.Network;
import cz.muni.fi.pv021.impl.network.Neuron;
import cz.muni.fi.pv021.impl.network.NeuronLayer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static cz.muni.fi.pv021.impl.Vectors.Vectors.VECTOR_DIMENSION;


/**
 * Main class serves as an entry point for our neural network project.
 * Configures network parameters, initializes the network, and manages the training and testing processes.
 */

public class Main {

    public static Random r = new Random(0); // Random number generator with seed value 0

    public static double learningRate = -0.001; // Learning rate for the network

    public static int BATCH_SIZE = 32; // Size of the batch

    public static List<Integer> neuronCounts = new ArrayList<>(Arrays.asList(VECTOR_DIMENSION, 128, 64, 10, 0));  // Neuron counts in each layer of the network

    public static int layerCount = neuronCounts.size() - 1; // Number of layers in the network


    public static int epochs = 8; // Number of epochs

    /**
     * The main method first creates a new network with layerCount number of layers and neuronCounts number of neurons
     * in corresponding layers and then trains and tests this network in given number of epochs.
     *
     * @param args command-line arguments
     * @throws IOException IOException If an I/O error occurs
     */
    public static void main(String[] args) throws IOException {
        Network network = createNetwork();
        NetworkTrainer trainer = new NetworkTrainer("data/fashion_mnist_train_vectors.csv",
                "data/fashion_mnist_train_labels.csv", network);
        NetworkTester testerTS = new NetworkTester("data/fashion_mnist_train_vectors.csv",
                "data/fashion_mnist_train_labels.csv", "train_predictions.csv", network);
        NetworkTester tester = new NetworkTester("data/fashion_mnist_test_vectors.csv",
                "data/fashion_mnist_test_labels.csv", "test_predictions.csv", network);
        for (int i = 0; i < epochs; i++){
            trainer.train();
            if (i == epochs - 1) {
                testerTS.test();
                tester.test();
            }
        }
    }

    /**
     * Creates a new network with layerCount layers and neuronCounts neurons in corresponding layers.
     *
     * @return a new Network instance
     */
    public static Network createNetwork(){
        List<NeuronLayer> layers = new ArrayList<>(layerCount);
        for (int i = 0; i < layerCount; i++){
            List<Neuron> neurons = new ArrayList<>(neuronCounts.get(i));
            for (int j = 0; j < neuronCounts.get(i); j++) {
                neurons.add(new Neuron(neuronCounts.get(i + 1), i, neuronCounts.get(i)));
            }
            layers.add(new NeuronLayer(neurons));
        }
        return new Network(layers);
    }
}