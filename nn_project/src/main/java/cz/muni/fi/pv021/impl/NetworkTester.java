package cz.muni.fi.pv021.impl;

import cz.muni.fi.pv021.impl.Vectors.Vectors;
import cz.muni.fi.pv021.impl.network.Network;
import cz.muni.fi.pv021.impl.network.Neuron;
import cz.muni.fi.pv021.impl.network.NeuronLayer;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

/**
 * Class representing a test of a trained Network with test input vectors and labels. The class does not test vectors by
 * comparing results with labels, it only writes results to output file.
 */
public class NetworkTester {

    private final Vectors vectors;
    private final String outputName;

    private FileWriter fileWriter;

    private final Network network;

    /**
     * Creates NetworkTester which can read input vectors from given test file and check how well can given
     * Network classify vectors (return the same answer as label).
     *
     * @param vectorTestFile name of file with test vectors
     * @param labelTestFile name of file with test labels
     * @param outputFile name of file for writing results from network
     * @param network network
     * @throws IOException
     */
    public NetworkTester(String vectorTestFile, String labelTestFile, String outputFile, Network network) throws IOException {
        this.network = network;
        vectors = new Vectors(vectorTestFile, labelTestFile);
        outputName = outputFile;
    }

    /**
     * This method tests the network by reading vectors, running them through the network and then writing the results
     * to the output file.
     *
     * @throws IOException
     */
    public void test() throws IOException {
        vectors.readyTestInputs();
        File outputFile;
        outputFile = new File(outputName);
        fileWriter = new FileWriter(outputFile);
        testVectors();
        fileWriter.close();
    }

    /**
     * Reads every vector, tests each of them by calling testVector and then resets network for another run.
     *
     * @throws IOException
     */
    private void testVectors() throws IOException {
        Double[] vector;
        while ((vector = vectors.readVector()) != null){
            testVector(vector);
            network.resetNetwork();
        }
        network.resetNetwork();
    }

    /**
     * This method tests given vector by running it through the network and writing the result to output file.
     *
     * First the method sets vector values as inputs of the first (input) layer.
     *
     * Then for every neuron in every layer except the last, the input of this neuron is sent to corresponding
     * activation function and the result is then multiplied with every weight and added to neuron input of neuron
     * with which this weight is connected.
     *
     * If current neuron is not input neuron, bias is added to neuron input before changing inputs of neurons in the
     * layer above.
     *
     * Bias is added to the last layer of neurons.
     *
     * For every neuron input in the last layer the method then calculates SOFTMAX activated output and chooses the
     * largest as predicted label. Number of this label is then written as prediction to the output file.
     *
     * @param vector vector to be tested
     */
    public void testVector(Double[] vector) throws IOException {
        int vectorIndex = 0;
        for (Neuron neuron : network.layers.get(0).neurons) {
            neuron.input = vector[vectorIndex];
            vectorIndex++;
        }
        for (int i = 0; i < network.layerCount - 1; i++){
            for (Neuron neuron : network.layers.get(i).neurons) {
                if (i != 0) {
                    neuron.input += neuron.gaussBias;
                }
                double output = ActivationFunction.relu(neuron.input);
                for (int j = 0; j < network.layers.get(i+1).neuronCount; j++){
                    network.layers.get(i+1).neurons.get(j).input += neuron.weights[j] * output;
                }
            }
        }
        for (Neuron neuron : network.layers.get(network.layerCount - 1).neurons) {
            neuron.input += neuron.gaussBias;
        }
        NeuronLayer lastLayer = network.layers.get(network.layerCount - 1);
        int count = 0;
        double maxValue = 0;
        int maxLabel = 0;
        double outputValue;
        for (Neuron neuron : lastLayer.neurons) {
            outputValue = ActivationFunction.softmax(lastLayer, neuron.input);
            if (outputValue > maxValue) {
                maxLabel = count;
                maxValue = outputValue;
            }
            count++;
        }
        fileWriter.write(maxLabel + "\n");
    }
}
