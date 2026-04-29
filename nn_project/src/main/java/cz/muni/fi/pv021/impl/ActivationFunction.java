package cz.muni.fi.pv021.impl;

import cz.muni.fi.pv021.impl.network.Neuron;
import cz.muni.fi.pv021.impl.network.NeuronLayer;

/**
 * This class provides various activation functions and their derivation used within the neural cz.muni.fi.pv021.impl.network.
 */

public class ActivationFunction {

    /**
     * Applies Rectified Linear Unit (ReLU) activation function.
     *
     * @param innerPotential is the inner potential of neuron on which we want to apply activation function.
     * @return the output after applying the ReLU activation function.
     */
    public static Double relu(Double innerPotential) {
        return Math.max(innerPotential, 0);
    }

    /**
     * Computes the derivative of the ReLU activation function.
     *
     * @param innerPotential is the inner potential of neuron on which we want to apply derivative of ReLU.
     * @return The derivative of the ReLU activation function.
     */
    public static double reluDerivation(Double innerPotential) {
        return innerPotential >= 0 ? 1 : 0;
    }

    /**
     * Applies the Sigmoid activation function.
     *
     * @param innerPotential is the inner potential of neuron on which we want to apply Sigmoid activation function.
     * @return the output after applying the Sigmoid activation function.
     */
    public static double sigmoid(Double innerPotential) {
        return Math.exp(innerPotential) / (Math.exp(innerPotential) + 1);
    }

    /**
     * Computes the derivative of the Sigmoid activation function.
     *
     * @param innerPotential is the inner potential of neuron on which we want to apply derivative of Sigmoid function.
     * @return the derivative of the Sigmoid activation function.
     */
    public static double sigmoidDerivation(Double innerPotential) {
        double currInnerPotential = sigmoid(innerPotential);
        return currInnerPotential * (1 - currInnerPotential);
    }

    /**
     * Applies the Softmax activation function on the last layer.
     *
     * @param innerPotential inner potential of the ith (current) neuron.
     * @param lastLayer layer containing neurons from lastLayer.
     * @return The output after applying the Softmax activation function.
     */
    public static Double softmax(NeuronLayer lastLayer, Double innerPotential) {
        double nominator = Math.exp(innerPotential);
        double denominator = 0;
        for(Neuron neuron : lastLayer.neurons) {
             denominator += Math.exp(neuron.input);
        }
        return nominator / denominator;
    }

    /**
     * Computes the base/@ logarithm of the input.
     * @param input is the input of the neuron.
     * @return the base/@ logarithm of the input.
     */
    public static double logBase2(double input) {
        return Math.log(input) / Math.log(2);
    }
}