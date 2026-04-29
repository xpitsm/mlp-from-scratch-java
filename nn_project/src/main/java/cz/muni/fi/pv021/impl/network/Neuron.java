package cz.muni.fi.pv021.impl.network;

import static cz.muni.fi.pv021.impl.Main.r;

/**
 * Class representing a neuron in a neural network.
 */
public class Neuron {
    public int layer;

    public double gaussBias;

    public double biasError = 0;

    public double[] weights;

    public double[] weightErrors;

    public double input = 0;

    public int round = 0;

    /**
     * Creates a neuron of given layer with nextLayerNeuronCount outgoing weights.
     * Initializes these weights with normal distribution with mean = 0 and standard deviation equal to 2 divided by the
     * number of neurons in this layer. Weight errors are set to 0.
     *
     * Bias weight and error is initialized the same way.
     *
     * @param nextLayerNeuronCount number of neurons in the next layer
     * @param layer layer number
     * @param layerNeuronCount number of neurons in this layer
     */
    public Neuron(int nextLayerNeuronCount, int layer, int layerNeuronCount) {
        this.layer = layer;
        weights = new double[nextLayerNeuronCount];
        weightErrors = new double[nextLayerNeuronCount];
        double std = Math.sqrt(2.0 / layerNeuronCount);
        gaussBias = r.nextGaussian() * std;
        for (int i = 0; i < nextLayerNeuronCount; i++) {
            weights[i] = r.nextGaussian() * std;
            weightErrors[i] = 0;
        }
    }

    /**
     * Prepares neuron for another forward run by resetting inner potential of the neuron(input).
     */
    public void reset() {
        input = 0;
    }


    /**
     * Updates neuron weights and bias after an error gradient is calculated from a training batch.
     *
     * First, the neuron input is reset to zero for use in the next forward pass.
     *
     * Then each weight is updated using the accumulated weight error and the learning rate.
     * After the update, the accumulated weight error is reset to zero.
     *
     * The bias is updated in the same way and its error is also reset to zero.
     *
     * @param learningRate learning rate used to update weights and bias
     */
    public void update(double learningRate) {
        input = 0;
        for (int i = 0; i < weights.length; i++) {
            weights[i] += learningRate * weightErrors[i];
            weightErrors[i] = 0;
        }
        gaussBias += learningRate * biasError;
        biasError = 0;

        round++;
    }
}
