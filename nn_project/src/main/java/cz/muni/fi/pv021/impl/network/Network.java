package cz.muni.fi.pv021.impl.network;

import java.util.List;

import static cz.muni.fi.pv021.impl.Main.learningRate;

/**
 * Class representing the neural network.
 */
public class Network {
    public List<NeuronLayer> layers;

    public int layerCount;

    /**
     * Creates Network with given layers.
     *
     * @param layers list of neuron layers
     */
    public Network(List<NeuronLayer> layers){
        this.layers = layers;
        layerCount = layers.size();
    }

    /**
     * Resets the Network after run.
     *
     * Sets all inputs of neurons to 0, by calling reset().
     */
    public void resetNetwork() {
        for (NeuronLayer layer : layers) {
            for (Neuron neuron: layer.neurons) {
                neuron.reset();
            }
        }
    }

    /**
     * Updates neuron weights of all neurons in the Network after calculating the error gradient from training batch.
     *
     * The method updates every neuron by calling update() method.
     */
    public void updateNetwork() {
        for (NeuronLayer layer : layers) {
            for (Neuron neuron: layer.neurons) {
                neuron.update(learningRate);
            }
        }
    }
}
