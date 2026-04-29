package cz.muni.fi.pv021.impl.network;

import java.util.List;

/**
 * This class represents a layer of neurons in a neural network.
 */
public class NeuronLayer {
    /**
     * The number of neurons in the layer.
     */
    public int neuronCount;

    /**
     * The list of neurons in the layer.
     */
    public List<Neuron> neurons;

    /**
     * Constructs a new NeuronLayer with the specified list of neurons.
     *
     * @param neurons the list of neurons in the layer.
     */
    public NeuronLayer(List<Neuron> neurons) {
        this.neurons = neurons;
        neuronCount = neurons.size();
    }
}
