package cz.muni.fi.pv021.impl;

import cz.muni.fi.pv021.impl.network.Network;
import cz.muni.fi.pv021.impl.network.Neuron;
import cz.muni.fi.pv021.impl.network.NeuronLayer;

import java.util.ArrayList;
import java.util.List;

// import static cz.muni.fi.pv021.impl.NetworkTrainer.errorFunction;

/**
 * Class representing one cycle of gradient computation. This class is given one vector and label and computes errors
 * for every weight and bias in the Network. These are then added to corresponding variables in neurons.
 */
public class GradientComputer {

    public Network network;

    public int label = 0;

    public List<List<Double>> outputError = new ArrayList<>();

    /**
     * Creates GradientComputer for given Network.
     *
     * @param network Network
     */
    public GradientComputer(Network network){
        this.network = network;
    }

    /**
     * This method computes error gradient for given vector and label.
     *
     * The computation follows gradient computation template for MLPs from slide 96 of presentation. First the method
     * calls forwardPass which computes inner potentials of neurons for given vector,
     * then backwardPass computes neuron errors for every neuron and
     * updateErrors computes errors for weights and biases and adds them to the errors stored in Neuron objects in
     * current Network.
     *
     * @param vector vector
     * @param label label
     */
    public void compute(Double[] vector, int label){
        this.label = label;
        forwardPass(vector, label);
        backwardPass();
        updateErrors();
        outputError = new ArrayList<>();
    }

    /**
     * This method makes forward pass through neural network for given vector and computes neuron outputs.
     *
     * First the method sets vector values as inputs of the first (input) layer.
     *
     * Then for every neuron in every layer except the last, the output of this neuron created by calling corresponding
     * activation function with neuron input and the result is then multiplied with each weight and added to neuron
     * input of neuron with which this weight is connected.
     *
     * If current neuron is not input neuron, bias is added to neuron input before changing inputs of neurons in layer
     * above.
     *
     * Bias is then added to the last layer of neurons.
     *
     * @param vector vector
     */
    private void forwardPass(Double[] vector, int label) {
        int vectorIndex = 0;
        for (Neuron neuron : network.layers.get(0).neurons) {
            neuron.input = vector[vectorIndex];
            vectorIndex++;
        }
        for (int i = 0; i < network.layerCount - 1; i++){ //for every layer except the last
            for (Neuron neuron : network.layers.get(i).neurons) { // take all neurons from current layer
                if (i != 0) {
                    neuron.input += neuron.gaussBias; // input layer has no bias
                }
                for (int j = 0; j < network.layers.get(i+1).neuronCount; j++){
                    network.layers.get(i+1).neurons.get(j).input += neuron.weights[j] * ActivationFunction.relu(neuron.input);
                }
            }
        }
        // Commented out code is used for computing loss function which is used for testing learning effectivity.
        // Not used in program run.
        // int neuronCount = 0;
        for (Neuron neuron : network.layers.get(network.layerCount - 1).neurons) { // every neuron in output layer
            neuron.input += neuron.gaussBias;
            /* errorFunction.addOutputNeuronValue(
                    ActivationFunction.softmax(network.layers.get(network.layerCount - 1), neuron.input),
                    ((neuronCount == label) ? 1 : 0)); */
            //neuronCount++;
        }
        // errorFunction.nextVector();
    }

    /**
     * This method makes backwards pass through neural network for given vector and calculates neuron error for every
     * neuron using label corresponding to this vector. The neuron errors are not computed for the last layer, because
     * it uses SOFTMAX. Method lastLayerBackPassSoftmax is used to update weight errors going to the last layer and
     * to compute neuron errors for second to last layer.
     *
     * Then the method goes through every neuron layer except the last two and computes the error for its neurons.
     *
     * The errors are added to the newLayer List which is at the end added as new value to outputError.
     *
     * The error of a neuron is computed as a sum of each outgoing weight multiplied by neuron error of the neuron
     * corresponding to this weight (neuron in one layer above current neuron to which this weight connects) and by
     * derivation of activation function of corresponding neuron with its input as input value(derivatedActivation).
     *
     * Errors used in calculation of neuron error are always from one layer above, which means that upperLayer variable
     * is always the last value of outputError.
     */
    private void backwardPass() {
        lastLayerBackPassSoftmax(); // computes neuron errors for second to last layer and places them to outputError
        for (int i = network.layerCount - 3; i >= 0; i--){ // for every layer except the last two
            List<Double> newLayer = new ArrayList<>(); // new layer of neuron errors
            List<Double> upperLayer = outputError.get(outputError.size() - 1); // neuron errors of the layer above
            for (Neuron neuron : network.layers.get(i).neurons) {
                double error = 0;
                for (int j = 0; j < upperLayer.size(); j++){
                    double derivatedActivation = ActivationFunction
                            .reluDerivation(network.layers.get(i + 1).neurons.get(j).input);
                    error += upperLayer.get(j) * derivatedActivation * neuron.weights[j];
                }
                newLayer.add(error);
            }
            outputError.add(newLayer);
        }
    }

    /**
     * Updates bias errors computed for this vector for every neuron except those in the last layer (updated in
     * lastLayerInnerPotentialError).
     */
    private void updateBias() {
        int errorIndex = 0;
        for (int i = network.layerCount - 2; i >= 0; i--){
            for (int j = 0; j < network.layers.get(i).neuronCount; j++){
                Neuron neuron = network.layers.get(i).neurons.get(j);
                neuron.biasError += outputError.get(errorIndex).get(j) * ActivationFunction.reluDerivation(neuron.input);
            }
            errorIndex += 1;
        }
    }

    /**
     * Computes weight errors for all weights except those going to the last layer from neuron errors
     * computed in backwardPass and updates weight and bias errors in neuron objects from network.
     *
     * For every layer except the last two the method takes every neuron and computes correction for every weight of
     * this neuron by multiplying neuron error of the neuron in the layer above to which this weight goes, the output of
     * this neuron and derivation of activation function used in the layer above with respect to input of the neuron
     * to which this weight goes.
     *
     * These errors are then added to the weight errors in the neuron object.
     *
     * Then bias errors are updated by calling updateBias.
     */
    private void updateErrors(){
        int errorIndex = outputError.size() - 2; // begins from first hidden layer because input layer neuron error is not used in weight errors
        for (int i = 0; i < network.layerCount - 2; i++){ // all layers except the last two
            List<Double> upperLayer = outputError.get(errorIndex);
            for (Neuron neuron : network.layers.get(i).neurons) {
                for (int j = 0; j < neuron.weights.length; j++){
                    double weightCorrection = upperLayer.get(j) * ActivationFunction.relu(neuron.input) *
                            ActivationFunction.reluDerivation(network.layers.get(i + 1).neurons.get(j).input);
                    neuron.weightErrors[j] += weightCorrection;
                }
            }
            errorIndex--;
        }
        updateBias();
    }

    /**
     * Calculates derivated inner potential errors of the SOFTMAX layer (last layer), from which can be calculated
     * neuron errors of the second to last layer. Updates bias errors of the last layer.
     *
     * For every neuron in the last layer the inner potential error is calculated by subtracting expected output of
     * this neuron (1 if the label is supposed to be this neuron, 0 otherwise) from SOFTMAX activated inner potential of
     * this neuron. These values are then returned in derivatedInnerPotentialErrors.
     *
     * Bias errors are updated for these neurons.
     *
     * @param derivatedInnerPotentialErrors empty list to which derivated inner potential errors of the SOFTMAX layer
     *                                      are added
     */
    private void lastLayerInnerPotentialError(List<Double> derivatedInnerPotentialErrors) {
        NeuronLayer lastLayer = network.layers.get(network.layerCount - 1);
        int neuronCount = 0;
        for (Neuron neuron : lastLayer.neurons) {
            double innerPotentialError = ActivationFunction.softmax(lastLayer, neuron.input) - ((neuronCount == label) ? 1 : 0);
            derivatedInnerPotentialErrors.add(innerPotentialError);
            neuron.biasError += innerPotentialError;
            neuronCount++;
        }
    }

    /**
     * Computes inner potential errors for output layer using SOFTMAX and from them neuron error for second to last layer.
     * Weight errors going to the last layer are also computed.
     *
     * First the method calls lastLayerInnerPotentialError method, which computes inner potential errors for output layer.
     *
     * Then for every neuron in second to last layer we compute weight errors for all weights from activated inner potential
     * of the neuron multiplied by derivated inner potential error from corresponding neuron (neuron to which the weight goes).
     *
     * For every neuron in second to last layer we also compute neuron errors from each weight
     * multiplied by derivated inner potential error from corresponding neuron (neuron to which the weight goes).
     *
     * Then we add these errors as first layer of outputError list for use in computing errors in lesser layers.
     */
    private void lastLayerBackPassSoftmax() {
        List<Double> lastHiddenLayer = new ArrayList<>();
        List<Double> derivatedInnerPotentialErrors = new ArrayList<>();
        lastLayerInnerPotentialError(derivatedInnerPotentialErrors);
        for (Neuron neuron : network.layers.get(network.layerCount - 2).neurons) {
            double neuronError = 0;
            for (int i = 0; i < neuron.weights.length; i++) {
                neuronError += derivatedInnerPotentialErrors.get(i) * neuron.weights[i];
                neuron.weightErrors[i] += derivatedInnerPotentialErrors.get(i) * ActivationFunction.relu(neuron.input);
            }
            lastHiddenLayer.add(neuronError);
        }
        outputError.add(lastHiddenLayer);
    }

}
