package cz.muni.fi.pv021.impl.LossFunctions;

/**
 * Class representing categorical cross entropy loss function used when calculating how well the neural network
 * classification works. Not used currently, but was used when fine-tuning the hyperparameters.
 */
public class CategoricalCrossEntropy {
    public double[] vectorErrors;

    public int batchSize;

    public int round = 0;

    /**
     * Creates CategoricalCrossEntropy for given batch size.
     *
     * @param batchSize batch size
     */
    public CategoricalCrossEntropy(int batchSize) {
        this.batchSize = batchSize;
        vectorErrors = new double[batchSize];
    }

    /**
     * Adds error contribution of a neural network output neuron for a specific label.
     *
     * @param value The output value of the neuron
     * @param label The target label (desired output)
     */
    public void addOutputNeuronValue(double value, int label) {
        vectorErrors[round] += label * Math.log(value);
    }


    /**
     * Allows to proceed to the next vector
     */
    public void nextVector(){
        round++;
    }

    /**
     * Calculates and subsequently prints the value of accumulated cross-entropy error. Calls clean() to reset
     * round counter and vectorErrors list.
     */
    public void printValue(){
        double error = 0;
        for (double vectorError: vectorErrors) {
            error += vectorError;
        }
        // System.out.println("Cross entropy: " + error * ((-1.0) / round));
        clean();
    }

    /**
     * Resets round counter and vectorErrors list
     */
    private void clean() {
        round = 0;
        for (int i = 0; i < batchSize; i++) {
            vectorErrors[i] = 0;
        }
    }
}
