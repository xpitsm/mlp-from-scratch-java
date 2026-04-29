package cz.muni.fi.pv021.impl;

import cz.muni.fi.pv021.impl.LossFunctions.CategoricalCrossEntropy;
import cz.muni.fi.pv021.impl.Vectors.Vectors;
import cz.muni.fi.pv021.impl.network.Network;

import java.io.IOException;

import static cz.muni.fi.pv021.impl.Main.BATCH_SIZE;

/**
 * Class representing network training.
 */
public class NetworkTrainer {

    // public static CategoricalCrossEntropy errorFunction = new CategoricalCrossEntropy(BATCH_SIZE);

    private final Vectors vectors;

    private final Network network;

    private final GradientComputer computer;

    /**
     * Creates NetworkTrainer for given network with vector and label file names to read from.
     *
     * @param vectorTrainFile name of text file with vectors
     * @param labelTrainFile name of text file with labels
     * @param network cz.muni.fi.pv021.impl.network
     * @throws IOException
     */
    public NetworkTrainer(String vectorTrainFile, String labelTrainFile, Network network) throws IOException {
        this.network = network;
        vectors = new Vectors(vectorTrainFile, labelTrainFile);
        computer = new GradientComputer(network);
    }

    /**
     * Trains the network by reading vectors and labels, calculating error gradients and updating weights.
     *
     * Method readies inputs for training, then trains network with a batches by calling trainBatch and then updates the
     * network. If number of vectors is not divisible by batch size, the last batch will be smaller.
     *
     * Commented out code was used for choosing the hyperparameters.
     */
    public void train(){
        boolean readAll = false;
        vectors.readyInputs();
        while (!readAll) {
            readAll = trainBatch();
            network.updateNetwork();
            /* if (!readAll) {
                errorFunction.printValue();
            } */
        }
        network.updateNetwork(); // if number of vectors is divisible by batch size this should do nothing, else update with smaller batch
        // errorFunction.printValue();
    }

    /**
     * This method reads BATCH_SIZE vectors and labels and trains with them the network by calling computer with each vector from the batch.
     * Computer then calculates error gradients and adds them to the current weight errors in neurons.
     *
     * If there are not enough vectors, smaller batch is used.
     *
     * @return true if the new batch was full, false else
     */
    private boolean trainBatch(){
        int label;
        Double[] vector;
        for (int i = 0; i < BATCH_SIZE; i++) {
            if ((label = vectors.readLabel()) != -1 && (vector = vectors.readVector()) != null){
                computer.compute(vector, label);
                network.resetNetwork();
            } else {
                network.resetNetwork();
                return true;
            }
        }
        return false;
    }
}
