package cz.muni.fi.pv021.impl.Vectors;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static cz.muni.fi.pv021.impl.Main.r;

/**
 * Class representing vectors and corresponding labels read from a file and used for testing or training.
 */
public class Vectors {

    /**
     * Dimension of input vectors.
     */
    public static int VECTOR_DIMENSION = 784;

    /**
     * Reader for file with labels.
     */
    private final BufferedReader labelsFile;

    /**
     * Reader for file with vectors.
     */
    private final BufferedReader vectorsFile;

    public List<Integer> labels = new ArrayList<>();
    public List<Double[]> vectors = new ArrayList<>();

    private final int inputCount;

    private int vectorIndex = 0;

    private int labelIndex = 0;

    /**
     * Creates Vectors with BufferedReader for file with vectors and file with labels. Reads all the data stored in the
     * files and saves them to Lists. Then the vector data are standardized.
     *
     * @param vectorName name of text file with vectors
     * @param labelName name of text file with labels
     * @throws FileNotFoundException
     */
    public Vectors(String vectorName, String labelName) throws IOException {
        this.labelsFile = new BufferedReader(new FileReader(labelName));
        this.vectorsFile = new BufferedReader(new FileReader(vectorName));
        inputCount = readAllInputs();
        standardizeData();
    }

    /**
     * Reads vectors and labels form corresponding files and adds them to their List. Reads while there are both unread
     * vectors and labels. Vectors are parsed before being stored.
     *
     * @return number of both labels and vectors read
     * @throws IOException
     */
    public int readAllInputs() throws IOException {
        String vct;
        String lbl;
        int inputCount = 0;
        while((vct = vectorsFile.readLine()) != null && (lbl = labelsFile.readLine()) != null){
            labels.add(Integer.parseInt(lbl));
            parseVector(vct);
            inputCount++;
        }
        return inputCount;
    }

    /**
     * Resets the indexes to 0, so vectors and labels can be read from beginning. Used for preparing the data for
     * testing.
     */
    public void readyTestInputs() {
        labelIndex = 0;
        vectorIndex = 0;
    }

    /**
     * Resets the indexes to 0, so vectors and labels can be read from beginning and then shuffles the data. Used for
     * preparing the data for training.
     */
    public void readyInputs() {
        labelIndex = 0;
        vectorIndex = 0;
        vectorShuffle();
    }

    /**
     * Returns last unread vector, if there are any unread vectors left, else returns null.
     *
     * @return last unread vector
     */
    public Double[] readVector() {
        if (vectorIndex >= inputCount) {
            return null;
        }
        vectorIndex++;
        return vectors.get(vectorIndex - 1);
    }

    /**
     * Returns last unread label, if there are any unread labels left, else returns -1.
     *
     * @return last unread label
     */
    public int readLabel(){
        if (labelIndex >= inputCount) {
            return -1;
        }
        labelIndex++;
        return labels.get(labelIndex - 1);
    }

    /**
     * Standardizes the data by dividing every value in every vector by 255.
     */
    public void standardizeData() {
        for(Double[] vector : vectors) {
            for (int j = 0; j < vector.length; j++) {
                double standardizedVal = vector[j] / 255.0;
                vector[j] =  standardizedVal;
            }
        }
    }

    /**
     * Takes in CSV String vector, parses it into array of doubles and adds it to vectors variable.
     *
     * @param vector vector
     */
    private void parseVector(String vector) {
        int index = 0;
        Double[] vectorArray = new Double[VECTOR_DIMENSION];
        for (int i = 0; i < VECTOR_DIMENSION; i++) {
            vectorArray[i] = 0.0;
            while (index < vector.length() && vector.charAt(index) != ',') {
                vectorArray[i] = vectorArray[i] * 10 + Character.getNumericValue(vector.charAt(index));
                index++;
            }
            index++;
        }
        vectors.add(vectorArray);
    }

    /**
     * Shuffles the vectors and labels by randomly changing their order in their Lists.
     */
    private void vectorShuffle() {
        Double[] vectorVariable;
        int labelVariable;
        for (int i = 0; i < vectors.size(); i++) {
            int index = r.nextInt(vectors.size());
            vectorVariable = vectors.get(index);
            labelVariable = labels.get(index);
            vectors.set(index, vectors.get(i));
            labels.set(index, labels.get(i));
            vectors.set(i, vectorVariable);
            labels.set(i, labelVariable);
        }
    }
}
