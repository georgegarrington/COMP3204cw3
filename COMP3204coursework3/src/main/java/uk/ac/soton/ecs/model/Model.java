package uk.ac.soton.ecs.model;

import java.util.List;

public interface Model {

    /**
     * Run the model doing which will train it using the labelled training data
     * given in the constructor, and then make predictions of what class each
     * instance in the testing data is in
     */
    public void run();

    /**
     * After the model has been trained on the training data and
     * the testing data has been classified into predictions,
     * report the accuracy of the model
     */
    //public void report();

    public List<String> getResultsArr();

}