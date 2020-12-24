package uk.ac.soton.ecs.model;

import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.feature.FloatFV;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.basic.KNNAnnotator;

import java.util.List;

public abstract class Model {

    protected VFSGroupDataset<FImage> trainingData;
    protected VFSListDataset<FImage> testingData;
    protected GroupedRandomSplitter<String, FImage> splitter;

    /**
     * Run the model doing which will train it using the labelled training data
     * given in the constructor, and then make predictions of what class each
     * instance in the testing data is in
     */
    public abstract void run();

    /**
     * After the model has been trained on the training data and
     * the testing data has been classified into predictions,
     * report the accuracy of the model
     */
    public abstract void report();

    public abstract List<String> getResultsArr();

}