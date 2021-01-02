package uk.ac.soton.ecs.model;

import org.apache.commons.vfs2.FileObject;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.Classifier;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.FloatFV;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.AbstractAnnotator;
import org.openimaj.ml.annotation.basic.KNNAnnotator;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;
import java.util.Set;

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
     * Abstract version that makes the specific model call the super
     * report method, giving it's classifier as an argument
     */
    public abstract void report();

    /**
     * Abstract version that makes the specific model call the super
     * getResultsArr method, giving it's classifier which implements an
     * Annotator interface as an argument
     * @return
     */
    public abstract List<String> getResultsArr();

}