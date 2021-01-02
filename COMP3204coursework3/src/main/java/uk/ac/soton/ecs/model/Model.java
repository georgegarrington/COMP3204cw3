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
     * After the model has been trained on the training data and
     * the testing data has been classified into predictions,
     * report the accuracy of the model
     */
    protected void report(Classifier classifier){

        System.out.println("Starting evaluation...");
        Calendar start = Calendar.getInstance();
        ClassificationEvaluator<CMResult<String>, String, FImage> folds =
                new ClassificationEvaluator(classifier, splitter.getTestDataset(),
                        new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));
        System.out.println("\n-----------------------REPORT-----------------------\n");
        System.out.println(folds.analyse(folds.evaluate()).getSummaryReport());
        Calendar finish = Calendar.getInstance();
        System.out.println("Evaluation finished in: " + (finish.getTimeInMillis() - start.getTimeInMillis()) / 1000);

    }

    /**
     * Abstract version that makes the specific model call the super
     * getResultsArr method, giving it's classifier which implements an
     * Annotator interface as an argument
     * @return
     */
    public abstract List<String> getResultsArr();

    protected List<String> getResultsArr(AbstractAnnotator annotator){

        List<String> results = new ArrayList<String>();
        FileObject[] arr = testingData.getFileObjects();
        String[] names = new String[arr.length];

        for(int i = 0; i < arr.length; i++){
            names[i] = arr[i].getName().getBaseName();
        }

        for(int i = 0; i < names.length; i++){

            FImage image = testingData.get(i);
            Set<String> predictedClasses = annotator.classify(image).getPredictedClasses();
            //If there are multiple classes predicted then just choose the first one in the array form of the set
            results.add(names[i] + " " + predictedClasses.toArray()[0]);

        }

        return results;

    }

}