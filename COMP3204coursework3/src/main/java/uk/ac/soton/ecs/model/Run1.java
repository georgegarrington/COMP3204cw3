package uk.ac.soton.ecs.model;

import org.apache.commons.vfs2.FileObject;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.*;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.ml.annotation.basic.KNNAnnotator;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public class Run1 implements Model {

    private static int K = 2;
    private static final int RESOLUTION = 16;
    private VFSListDataset<FImage> testingData;
    private KNNAnnotator<FImage, String, FloatFV> classifier;
    private GroupedRandomSplitter<String, FImage> splitter;

    public Run1(VFSGroupDataset<FImage> trainingData, VFSListDataset<FImage> testingData){
        this.testingData = testingData;
        classifier = KNNAnnotator.create(new Flattener(), FloatFVComparison.EUCLIDEAN, K);
        //Use cross validation on the training data to estimate the accuracy of the model
        splitter = new GroupedRandomSplitter(trainingData, 80, 0,20);
    }

    /**
     * Run the model which will train it using the labelled training data
     * given in the constructor, and then make predictions of what class each
     * instance in the testing data is in and store this in the "results" field
     */
    public void run(){
        classifier.trainMultiClass(splitter.getTrainingDataset());
    }

    /**
     * After the model has been trained on the training data and
     * the testing data has been classified into predictions,
     * report the accuracy of the model by doing k-fold cross validation
     * on the training data
     */
    public void report(){
        ClassificationEvaluator<CMResult<String>, String, FImage> folds =
            new ClassificationEvaluator(classifier, splitter.getTestDataset(),
                new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));
        System.out.println("\n--------------------RUN 1 REPORT--------------------\n");
        System.out.println(folds.analyse(folds.evaluate()).getSummaryReport());
    }

    public List<String> getResultsArr(){

        List<String> results = new ArrayList<String>();
        FileObject[] arr = testingData.getFileObjects();
        String[] names = new String[arr.length];

        for(int i = 0; i < arr.length; i++){
            names[i] = arr[i].getName().getBaseName();
        }

        for(int i = 0; i < names.length; i++){

            FImage image = testingData.get(i);
            String name = names[i];
            Set<String> predictedClasses = classifier.classify(image).getPredictedClasses();

            for(String predictedClassName : predictedClasses){



            }

            System.out.println("The number of predicted classes is: " + predictedClasses.size());
            String str = name + " " + predictedClasses.toArray()[0];


            results.add(str);
            System.out.println("Just classified an image");
            System.out.println(str);

        }
        return results;

    }

    /**
     * KNN annotator will use this to flatten each image into the vector of
     * pixels it will use
     */
    class Flattener implements FeatureExtractor<FloatFV, FImage> {

        /**
         * Crops the image to a square about the center, resize to a fixed resolution,
         * tweak the image so that it has 0 mean and unit length and then flatten
         * the pixels into a single vector and return this
         */
        @Override
        public FloatFV extractFeature(FImage image) {
            int sqSize = Math.min(image.getWidth(), image.getHeight());

            // crops each image to a square about the centre
            FImage centered = image.extractCenter(sqSize, sqSize);

            //Shrink the image to the resolution, make sure to keep the same aspect ratio (probably unnecessary as its square)
            FImage shrunk = centered.process(
                    new ResizeProcessor(RESOLUTION, RESOLUTION, true));

            //Normalise it first so that it has 0 mean and unit length, Flatten the pixel matrix into a vector
            return new FloatFV(shrunk.normalise().getFloatPixelVector());
        }

    }

    public String toString(){
        return "run1";
    }

}