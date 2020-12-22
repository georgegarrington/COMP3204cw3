package uk.ac.soton.ecs.model;

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
import java.util.Arrays;
import java.util.List;

public class Run1 implements Model {

    private static int K = 1;
    private static final int RESOLUTION = 16;

    //Probably too many fields
    private VFSGroupDataset<FImage> trainingData;
    private VFSListDataset<FImage> testingData;
    private KNNAnnotator<FImage, String, FloatFV> classifier;
    private List<ClassificationResult<String>> results;
    private GroupedRandomSplitter<String, FImage> splitter;

    public Run1(VFSGroupDataset<FImage> trainingData, VFSListDataset<FImage> testingData){
        this.trainingData = trainingData;
        this.testingData = testingData;
        classifier = KNNAnnotator.create(new Flattener(), FloatFVComparison.EUCLIDEAN, K);
        //Use k-fold cross validation on the training data to estimate the accuracy of the model
        splitter = new GroupedRandomSplitter(trainingData, 90, 0,10);
        results = new ArrayList<ClassificationResult<String>>();
        //this.init();
    }

    //TODO Don't think we need this
    private void init(){


    }

    /**
     * Run the model which will train it using the labelled training data
     * given in the constructor, and then make predictions of what class each
     * instance in the testing data is in and store this in the "results" field
     */
    public void run(){
        //classifier.train(splitter.getTrainingDataset());
        classifier.trainMultiClass(splitter.getTrainingDataset());
        /*
        for(FImage image : testingData){
            results.add(classifier.classify(image));
        }*/
    }

    /**
     * After the model has been trained on the training data and
     * the testing data has been classified into predictions,
     * report the accuracy of the model by doing k-fold cross validation
     * on the training data
     */
    public void report(){

//        //Use k-fold cross validation on the training data to estimate the accuracy of the model
//        GroupedRandomSplitter<String, FImage> splitter = new GroupedRandomSplitter(trainingData, 450, 0,50);
               // new GroupedRandomSplitter(trainingData, 10, 0,10);


        ClassificationEvaluator<CMResult<String>, String, FImage> folds =
            new ClassificationEvaluator(classifier, splitter.getTestDataset(),
                new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));

        System.out.println("\n--------------------RUN 1 REPORT--------------------\n");
        //System.out.println(folds.analyse(folds.evaluate()).getDetailReport());
        System.out.println(folds.analyse(folds.evaluate()).getSummaryReport());
        //System.out.println(folds.analyse(folds.evaluate()).getDetailReport("Accuracy"));

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
            //Normalise it first so that it has 0 mean and unit length
            shrunk = shrunk.normalise();

            /*
            //Flatten the pixel matrix into a vector

            float[] vector = shrunk.getFloatPixelVector();
            float mean = 0;
            float sd = 0;
            float sum = 0;

            for(float f : vector){
                sum += f;
            }
            mean = sum / vector.length;

            sum = 0;
            for(int i = 0; i < vector.length; i++){
                sum += Math.pow(vector[i] - mean, 2);
            }
            sd = (float) Math.sqrt(sum / vector.length - 1);

            // Subtract by the main and divide by standard deviation to
            // zero the mean and make unit length, I think?
            for(int i = 0; i < vector.length; i++){
                vector[i] -= mean;
                vector[i] /= sd;
            }*/

            return new FloatFV(shrunk.getFloatPixelVector());
        }

    }

}