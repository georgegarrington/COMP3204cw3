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
import org.openimaj.ml.annotation.AbstractAnnotator;
import org.openimaj.ml.annotation.basic.KNNAnnotator;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public class Run1 extends Model {

    private static final int K = 9;
    private static int RESOLUTION = 16;
    protected KNNAnnotator<FImage, String, FloatFV> classifier;

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
    @Override
    public void run(){
        classifier.trainMultiClass(splitter.getTrainingDataset());
    }

    /**
     * The code is the same for Run1 and Run2, so make it a super method
     * instead
     */
    @Override
    public void report(){
        super.report(classifier);
    }

    /**
     * The code is the same for Run1 and Run2, so make it a super method
     * instead
     */
    @Override
    public List<String> getResultsArr(){
        return super.getResultsArr(classifier);
    }

    public void testResolutions(){

        int[] resolutions = {4,8,16,24,32};

        for(int i : resolutions){

            this.RESOLUTION = i;

            double sumAccuracy = 0.0;

            for(int j = 0; j < 10; j++){

                classifier = KNNAnnotator.create(new Flattener(), FloatFVComparison.EUCLIDEAN, K);
                classifier.trainMultiClass(splitter.getTrainingDataset());
                System.out.println("Now evaluating for resolution: " + RESOLUTION);
                sumAccuracy += super.report(classifier);

            }

            System.out.println("ACCURACY FOR RESOLUTION = " + RESOLUTION + " is " + (sumAccuracy / 10.0));

        }

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

}