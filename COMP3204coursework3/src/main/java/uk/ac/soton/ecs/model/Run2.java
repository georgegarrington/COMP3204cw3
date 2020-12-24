package uk.ac.soton.ecs.model;

import de.bwaldvogel.liblinear.SolverType;
import org.apache.commons.vfs2.FileObject;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.image.processing.algorithm.MeanCenter;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

import javax.swing.*;
import java.util.*;

public class Run2 implements Model {

    private static final int PATCH_SIZE = 8;
    private static final int STEP = 4;
    VFSGroupDataset<FImage> trainingData;
    VFSListDataset<FImage> testingData;
    GroupedRandomSplitter<String, FImage> splitter;
    LiblinearAnnotator classifier;

    public Run2(VFSGroupDataset<FImage> trainingData, VFSListDataset<FImage> testingData){
        this.trainingData = trainingData;
        this.testingData = testingData;
        splitter = new GroupedRandomSplitter(trainingData, 80, 0,20);
    }

    public void run(){

        //The sampled patch vectors for each image
        List<List<float[]>> imagePatchVectors = new ArrayList<List<float[]>>();

        for(FImage image : trainingData){
            List<float[]> patchVectors = getPatchSamples(image, 20);
        }

        FloatKMeans fkm = FloatKMeans.createKDTreeEnsemble(500);
        FloatCentroidsResult clusters = fkm.cluster(imagePatchVectors.toArray(new float[][]{}));
        HardAssigner<float[], float[], IntFloatPair> assigner = clusters.defaultHardAssigner();
        classifier = new LiblinearAnnotator<FImage, String>(
                new WordsExtractor(assigner), LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC,
                1, 0.00001
        );

        System.out.println("Starting training...");
        Calendar start = Calendar.getInstance();
        classifier.train(splitter.getTrainingDataset());
        Calendar finish = Calendar.getInstance();
        System.out.println("Training finished in: " + (finish.getTimeInMillis() - start.getTimeInMillis()) / 1000);

    }

    public void report(){

        System.out.println("Starting evaluation...");
        Calendar start = Calendar.getInstance();
        ClassificationEvaluator<CMResult<String>, String, FImage> folds =
                new ClassificationEvaluator(classifier, splitter.getTestDataset(),
                        new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));
        System.out.println("\n--------------------RUN 1 REPORT--------------------\n");
        System.out.println(folds.analyse(folds.evaluate()).getSummaryReport());
        Calendar finish = Calendar.getInstance();
        System.out.println("Evaluation finished in: " + (finish.getTimeInMillis() - start.getTimeInMillis()) / 1000);

    }

    public void preProcess(){

        /*
        for(FImage image : trainingData){

            List<LocalFeatureImpl<SpatialLocation, FloatFV>> samples = getPatchSamples(image, 30);

        }*/

    }

    /**
     * Take a sample of size n patches from the image (in the form
     * of the feature vectors of the patches)
     * @param image
     * @param n
     * @return
     */
    public List<float[]> getPatchSamples(FImage image, int n){

        List<float[]> allPatches = getPatchVectors(image);
        Collections.shuffle(allPatches);
        return allPatches.subList(0, n);

    }

    /**
     * Get n sample patches from the given image, where each element in the list returned
     * is a pair of the patch' feature vector and a spatial location
     * @param image
     * @return
     */
    public List<float[]> getPatchVectors(FImage image){

        List<Rectangle> patchRectangles = getPatchRectangles(image);
        //List<FloatKeypoint> keyPoints = new ArrayList<>();
        //List<LocalFeatureImpl<SpatialLocation, FloatFV>> featureLocs = new ArrayList();

        List<float[]> patchVectors = new ArrayList();

        for(Rectangle r : patchRectangles){

            //Get the image patch defined by the bounds of the rectangle
            //TODO might be too expensive to process every single patch, later we could
            //TODO do this only in the patches we randomly sampled

            /*
            For each patch, mean center and normalise it
             */
            FImage patch = image.extractROI(r).process(new MeanCenter()).normalise();

                /*
                Flatten the pixels of the patch into a vector, and put in a FloatFV
                which is basically just a wrapper for the float vector
                 */
            //FloatFV fv = new FloatFV(patch.getFloatPixelVector());

            //SpatialLocation loc = new SpatialLocation(r.x, r.y);

            //A list of pairs of spatial location and feature vector
            //LocalFeatureImpl<SpatialLocation, FloatFV> featureLoc = new LocalFeatureImpl(loc, fv);

            //FloatKeypoint keyPoint = new FloatKeypoint(r.x, r.y, 0, 1, patch.getFloatPixelVector());
            patchVectors.add(patch.getFloatPixelVector());

        }

        //In order to get a random sample of size n, shuffle the list and take the first n
        //Collections.shuffle(patchVectors);
        //return patchVectors.subList(0, n);

        return patchVectors;

    }

    /**
     * Get the rectangular bounds to make patches of the image with the given step and patch sizes
     * @param image
     * @return
     */
    public List<Rectangle> getPatchRectangles(FImage image){

        RectangleSampler sampler = new RectangleSampler(image, STEP, STEP, PATCH_SIZE, PATCH_SIZE);
        return sampler.allRectangles();

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

    public String toString(){
        return "run2";
    }

    class WordsExtractor implements FeatureExtractor<SparseIntFV, FImage> {

        BagOfVisualWords<float[]> bag;

        public WordsExtractor(HardAssigner<float[], float[], IntFloatPair> assigner){
            bag = new BagOfVisualWords(assigner);
        }

        @Override
        public SparseIntFV extractFeature(FImage object) {
            List<float[]> patchSamples = getPatchSamples(object, 20);
            return bag.aggregateVectorsRaw(patchSamples);
        }

    }

}