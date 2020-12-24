package uk.ac.soton.ecs.model;

import de.bwaldvogel.liblinear.SolverType;
import opennlp.tools.dictionary.serializer.Entry;
import org.apache.commons.vfs2.FileObject;
import org.apache.hadoop.util.bloom.Key;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.list.MemoryLocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.feature.local.keypoints.FloatKeypoint;
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

        List<LocalFeatureList<FloatKeypoint>> patchVectorsForAllImages = new ArrayList();

        System.out.println("Getting the patch samples from each image...");
        int counter = 0;
        for(FImage i : trainingData){
            patchVectorsForAllImages.add(getPatchSamples(i, 30));
            System.out.println("Finished getting the samples from sample: " + counter);
            counter++;
        }

        FloatKMeans fkm = FloatKMeans.createKDTreeEnsemble(600);
        DataSource<float[]> src = new LocalFeatureListDataSource<FloatKeypoint, float[]>(patchVectorsForAllImages);
        System.out.println("now going to start the clustering...");
        FloatCentroidsResult clusters = fkm.cluster(src);
        System.out.println("finished clustering");
        HardAssigner<float[], float[], IntFloatPair> assigner = clusters.defaultHardAssigner();
        classifier = new LiblinearAnnotator<FImage, String>(
                new WordsExtractor(assigner), LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC,
                1, 0.00001
        );
        System.out.println("Started training the model...");
        classifier.train(splitter.getTrainingDataset());
        System.out.println("Finished training the model.");

    }

    public void runOld(){

        /*
        //Get the total number of images in the training data set
        int totalSize = 0;
        for(String str : trainingData.getGroups()){
            totalSize += trainingData.get(trainingData).size();
        }*/

        //float[][] imagePatchVectors = new float[totalSize][];

        //The sampled patch vectors for each image

        /*

        VERSION 1

        List<float[]> imagePatchVectors = new ArrayList();

        //int i = 0;

        System.out.println("Starting patch sampling...");
        int i = 0;
        for(FImage image : trainingData){
            List<float[]> patchVectors = getPatchSamples(image, 50);
            System.out.println("Just finished sampling image: " + i);
            i++;
            imagePatchVectors.addAll(patchVectors);
        }

        FloatKMeans fkm = FloatKMeans.createKDTreeEnsemble(600);
        System.out.println("Now going to try the array bit that wasn't working");
        FloatCentroidsResult clusters = fkm.cluster(imagePatchVectors.toArray(new float[][]{}));
        System.out.println("It worked :)");
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
        */

    }

    public void report(){

        System.out.println("Starting evaluation...");
        Calendar start = Calendar.getInstance();
        ClassificationEvaluator<CMResult<String>, String, FImage> folds =
                new ClassificationEvaluator(classifier, splitter.getTestDataset(),
                        new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));
        System.out.println("\n--------------------RUN 2 REPORT--------------------\n");
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
    public MemoryLocalFeatureList<FloatKeypoint> getPatchSamples(FImage image, int n){
        MemoryLocalFeatureList<FloatKeypoint> allPatches = getPatchVectors(image);
        Collections.shuffle(allPatches);
        return allPatches.subList(0, n);
    }

    /**
     * Get n sample patches from the given image, where each element in the list returned
     * is a pair of the patch' feature vector and a spatial location
     * @param image
     * @return
     */
    public MemoryLocalFeatureList<FloatKeypoint> getPatchVectors(FImage image){

        List<Rectangle> patchRectangles = getPatchRectangles(image);
        //List<FloatKeypoint> keyPoints = new ArrayList<>();
        //List<LocalFeatureImpl<SpatialLocation, FloatFV>> featureLocs = new ArrayList();

        MemoryLocalFeatureList<FloatKeypoint> patchVectors = new MemoryLocalFeatureList();

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

            FloatKeypoint keyPoint = new FloatKeypoint(r.x, r.y, 0, 1, patch.getFloatPixelVector());
            patchVectors.add(keyPoint);

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


            //System.out.println("The number of predicted classes is: " + predictedClasses.size());
            String str = name + " " + predictedClasses.toArray()[0];


            results.add(str);
            //System.out.println("Just classified an image");
           // System.out.println(str);

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

        /*
        public SparseIntFV extractFeatureOld(FImage object) {
            MemoryLocalFeatureList<FloatKeypoint> patchSamples = getPatchSamples(object, 20);
            //return bag.aggregateVectorsRaw(patchSamples);
            return null;
        }*/

        @Override
        public SparseIntFV extractFeature(FImage image) {

            BlockSpatialAggregator<float[], SparseIntFV> aggegator =
                    new BlockSpatialAggregator<float[], SparseIntFV>(bag, 2, 2);
            return aggegator.aggregate(getPatchVectors(image), image.getBounds());

        }
    }

}