package uk.ac.soton.ecs.model;

import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.image.processing.algorithm.MeanCenter;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Run2 implements Model {

    private static final int PATCH_SIZE = 8;
    private static final int STEP = 4;
    VFSGroupDataset<FImage> trainingData;
    VFSListDataset<FImage> testingData;

    public Run2(VFSGroupDataset<FImage> trainingData, VFSListDataset<FImage> testingData){
        this.trainingData = trainingData;
        this.testingData = testingData;
    }

    public void run(){

        /*
        //The sampled patch vectors for each image
        List<List<float[]>> imagePatchVectors = new ArrayList<List<float[]>>();

        for(FImage image : trainingData){
            List<float[]> patchVectors = getPatchSamples(image, 20);
        }

        FloatKMeans fkm = FloatKMeans.createKDTreeEnsemble(500);
        FloatCentroidsResult clusters = fkm.cluster(imagePatchVectors.toArray(new float[][]{}));
        HardAssigner<float[], float[], IntFloatPair> assigner = clusters.defaultHardAssigner();
        */


    }

    public void preProcess(){

        /*
        for(FImage image : trainingData){

            List<LocalFeatureImpl<SpatialLocation, FloatFV>> samples = getPatchSamples(image, 30);

        }*/

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

        return patchVectors

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


    public void report(){

    }

    public List<String> getResultsArr(){

        return null;

    }

    public String toString(){
        return "run2";
    }

    class WordsExtractor implements FeatureExtractor<FloatFV, FImage> {

        BagOfVisualWords<float[]> bag;

        WordsExtractor(Run2 run2, HardAssigner<float[], float[], IntFloatPair> assigner){

            bag = new BagOfVisualWords(assigner);

        }

        @Override
        public FloatFV extractFeature(FImage object) {

            List<float[]> patchSamples = getPatchVectors(object);



            return null;

        }

    }

}