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
import java.util.*;

public class Run2 extends Model {

    private static final int PATCH_SIZE = 4;
    private static final int STEP = 2;
    private LiblinearAnnotator classifier;

    public Run2(VFSGroupDataset<FImage> trainingData, VFSListDataset<FImage> testingData){
        this.trainingData = trainingData;
        this.testingData = testingData;
        splitter = new GroupedRandomSplitter(trainingData, 80, 0,20);
    }

    @Override
    public void run(){

        List<LocalFeatureList<FloatKeypoint>> patchSamplesForAllImages = new ArrayList();

        System.out.println("Getting the patch samples from each image...");
        int counter = 0;
        for(FImage i : trainingData){
            patchSamplesForAllImages.add(getPatchSamples(i, 30));
            System.out.println("Finished getting the samples from image: " + counter);
            counter++;
        }

        FloatKMeans fkm = FloatKMeans.createKDTreeEnsemble(500);

        /*
        We want to feed multiple lists of features to the k means
        clustering algorithm so use this helper class to handle this
         */
        LocalFeatureListDataSource<FloatKeypoint, float[]> src = new LocalFeatureListDataSource(patchSamplesForAllImages);

        System.out.println("Starting clustering...");
        //Do the k means clustering and then get the hard assigner from it
        HardAssigner<float[], float[], IntFloatPair> assigner = fkm.cluster(src).defaultHardAssigner();
        System.out.println("Finished clustering.");

        //Use the standard C value of 1 and a very small epsilon value
        classifier = new LiblinearAnnotator<FImage, String>(
            new WordsExtractor(assigner), LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC,
            1, 0.00001
        );
        System.out.println("Started training the model...");
        classifier.train(splitter.getTrainingDataset());
        System.out.println("Finished training the model.");

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
     * Take a sample of size n from the patches of the image (in the form
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
     * Get ALL the feature vectors from the patches of the given image
     * @param image
     * @return
     */
    public MemoryLocalFeatureList<FloatKeypoint> getPatchVectors(FImage image){

        List<Rectangle> patchRectangles =
                new RectangleSampler(image, STEP, STEP, PATCH_SIZE, PATCH_SIZE).allRectangles();

        /*
        To make a data source from a list of local features, openimaj requires
        us to use this specific child class of a list
        */
        MemoryLocalFeatureList<FloatKeypoint> patchVectors = new MemoryLocalFeatureList();

        for(Rectangle r : patchRectangles){

            /*
            extractROI will get the actual patch of the image (a subimage of the image)
            represented by the bounds of the rectangle. For each patch, mean center and normalise it
             */
            FImage patch = image.extractROI(r).process(new MeanCenter()).normalise();

            /*
            Store the location of the keypoint (from the coordinates of the rectangle)
            aswell as its pixels. There is never any tilting so orientation is 0 and the scale is always 1
             */
            FloatKeypoint keyPoint = new FloatKeypoint(r.x, r.y, 0, 1, patch.getFloatPixelVector());
            patchVectors.add(keyPoint);

        }

        return patchVectors;

    }

    @Override
    /**
     * The code is the same for Run1 and Run2, so make it a super method
     * instead
     */
    public List<String> getResultsArr(){
        return super.getResultsArr(classifier);
    }

    /**
     * Create a bag of visual words for each image
     */
    class WordsExtractor implements FeatureExtractor<SparseIntFV, FImage> {

        BlockSpatialAggregator<float[], SparseIntFV> aggregator;

        public WordsExtractor(HardAssigner<float[], float[], IntFloatPair> assigner){
            aggregator = new BlockSpatialAggregator(new BagOfVisualWords(assigner), 2, 2);
        }

        @Override
        public SparseIntFV extractFeature(FImage image) {

            /*
            Return a massive aggregated vector using the bag of words of samples of
            all images and ALL the patch vectors for this image
             */
            return aggregator.aggregate(getPatchVectors(image), image.getBounds());

        }

    }

}