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

    private static final int PATCH_SIZE = 8;
    private static final int STEP = 4;
    private LiblinearAnnotator classifier;

    public Run2(VFSGroupDataset<FImage> trainingData, VFSListDataset<FImage> testingData){
        this.trainingData = trainingData;
        this.testingData = testingData;
        splitter = new GroupedRandomSplitter(trainingData, 80, 0,20);
    }

    @Override
    public void run(){

        List<LocalFeatureList<FloatKeypoint>> patchVectorsForAllImages = new ArrayList();

        System.out.println("Getting the patch samples from each image...");
        int counter = 0;
        for(FImage i : trainingData){
            patchVectorsForAllImages.add(getPatchSamples(i, 30));
            System.out.println("Finished getting the samples from image: " + counter);
            counter++;
        }

        FloatKMeans fkm = FloatKMeans.createKDTreeEnsemble(600);
        DataSource<float[]> src = new LocalFeatureListDataSource<FloatKeypoint, float[]>(patchVectorsForAllImages);
        System.out.println("Starting clustering...");
        FloatCentroidsResult clusters = fkm.cluster(src);
        System.out.println("Finished clustering...");
        HardAssigner<float[], float[], IntFloatPair> assigner = clusters.defaultHardAssigner();
        classifier = new LiblinearAnnotator<FImage, String>(
                new WordsExtractor(assigner), LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC,
                1, 0.00001
        );
        System.out.println("Started training the model...");
        classifier.train(splitter.getTrainingDataset());
        System.out.println("Finished training the model.");

    }

    @Override
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
        MemoryLocalFeatureList<FloatKeypoint> patchVectors = new MemoryLocalFeatureList();

        for(Rectangle r : patchRectangles){

            /*
            extractROI will get the actual patch of the image (a subimage of the image)
            represented by the bounds of the rectangle. For each patch, mean center and normalise it
             */
            FImage patch = image.extractROI(r).process(new MeanCenter()).normalise();

            FloatKeypoint keyPoint = new FloatKeypoint(r.x, r.y, 0, 1, patch.getFloatPixelVector());
            patchVectors.add(keyPoint);

        }

        return patchVectors;

    }

    @Override
    public List<String> getResultsArr(){

        List<String> results = new ArrayList<String>();
        FileObject[] arr = testingData.getFileObjects();
        String[] names = new String[arr.length];

        for(int i = 0; i < arr.length; i++){
            names[i] = arr[i].getName().getBaseName();
        }

        for(int i = 0; i < names.length; i++){

            FImage image = testingData.get(i);
            Set<String> predictedClasses = classifier.classify(image).getPredictedClasses();
            //If there are multiple classes predicted then just choose the first one in the array form of the set
            results.add(names[i] + " " + predictedClasses.toArray()[0]);

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
        public SparseIntFV extractFeature(FImage image) {

            BlockSpatialAggregator<float[], SparseIntFV> aggegator =
                    new BlockSpatialAggregator<float[], SparseIntFV>(bag, 2, 2);
            return aggegator.aggregate(getPatchVectors(image), image.getBounds());

        }

    }

}