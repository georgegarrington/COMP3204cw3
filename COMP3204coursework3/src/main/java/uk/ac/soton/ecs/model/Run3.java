package uk.ac.soton.ecs.model;

import de.bwaldvogel.liblinear.SolverType;
import opennlp.tools.dictionary.serializer.Entry;
import org.apache.commons.vfs2.FileObject;
import org.apache.hadoop.util.bloom.Key;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.list.MemoryLocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.feature.local.aggregate.PyramidSpatialAggregator;
import org.openimaj.image.feature.local.keypoints.FloatKeypoint;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.image.processing.algorithm.MeanCenter;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.util.pair.IntFloatPair;
import java.util.*;

public class Run3 {

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

        DenseSIFT dsift = new DenseSIFT(5, 7);
        PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 7);

        // K-means clustering
        HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(splitter.getTrainingDataset(), pdsift);


        FeatureExtractor<DoubleFV, FImage> extractor = new Group_Project.Group_Project.PHOWExtractor(pdsift, assigner);

        HomogeneousKernelMap hkm2 = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2, 2.1, HomogeneousKernelMap.WindowType.Rectangular);

        extractor = hkm2.createWrappedExtractor(extractor);


        System.out.println("Here2");
        // PHOW Feature Extractor
        //FeatureExtractor<DoubleFV, FImage> extractor = new PHOWExtractor(pdsift, assigner);

        System.out.println("Here3");
        // Construct and train classifier
        LiblinearAnnotator<FImage, String> classifier = new LiblinearAnnotator<FImage, String>(extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);

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


    static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(Dataset<FImage> sample, PyramidDenseSIFT<FImage> pdsift){
        List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();

        for (FImage rec : sample) {
            FImage img = rec.getImage();

            pdsift.analyseImage(img);
            allkeys.add(pdsift.getByteKeypoints(0.005f));
        }

        if (allkeys.size() > 10000)
            allkeys = allkeys.subList(0, 10000);

        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(300);
        DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
        ByteCentroidsResult result = km.cluster(datasource);


        return result.defaultHardAssigner();
    }

    // Feature Extractor
    static class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {
        PyramidDenseSIFT<FImage> pdsift;
        HardAssigner<byte[], float[], IntFloatPair> assigner;

        public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner)
        {
            this.pdsift = pdsift;
            this.assigner = assigner;
        }

        public DoubleFV extractFeature(FImage object) {
            FImage image = object.getImage();
            pdsift.analyseImage(image);

            BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);

            BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<byte[], SparseIntFV>(
                    bovw, 2, 2);

            return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
        }
    }



    // Feature Extractor with Pyramid Spatial Aggregator
    static class PyramidExtractor implements FeatureExtractor<DoubleFV, FImage> {
        PyramidDenseSIFT<FImage> pdsift;
        HardAssigner<byte[], float[], IntFloatPair> assigner;

        public PyramidExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner)
        {
            this.pdsift = pdsift;
            this.assigner = assigner;
        }

        public DoubleFV extractFeature(FImage object) {
            FImage image = object.getImage();
            pdsift.analyseImage(image);

            BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);

            PyramidSpatialAggregator<byte[], SparseIntFV> spatial = new PyramidSpatialAggregator<byte[], SparseIntFV>(
                    bovw, 2, 2);

            return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
        }
    }

}
