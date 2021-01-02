package uk.ac.soton.ecs;

//None of this is used
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;
import uk.ac.soton.ecs.io.DataLoader;
import uk.ac.soton.ecs.model.Run1;

import java.io.File;

public class Main {

    //Specific paths different for each member testing and training directories on their device
    private static final String GOPIKA_TESTING = genPathStr(new String[]{System.getProperty("user.home"), "Downloads", "testing", "testing"}); //"C:\\Users\\gopik\\Downloads\\testing\\testing";
    private static final String GOPIKA_TRAINING = genPathStr(new String[]{System.getProperty("user.home"), "Downloads", "training", "training"}); //"C:\\Users\\gopik\\Downloads\\training\\training";
    private static final String GEORGE_TESTING = genPathStr(new String[]{System.getProperty("user.home"), "Desktop", "testing"});
    private static final String GEORGE_TRAINING = genPathStr(new String[]{System.getProperty("user.home"), "Desktop", "training"});

    public static void main(String[] args) {

        DataLoader loader = new DataLoader();
        VFSGroupDataset<FImage> trainingData =
                loader.loadTrainingData(GEORGE_TRAINING);
        VFSListDataset<FImage> testingData =
                loader.loadTestingData(GEORGE_TESTING);

        new Run1(trainingData, testingData).testResolutions();

        /*
        Model[] models = new Model[]{
            new Run1(trainingData, testingData),
            //new Run2(trainingData, testingData)
        };

        //Use for saving the results to a file
        ResultSerializer rs = new ResultSerializer();

        for(int i = 0; i < models.length; i++){

            models[i].run();
            ((Run1) models[i]).custom();
            models[i].report();
            
            try {
                rs.serializeResults("run" + (i + 1), models[i].getResultsArr());
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }

        }*/

    }

    /**
     * Generate a string path from an array of strings,
     * handles OS specific file seperator
     * @param arr
     * @return
     */
    public static String genPathStr(String[] arr){
        String out = arr[0];
        for(int i = 1; i < arr.length; i++){
            out += File.separator + arr[i];
        }
        return out;
    }

}