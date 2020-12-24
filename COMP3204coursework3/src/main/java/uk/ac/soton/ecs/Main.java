package uk.ac.soton.ecs;

//None of this is used
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.typography.hershey.HersheyFont;
import uk.ac.soton.ecs.io.DataLoader;
import uk.ac.soton.ecs.io.ResultSerializer;
import uk.ac.soton.ecs.model.Model;
import uk.ac.soton.ecs.model.Run1;
import uk.ac.soton.ecs.model.Run2;

import java.io.File;
import java.io.FileNotFoundException;

public class Main {

    public static void main(String[] args) {

        DataLoader loader = new DataLoader();

        /*
        Can't get relative paths to work (see the commented section below,
        if you have any idea how to do this please change it) so for now
        just declare the absolute paths of testing and training as an
        array of strings like this, good to do like this as some of us have
        Mac and Windows so this works on both
         */
        String[] trainingDirArr = new String[]{
                System.getProperty("user.home"),
                "Desktop",
                "training"
        };
        String[] testingDirArr = new String[]{
                System.getProperty("user.home"),
                "Desktop",
                "testing"
        };

        VFSGroupDataset<FImage> trainingData =
                loader.loadSupervisedTrainingData("C:\\Users\\gopik\\Downloads\\training\\training");
        VFSListDataset<FImage> testingData =
                loader.loadTestingData("C:\\Users\\gopik\\Downloads\\testing\\testing");


        //I want to get the path to work relative to the source directory
        //but I can't figure out how, says we have to use URIs or something.
        //Does anyone know?
        //VFSGroupDataset<FImage> trainingData =
        //        loader.loadSupervisedTrainingData("..\\..\\..\\training");
        //VFSListDataset<FImage> testingData =
        //        loader.loadTestingData("..\\..\\..\\testing");

        //TODO for testing delete later
        //DisplayUtilities.display("testing data", testingData);

        Model[] models = new Model[]{
                //new Run1(trainingData, testingData),
                new Run2(trainingData, testingData)
        };

        ResultSerializer rs = new ResultSerializer();

        for(Model m : models){

            m.run();
            m.report();
            try {
                rs.serializeResults(m.toString(),m.getResultsArr());
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }

        }

    }

    /**
     * Generate a string path from an array of strings
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