package uk.ac.soton.ecs.model;

import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;

public class Run2 implements Model {

    VFSGroupDataset<FImage> trainingData;
    VFSListDataset<FImage> testingData;

    public Run2(VFSGroupDataset<FImage> trainingData, VFSListDataset<FImage> testingData){
        this.trainingData = trainingData;
        this.testingData = testingData;
        this.init();
    }

    //TODO don't think this is necessary
    private void init(){

    }

    public void run(){

    }

    public void report(){

    }

}