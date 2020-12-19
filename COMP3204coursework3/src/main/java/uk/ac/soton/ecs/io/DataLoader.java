package uk.ac.soton.ecs.io;

import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;

public class DataLoader {

    /**
     * Images will be grouped by the directories they are in where
     * they are all in the same class
     * @return
     */
    public VFSGroupDataset<FImage> loadTrainingData(){

        return null;

    }

    /**
     * Testing data is unlabelled
     * @return
     */
    public VFSListDataset<FImage> loadTestingData(){

        return null;

    }

}