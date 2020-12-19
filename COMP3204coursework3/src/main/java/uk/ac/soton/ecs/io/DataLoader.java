package uk.ac.soton.ecs.io;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.Image;
import org.openimaj.image.ImageUtilities;

public class DataLoader {

    /**
     * Images will be grouped by the directories they are in where
     * they are all in the same class
     * @return
     */
    public VFSGroupDataset<FImage> loadTrainingData(String dirPath){

        VFSGroupDataset<FImage> classes = null;

        try {
            classes = new VFSGroupDataset<FImage>(dirPath, ImageUtilities.FIMAGE_READER);
        } catch (FileSystemException e) {
            e.printStackTrace();
        }

        return classes;

    }

    /**
     * Testing data is unlabelled
     * @return
     */
    public VFSListDataset<FImage> loadTestingData(String dirPath){

        VFSListDataset<FImage> testing = null;

        try {
            testing = new VFSListDataset<FImage>(dirPath, ImageUtilities.FIMAGE_READER);
        } catch (FileSystemException e) {
            e.printStackTrace();
        }

        return testing;

    }

}