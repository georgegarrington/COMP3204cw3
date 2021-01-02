package uk.ac.soton.ecs.io;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.Image;
import org.openimaj.image.ImageUtilities;

/**
 * Used for loading testing and training image data from their directories
 */
public class DataLoader {

    /**
     * Groups images into their respective classes determined by the directories they
     * are in, use for supervised learning methods
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
     * Testing data is unlabelled, so load it as just one list of images
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