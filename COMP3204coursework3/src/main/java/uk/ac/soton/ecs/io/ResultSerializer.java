package uk.ac.soton.ecs.io;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.List;

/**
 * Use this class to make our text files from the results of each run
 */
public class ResultSerializer {

    public void serializeResults(String filename, List<String> results) throws FileNotFoundException {

        PrintWriter pw = new PrintWriter(new File(filename + ".txt"));
        /*
        for(int i = 0; i < results.size(); i++){
            pw.println(results.get(i));
        }*/
        for(String s : results){
            pw.println(s);
        }
        pw.close();
    }

}