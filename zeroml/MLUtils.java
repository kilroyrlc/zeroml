package zeroml;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;

import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.factory.Nd4j;

import kaflib.graphics.GraphicsUtils;
import kaflib.utils.CheckUtils;
import kaflib.utils.RandomUtils;

public class MLUtils {

	/**
	 * Loads an image matching the loader spec.
	 * @param file
	 * @param loader
	 * @return
	 * @throws Exception
	 */
    public static INDArray loadImage(final File file,
    								 final NativeImageLoader loader) throws Exception {
        INDArray content = loader.asMatrix(file);
        return content;
    }
	
    /**
     * Loads an image matching the loader spec.  Normalizes it.
     * @param file
     * @param loader
     * @param normalization
     * @return
     * @throws Exception
     */
    public static INDArray loadAndNormalizeImage(final File file,
			 									 final NativeImageLoader loader,
			 									 final DataNormalization normalization) throws Exception {
    	INDArray content = loadImage(file, loader);
    	normalization.transform(content);
    	return content;
    }
    		
    public static void saveImage(final INDArray array,
    					   final DataNormalization normalization,
    					   final File file) throws Exception {
    	CheckUtils.check(array, "array");
    	
    	INDArray duplicate = array.dup();
    	if (normalization != null) {
    		normalization.revertFeatures(duplicate);
    	}
        GraphicsUtils.writeJPG(getImage(duplicate), file);
    }
    
       
    public static BufferedImage getImage(final INDArray array) {
        int[] shape = array.shape();

        int height = shape[2];
        int width = shape[3];
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int red = array.getInt(0, 2, y, x);
                int green = array.getInt(0, 1, y, x);
                int blue = array.getInt(0, 0, y, x);

                //handle out of bounds pixel values
                red = Math.min(red, 255);
                green = Math.min(green, 255);
                blue = Math.min(blue, 255);

                red = Math.max(red, 0);
                green = Math.max(green, 0);
                blue = Math.max(blue, 0);
                image.setRGB(x, y, new Color(red, green, blue).getRGB());
            }
        }
        return image;
    }
    
    public static INDArray createNoiseArray(final int channels, 
    										final int width, 
    										final int height,
    										final double amount) throws Exception {
            int count = channels * height * width;
            double[] result = new double[count];
            for (int i = 0; i < result.length; i++) {
                result[i] = RandomUtils.randomDouble(-1 * amount, amount);
            }
            return Nd4j.create(result, new int[]{1, channels, height, width});
    }
    
    
    
}
