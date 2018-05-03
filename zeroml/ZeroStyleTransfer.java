package zeroml;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.indexing.functions.Value;
import org.nd4j.linalg.learning.AdamUpdater;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.ops.transforms.Transforms;

import kaflib.graphics.GraphicsUtils;
import kaflib.gui.StatusField;
import kaflib.types.Directory;
import kaflib.types.Worker;
import kaflib.utils.CheckUtils;
import kaflib.utils.FileUtils;
import kaflib.utils.RandomUtils;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ZeroStyleTransfer extends Worker {

    private final String[] layers = new String[]{
        "input_1",
        "block1_conv1",
        "block1_conv2",
        "block1_pool",
        "block2_conv1",
        "block2_conv2",
        "block2_pool",
        "block3_conv1",
        "block3_conv2",
        "block3_conv3",
        "block3_pool",
        "block4_conv1",
        "block4_conv2",
        "block4_conv3",
        "block4_pool",
        "block5_conv1",
        "block5_conv2",
        "block5_conv3",
        "block5_pool",
        "flatten",
        "fc1",
        "fc2"
    };
    private final String[] style_layers = new String[]{
        "block1_conv1,0.5",
        "block2_conv1,1.0",
        "block3_conv1,1.5",
        "block4_conv2,3.0",
        "block5_conv1,4.0"
    };
    private final String content_layer = "block4_conv2";

    private final double beta_momentum = 0.8;
    private final double beta2_momentum = 0.999;
    private final double epsilon = 0.00000008;

    private final double alpha = 0.025;
    private final double beta = 5.0;

    private final double learning_rate = 2;
    private final double noise_amount = 20;
    private final int iterations = 20;

    private final File source_content;
    private final File source_style; 
    private final int save_every = 5;
    private final int recompute_noisy_every = 5;
    private final Directory output_directory = new Directory(".");

    private final int height;
    private final int width;
    private final int channels = 3;
    private final DataNormalization normalizer;
    private final NativeImageLoader loader;
    
    private String shape;
    private INDArray content;
    private INDArray style;
    private final INDArray content_original;
    private final INDArray style_original;
    
    private final String file_prefix;

    private final StatusField gui;

    public static void main(String[] args) {
    	try {        
    		Object handle = new Object();
    		int count = 5;
        	StatusField gui = new StatusField("Style transfer");
        	gui.getProgressBar().register(handle, count);

        	Directory source = new Directory("Z:\\data\\graphics\\neural\\source");
        	Directory style = new Directory("Z:\\data\\graphics\\neural\\style");
        	List<File> source_files = RandomUtils.randomize(source.list("jpg"));
        	List<File> style_files = RandomUtils.randomize(style.list("jpg"));
        	
        	for (int i = 0; i < count; i++) {
        		runFiles(source_files.get(i), style_files.get(i), gui);
        		gui.getProgressBar().increment(handle);        		
        	}
    		
            gui.setVisible(false);
            gui.close();

    	}
    	catch (Exception e) {
    		e.printStackTrace();
    	}
    }

    public static void runFiles(final File content, final File style, final StatusField gui) throws Exception {
		ZeroStyleTransfer transfer = new ZeroStyleTransfer(content, 
														   style,
														   gui);
		transfer.start();
		transfer.blockUntilDone(null);
		transfer = null;
		System.gc();
    }
    
    public ZeroStyleTransfer(final File content, final File style, final StatusField gui) throws Exception {
		CheckUtils.checkReadable(content, "content image");
		CheckUtils.checkReadable(style, "style image");

		this.gui = gui;
		normalizer = new VGG16ImagePreProcessor();
    	file_prefix = FileUtils.getFilenameWithoutExtension(content) + "_" +
    				  FileUtils.getFilenameWithoutExtension(style);
    	
		// Process content.
    	source_content = new File(output_directory, file_prefix + "_source.jpg");
    	BufferedImage image = GraphicsUtils.read(content);
    	image = GraphicsUtils.fill(image, 224, 224);
    	GraphicsUtils.writeJPG(image, source_content);
    	width = image.getWidth();
    	height = image.getHeight();
    	loader = new NativeImageLoader(height, width, channels);
    	this.content = MLUtils.loadAndNormalizeImage(source_content, loader, normalizer);
    	this.content_original = this.content.dup();
    	this.shape = this.content_original.shapeInfoToString();
    	
    	// Process style.
    	image = GraphicsUtils.read(style);
    	image = GraphicsUtils.fill(image, width, height);
    	source_style = new File(output_directory, file_prefix + "_style.jpg");
    	GraphicsUtils.writeJPG(image, source_style);
    	this.style = MLUtils.loadAndNormalizeImage(source_style, loader, normalizer);
    	this.style_original = this.style.dup();
    	if (!this.shape.equals(this.style_original.shapeInfoToString())) {
    		throw new Exception("Unmatched shapes.");
    	}
    	
    }
    
	@Override
	protected void process() throws Exception {
        gui.setText("Loading vgg model.");
        
        ComputationGraph vgg16FineTune = loadModel();
        gui.setText("Loading images.");
        
        gui.setText("Creating maps.");
        
        Map<String, INDArray> activationsContentMap = vgg16FineTune.feedForward(content_original, true);
        Map<String, INDArray> activationsStyleMap = vgg16FineTune.feedForward(style_original, true);
        HashMap<String, INDArray> activationsStyleGramMap = buildStyleGramValues(activationsStyleMap);

        AdamUpdater adamUpdater = createADAMUpdater();
        gui.setText("Created arrays/maps.");

        Object iteration_handle = new Object();
        gui.getProgressBar().register(iteration_handle, iterations);
        for (int iteration = 0; iteration < iterations; iteration++) {
        	if (iteration == 0) {
//        	if (iteration % recompute_noisy_every == 0) {
        		content = createNoisySourceImage();
            	MLUtils.saveImage(content, normalizer, 
            			new File(output_directory, String.format("noise_%03d.jpg", iteration)));
        		
        	}
        	
            gui.setText("Iteration: " + iteration + ".");
            gui.getProgressBar().increment(iteration_handle);

            INDArray[] input = new INDArray[] { content };
            Map<String, INDArray> activationsCombMap = vgg16FineTune.feedForward(input, true, false);

            gui.setText("Iteration: " + iteration + " back propagating style.");
            INDArray styleBackProb = backPropagateStyles(vgg16FineTune, activationsStyleGramMap, activationsCombMap);
            gui.setText("Iteration: " + iteration + " back propagating content.");
            INDArray backPropContent = backPropagateContent(vgg16FineTune, activationsContentMap, activationsCombMap);
            gui.setText("Iteration: " + iteration + " back propagating all.");
            INDArray backPropAllValues = backPropContent.muli(alpha).addi(styleBackProb.muli(beta));
            gui.setText("Iteration: " + iteration + " applying updater.");
            adamUpdater.applyUpdater(backPropAllValues, iteration, 0);
            content.subi(backPropAllValues);

            gui.setText("Iteration: " + iteration + " loss: " + 
            			totalLoss(activationsStyleMap, activationsCombMap, activationsContentMap));
            
            
            if (iteration != 0 && iteration % save_every == 0) {
            	MLUtils.saveImage(content, normalizer, 
            			new File(output_directory, String.format(file_prefix + "_%03d.jpg", iteration)));
            	
//                INDArray[] i = new INDArray[] { content_original };
//                Map<String, INDArray> m = vgg16FineTune.feedForward(i, true, false);
//            	for (String s : m.keySet()) {
//            		if (m.get(s).shapeInfoToString().equals(shape)) {
//                    	MLUtils.saveImage(m.get(s), normalizer, 
//                    			new File(output_directory, String.format(s + "_%03d.jpg", iteration)));
//            		}
//            	}
            }
        }
        gui.getProgressBar().release(iteration_handle);

    }

    private INDArray backPropagateStyles(ComputationGraph vgg16FineTune, HashMap<String, INDArray> activationsStyleGramMap, Map<String, INDArray> activationsCombMap) {
        INDArray styleBackProb = Nd4j.zeros(1, channels, height, width);
        for (String styleLayer : style_layers) {
            String[] split = styleLayer.split(",");
            String styleLayerName = split[0];
            INDArray styleGramValues = activationsStyleGramMap.get(styleLayerName);
            INDArray combValues = activationsCombMap.get(styleLayerName);
            double weight = Double.parseDouble(split[1]);
            int index = findLayerIndex(styleLayerName);
            INDArray dStyleValues = derivativeLossStyleInLayer(styleGramValues, combValues).transpose();
            styleBackProb.addi(backPropagate(vgg16FineTune, dStyleValues.reshape(combValues.shape()), index).muli(weight));
        }
        return styleBackProb;
    }

    private INDArray backPropagateContent(ComputationGraph vgg16FineTune, Map<String, INDArray> activationsContentMap, Map<String, INDArray> activationsCombMap) {
        INDArray activationsContent = activationsContentMap.get(content_layer);
        INDArray activationsComb = activationsCombMap.get(content_layer);
        INDArray dContentLayer = derivativeLossContentInLayer(activationsContent, activationsComb);
        return backPropagate(vgg16FineTune, dContentLayer.reshape(activationsComb.shape()), findLayerIndex(content_layer));
    }

    private AdamUpdater createADAMUpdater() {
        AdamUpdater adamUpdater = new AdamUpdater(new Adam(learning_rate, beta_momentum, beta2_momentum, epsilon));
        adamUpdater.setStateViewArray(Nd4j.zeros(1, 2 * channels * width * height),
            new int[]{1, channels, height, width}, 'c',
            true);
        return adamUpdater;
    }

    
    /**
     * Adds noise to the source image.
     * @return
     * @throws Exception
     */
    private INDArray createNoisySourceImage() throws Exception {
        INDArray noise = MLUtils.createNoiseArray(channels, width, height, noise_amount);
        noise.addi(this.content_original);
        return noise;
    }

    /*
     * Since style activation are not changing we are saving some computation by calculating style grams only once
     */
    private HashMap<String, INDArray> buildStyleGramValues(Map<String, INDArray> activationsStyle) {
        HashMap<String, INDArray> styleGramValuesMap = new HashMap<>();
        for (String styleLayer : style_layers) {
            String[] split = styleLayer.split(",");
            String styleLayerName = split[0];
            INDArray styleValues = activationsStyle.get(styleLayerName);
            styleGramValuesMap.put(styleLayerName, gramMatrix(styleValues));
        }
        return styleGramValuesMap;
    }

    private int findLayerIndex(String styleLayerName) {
        int index = 0;
        for (int i = 0; i < layers.length; i++) {
            if (styleLayerName.equalsIgnoreCase(layers[i])) {
                index = i;
                break;
            }
        }
        return index;
    }

    private double totalLoss(Map<String, INDArray> activationsStyleMap, Map<String, INDArray> activationsCombMap, Map<String, INDArray> activationsContentMap) {
        Double stylesLoss = allStyleLayersLoss(activationsStyleMap, activationsCombMap);
        return alpha * contentLoss(activationsCombMap.get(content_layer).dup(), activationsContentMap.get(content_layer).dup()) + beta * stylesLoss;
    }

    private Double allStyleLayersLoss(Map<String, INDArray> activationsStyleMap, Map<String, INDArray> activationsCombMap) {
        Double styles = 0.0;
        for (String styleLayers : style_layers) {
            String[] split = styleLayers.split(",");
            String styleLayerName = split[0];
            double weight = Double.parseDouble(split[1]);
            styles += styleLoss(activationsStyleMap.get(styleLayerName).dup(), activationsCombMap.get(styleLayerName).dup()) * weight;
        }
        return styles;
    }

    /**
     * After passing in the content, style, and combination images,
     * compute the loss with respect to the content. Based off of:
     * https://harishnarayanan.org/writing/artistic-style-transfer/
     *
     * @param combActivations    Intermediate layer activations from the three inputs
     * @param contentActivations Intermediate layer activations from the three inputs
     * @return Weighted content loss component
     */

    private double contentLoss(INDArray combActivations, INDArray contentActivations) {
        return sumOfSquaredErrors(contentActivations, combActivations) / (4.0 * (channels) * (width) * (height));
    }

    /**
     * This method is simply called style_loss in
     * https://harishnarayanan.org/writing/artistic-style-transfer/
     * but it takes inputs for intermediate activations from a particular
     * layer, hence my re-name. These values contribute to the total
     * style loss.
     *
     * @param style       Activations from intermediate layer of CNN for style image input
     * @param combination Activations from intermediate layer of CNN for combination image input
     * @return Loss contribution from this comparison
     */
    private double styleLoss(INDArray style, INDArray combination) {
        INDArray s = gramMatrix(style);
        INDArray c = gramMatrix(combination);
        int[] shape = style.shape();
        int N = shape[0];
        int M = shape[1] * shape[2];
        return sumOfSquaredErrors(s, c) / (4.0 * (N * N) * (M * M));
    }

    private INDArray backPropagate(ComputationGraph vgg16FineTune, INDArray dLdANext, int startFrom) {

        for (int i = startFrom; i > 0; i--) {
            Layer layer = vgg16FineTune.getLayer(layers[i]);
            dLdANext = layer.backpropGradient(dLdANext).getSecond();
        }
        return dLdANext;
    }


    /**
     * Element-wise differences are squared, and then summed.
     * This is modelled after the content_loss method defined in
     * https://harishnarayanan.org/writing/artistic-style-transfer/
     *
     * @param a One tensor
     * @param b Another tensor
     * @return Sum of squared errors: scalar
     */
    private double sumOfSquaredErrors(INDArray a, INDArray b) {
        INDArray diff = a.sub(b); // difference
        INDArray squares = Transforms.pow(diff, 2); // element-wise squaring
        return squares.sumNumber().doubleValue();
    }

    /**
     * Equation (2) from the Gatys et all paper: https://arxiv.org/pdf/1508.06576.pdf
     * This is the derivative of the content loss w.r.t. the combo image features
     * within a specific layer of the CNN.
     *
     * @param contentActivations Features at particular layer from the original content image
     * @param combActivations    Features at same layer from current combo image
     * @return Derivatives of content loss w.r.t. combo features
     */
    private INDArray derivativeLossContentInLayer(INDArray contentActivations, INDArray combActivations) {

        combActivations = combActivations.dup();
        contentActivations = contentActivations.dup();

        double channels = combActivations.shape()[0];
        double w = combActivations.shape()[1];
        double h = combActivations.shape()[2];

        double contentWeight = 1.0 / (2 * (channels) * (w) * (h));
        // Compute the F^l - P^l portion of equation (2), where F^l = comboFeatures and P^l = originalFeatures
        INDArray diff = combActivations.sub(contentActivations);
        // This multiplication assures that the result is 0 when the value from F^l < 0, but is still F^l - P^l otherwise
        return flatten(diff.muli(contentWeight).muli(ensurePositive(combActivations)));
    }

    /**
     * Computing the Gram matrix as described here:
     * https://harishnarayanan.org/writing/artistic-style-transfer/
     * Permuting dimensions is not needed because DL4J stores
     * the channel at the front rather than the end of the tensor.
     * Basically, each tensor is flattened into a vector so that
     * the dot product can be calculated.
     *
     * @param x Tensor to get Gram matrix of
     * @return Resulting Gram matrix
     */
    private INDArray gramMatrix(INDArray x) {
        INDArray flattened = flatten(x);
        return flattened.mmul(flattened.transpose());
    }

    private INDArray flatten(INDArray x) {
        int[] shape = x.shape();
        return x.reshape(shape[0] * shape[1], shape[2] * shape[3]);
    }


    /**
     * Equation (6) from the Gatys et all paper: https://arxiv.org/pdf/1508.06576.pdf
     * This is the derivative of the style error for a single layer w.r.t. the
     * combo image features at that layer.
     *
     * @param styleGramFeatures Intermediate activations of one layer for style input
     * @param comboFeatures     Intermediate activations of one layer for combo image input
     * @return Derivative of style error matrix for the layer w.r.t. combo image
     */
    private INDArray derivativeLossStyleInLayer(INDArray styleGramFeatures, INDArray comboFeatures) {

        comboFeatures = comboFeatures.dup();
        double N = comboFeatures.shape()[0];
        double M = comboFeatures.shape()[1] * comboFeatures.shape()[2];

        double styleWeight = 1.0 / ((N * N) * (M * M));
        // Corresponds to G^l in equation (6)
        INDArray contentGram = gramMatrix(comboFeatures);
        // G^l - A^l
        INDArray diff = contentGram.sub(styleGramFeatures);
        // (F^l)^T * (G^l - A^l)
        INDArray trans = flatten(comboFeatures).transpose();
        INDArray product = trans.mmul(diff);
        // (1/(N^2 * M^2)) * ((F^l)^T * (G^l - A^l))
        INDArray posResult = product.muli(styleWeight);
        // This multiplication assures that the result is 0 when the value from F^l < 0, but is still (1/(N^2 * M^2)) * ((F^l)^T * (G^l - A^l)) otherwise
        return posResult.muli(ensurePositive(trans));
    }

    private INDArray ensurePositive(INDArray comboFeatures) {
        BooleanIndexing.applyWhere(comboFeatures, Conditions.lessThan(0.0f), new Value(0.0f));
        BooleanIndexing.applyWhere(comboFeatures, Conditions.greaterThan(0.0f), new Value(1.0f));
        return comboFeatures;
    }

    private ComputationGraph loadModel() throws IOException {
        ZooModel zooModel = new VGG16();
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
        vgg16.initGradientsView();
        System.out.println(vgg16.summary());
        return vgg16;
    }


}
