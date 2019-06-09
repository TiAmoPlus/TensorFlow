package com.example.not_mnist;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Trace;
import android.util.Log;
import android.util.Size;

import org.tensorflow.Operation;
import org.tensorflow.Shape;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;

/** A classifier specialized to label images using TensorFlow. */
public class TensorFlowImageClassifier implements Classifier {
    private static final String TAG = "TFIC";

    // Only return this many results with at least this confidence.

    private static final int MAX_RESULTS = 10;
    private static final float THRESHOLD = 0.1f;

    // Config values.
    private String inputName;
    private String outputName;
    private int inputSize;
    private int imageMean;
    private float imageStd;

    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();
    private int[] intValues;
    private float[] floatValues;
    private float[] outputs;
    private String[] outputNames;

    private boolean logStats = false;

    private TensorFlowInferenceInterface inferenceInterface;

    private TensorFlowImageClassifier() {}

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param labelFilename The filepath of label file for classes.
     * @param inputSize The input size. A square image of inputSize x inputSize is assumed.
     * @param imageMean The assumed mean of the image values.
     * @param imageStd The assumed std of the image values.
     * @param inputName The label of the image input node.
     * @param outputName The label of the output node.
     * @throws IOException
     */
    public static Classifier create(
            AssetManager assetManager,
            String modelFilename,
            String labelFilename,
            int inputSize,
            int imageMean,
            float imageStd,
            String inputName,
            String outputName) {
        TensorFlowImageClassifier c = new TensorFlowImageClassifier();
        c.inputName = inputName;
        c.outputName = outputName;

        // Read the label names into memory.
        // TODO(andrewharp): make this handle non-assets.
        String actualFilename = labelFilename.split("file:///android_asset/")[1];
        Log.i(TAG, "Reading labels from: " + actualFilename);
        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(assetManager.open(actualFilename)));
            String line;
            while ((line = br.readLine()) != null) {
                c.labels.add(line);
            }
            br.close();
        } catch (IOException e) {
            throw new RuntimeException("Problem reading label file!" , e);
        }

        c.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);


        // The shape of the output is [N, NUM_CLASSES], where N is the batch size.
        final Operation operation = c.inferenceInterface.graphOperation(outputName);



        final int numClasses = (int) operation.output(0).shape().size(1);

        Log.i(TAG, "Read " + c.labels.size() + " labels, output layer size is " + numClasses);

        // Ideally, inputSize could have been retrieved from the shape of the input operation.  Alas,
        // the placeholder node for input in the graphdef typically used does not specify a shape, so it
        // must be passed in as a parameter.
        c.inputSize = inputSize;
        c.imageMean = imageMean;
        c.imageStd = imageStd;

        // Pre-allocate buffers.
        c.outputNames = new String[] {outputName};
        c.intValues = new int[inputSize * inputSize];
//        c.floatValues = new float[inputSize * inputSize * 3];
        c.floatValues = new float[inputSize * inputSize];
        c.outputs = new float[numClasses];

        return c;
    }

    /**
     * \
     * @param bitmap 原bitmap位示图
     * @param schema 模式 0:(最大值+最小值)/2.1: (red+blue+green)/3.2:red*0.3+green*0.59+0.11*blue
     * @return 返回灰度的Bitmap
     */
    public Bitmap gray(Bitmap bitmap, int schema)
    {
        Bitmap bm = Bitmap.createBitmap(bitmap.getWidth(),bitmap.getHeight(), bitmap.getConfig());
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        for(int row=0; row<height; row++){
            for(int col=0; col<width; col++){
                int pixel = bitmap.getPixel(col, row);// ARGB
                int red = Color.red(pixel); // same as (pixel >> 16) &0xff
                int green = Color.green(pixel); // same as (pixel >> 8) &0xff
                int blue = Color.blue(pixel); // same as (pixel & 0xff)
                int alpha = Color.alpha(pixel); // same as (pixel >>> 24)
                int gray = 0;
                if(schema == 0)
                {
                    gray = (Math.max(blue, Math.max(red, green)) +
                            Math.min(blue, Math.min(red, green))) / 2;
                }
                else if(schema == 1)
                {
                    gray = (red + green + blue) / 3;
                }
                else if(schema == 2)
                {
                    gray = (int)(0.3 * red + 0.59 * green + 0.11 * blue);
                }
                Log.d("12","gray:"+gray);
                bm.setPixel(col, row, Color.argb(alpha, gray, gray, gray));
            }
        }
        return bm;
    }
    //将图像像素数据转为一维数组，isReverse判断是否需要反化图像
    public int[] getGrayPix_R(Bitmap bp,boolean isReverse){
        int[]pxs=new int[784];
        int acc=0;
        for(int m=0;m<28;++m){
            for(int n=0;n<28;++n){
                if(isReverse)
                    pxs[acc]=255-Color.red(bp.getPixel(n,m));
                else
                    pxs[acc]=Color.red(bp.getPixel(n,m));
                Log.d("12","gray_"+acc+":"+pxs[acc]+"_");
                ++acc;
            }
        }
        return pxs;

    }

    @Override
    public List<Recognition> recognizeImage(final Bitmap bitmap) {
        // Log this method so that it can be analyzed with systrace.
//        Trace.beginSection("recognizeImage");
//
//        Trace.beginSection("preprocessBitmap");
//        Trace.endSection();
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        Bitmap greyBitmap=gray(bitmap,2);
        intValues=getGrayPix_R(greyBitmap,true);
/*        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3 + 0] = Math.abs((((val >> 16) & 0xFF) - imageMean) / imageStd/255);
            floatValues[i * 3 + 1] = Math.abs((((val >> 8) & 0xFF) - imageMean) / imageStd/255);
            floatValues[i * 3 + 2] = Math.abs(((val & 0xFF) - imageMean) / imageStd/255);
        }*/
        for (int i = 0; i < intValues.length; ++i) {
            floatValues[i]=(float)(intValues[i]/255.0);
        }

        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
//        inferenceInterface.feed(inputName, floatValues, 1,32*32*3);
        inferenceInterface.feed(inputName, floatValues, 1,inputSize*inputSize);


        Trace.endSection();

        // Run the inference call.
        Trace.beginSection("run");
        inferenceInterface.run(outputNames);
        Trace.endSection();

        // Copy the output Tensor back into the output array.
        Trace.beginSection("fetch");
        inferenceInterface.fetch(outputName, outputs);
        Trace.endSection();

        // Find the best classifications.
        PriorityQueue<Recognition> pq =
                new PriorityQueue<Recognition>(
                        3,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });
        for (int i = 0; i < outputs.length; ++i) {
            if (outputs[i] > THRESHOLD) {
                pq.add(
                        new Recognition(
                                "" + i, labels.size() > i ? labels.get(i) : "unknown", outputs[i], null));
            }
        }
        final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }
        Trace.endSection(); // "recognizeImage"
        return recognitions;
    }

    @Override
    public void enableStatLogging(boolean logStats) {
        this.logStats = logStats;
    }

    @Override
    public String getStatString() {
        return inferenceInterface.getStatString();
    }

    @Override
    public void close() {
        inferenceInterface.close();
    }
}