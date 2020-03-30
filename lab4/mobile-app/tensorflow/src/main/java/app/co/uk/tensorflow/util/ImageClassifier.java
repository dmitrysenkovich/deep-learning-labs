package app.co.uk.tensorflow.util;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Build;

import androidx.annotation.RequiresApi;

import app.co.uk.tensorflow.model.Result;
import io.reactivex.Single;
import org.tensorflow.lite.Interpreter;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.*;

import static app.co.uk.tensorflow.util.Keys.DIM_BATCH_SIZE;
import static app.co.uk.tensorflow.util.Keys.DIM_IMG_SIZE_X;
import static app.co.uk.tensorflow.util.Keys.DIM_IMG_SIZE_Y;
import static app.co.uk.tensorflow.util.Keys.DIM_PIXEL_SIZE;
import static app.co.uk.tensorflow.util.Keys.INPUT_SIZE;
import static app.co.uk.tensorflow.util.Keys.MODEL_PATH;

@RequiresApi(api = Build.VERSION_CODES.CUPCAKE)
public class ImageClassifier {

    private AssetManager assetManager;
    private Interpreter interpreter;
    private ByteBuffer imgData;
    private int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];

    public ImageClassifier(AssetManager assetManager) {
        this.assetManager = assetManager;
        imgData = ByteBuffer.allocateDirect(DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE * 4);
        imgData.order(ByteOrder.nativeOrder());
        try {
            interpreter = new Interpreter(loadModelFile(assetManager, MODEL_PATH));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        imgData.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < DIM_IMG_SIZE_X; i++) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; j++) {
                int value = intValues[pixel++];
                imgData.putFloat((shr(value, 16) & 0xFF) / 255f);
                imgData.putFloat((shr(value, 8) & 0xFF) / 255f);
                imgData.putFloat((value & 0xFF) / 255f);
            }
        }
    }

    public static int shr(int i, int distance) {
        return (i >>> distance) | (i << -distance);
    }

    @RequiresApi(api = Build.VERSION_CODES.CUPCAKE)
    private MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename) {
        AssetFileDescriptor fileDescriptor = null;
        try {
            fileDescriptor = assets.openFd(modelFilename);
        } catch (IOException e) {
            e.printStackTrace();
        }
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        try {
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public String recognizeImage(Bitmap bitmap) {
        convertBitmapToByteBuffer(bitmap);
        Object[] inputs = new Object[]{imgData};
        Map<Integer, Object> outputs = new HashMap();
        outputs.put(0, new float[][] { { 128f, 128f, 128f, 128f, 128f } });
        outputs.put(1, new float[][] { { 128f, 128f, 128f, 128f, 128f, 128f, 128f, 128f, 128f, 128f, 128f } });
        outputs.put(2, new float[][] { { 128f, 128f, 128f, 128f, 128f, 128f, 128f, 128f, 128f, 128f, 128f } });
        outputs.put(3, new float[][] { { 128f, 128f, 128f, 128f, 128f, 128f, 128f, 128f, 128f, 128f, 128f } });
        outputs.put(4, new float[][] { { 128f, 128f, 128f, 128f, 128f, 128f, 128f, 128f, 128f, 128f, 128f } });
        outputs.put(5, new float[][] { { 128f, 128f, 128f, 128f, 128f, 128f, 128f, 128f, 128f, 128f, 128f } });
        interpreter.runForMultipleInputsOutputs(inputs, outputs);
        int numbersCountRecognized = getMaxProbLabel(((float[][]) outputs.get(0))[0]) + 1;
        String result = "";
        for (int i = 0; i < numbersCountRecognized; i++) {
            int label = getMaxProbLabel(((float[][]) outputs.get(i+1))[0]);
            result += String.valueOf(label);
        }
        return result;
    }

    private int getMaxProbLabel(float[] probs) {
        float max = Float.MIN_VALUE;
        int maxIndex = -1;
        for (int i = 0; i < probs.length; i++) {
            if (probs[i] > max) {
                max = probs[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public void close() {
        interpreter.close();
    }
}

