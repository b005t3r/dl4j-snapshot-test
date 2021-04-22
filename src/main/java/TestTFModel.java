import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Map;

import static org.bytedeco.opencv.global.opencv_core.CV_8UC3;

public class TestTFModel {
    public static void main(String[] args) throws IOException {
        testPoseRecognition();
    }

    public static void testPoseRecognition() throws IOException {
        final File inputImageFile   = new File("data/img.png");
        final File modelFile        = new File("data/ssdlite_mobilenet_v2_coco_2018_05_09_frozen_inference_graph.pb");
        final int inputTensorSize   = 100;

        BufferedImage image = ImageIO.read(inputImageFile);
        toBGR(image);

        Mat tmpFrame = imageToMat(image, null);
        Mat tmpScaledFrame = new Mat();
        opencv_imgproc.resize(tmpFrame, tmpScaledFrame, new org.bytedeco.opencv.opencv_core.Size(inputTensorSize, inputTensorSize), 0, 0, opencv_imgproc.CV_INTER_AREA);
        ByteBuffer buffer = tmpScaledFrame.createBuffer();
        byte[] data = new byte[inputTensorSize * inputTensorSize * 3];
        buffer.get(data);
        INDArray input = Nd4j.create(data, new long[]{1, inputTensorSize, inputTensorSize, 3}, DataType.UINT8);

        SameDiff sd = SameDiff.importFrozenTF(modelFile);
        //System.out.println(sd.getVariable("image_tensor"));

        Map<String, INDArray> result = sd.batchOutput()
                .input("image_tensor", input)
                .output("detection_boxes", "detection_classes", "detection_scores")
                .output();

        System.out.println(result);
    }

    private static void toBGR(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();

        // Get the int type representation of each rgb pixel in the picture
        int[] rgbPixels = image.getRGB(0, 0, width, height, null, 0, width);
        int[] bgrPixels = new int[rgbPixels.length];

        for (int i = 0; i < rgbPixels.length; i++) {
            int color = rgbPixels[i];
            int red = ((color & 0x00FF0000) >> 16);
            int green = ((color & 0x0000FF00) >> 8);
            int blue = color & 0x000000FF;
            // Combine the values ​​of the three channels of rgb into one int value, and swap the b channel and r channel at the same time
            bgrPixels[i] = (red & 0x000000FF) | (green << 8 & 0x0000FF00) | (blue << 16 & 0x00FF0000);
        }

        image.setRGB(0, 0, width, height, bgrPixels, 0, width);
    }

    private static Mat imageToMat(BufferedImage img, Mat result) {
        WritableRaster wr = img.getRaster();
        byte[] bufferPixels = ((DataBufferByte) wr.getDataBuffer()).getData();

        int step = bufferPixels.length / img.getHeight();

        if(result == null || result.type() != CV_8UC3 || result.cols() != img.getWidth() || result.rows() != img.getHeight() || result.step() != step) {
            result = new Mat(img.getHeight(), img.getWidth(), CV_8UC3, new BytePointer(bufferPixels), step);
        }
        else {
            result.data().put(bufferPixels, 0, bufferPixels.length);
        }

        return result;
    }
}
