package org.orangepalantir;

import ij.ImageJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.measure.Calibration;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.SessionFunction;
import org.tensorflow.Signature;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.file.Paths;
import java.util.List;

public class FloatRunner {

    static byte[] getImageData(ImagePlus plus, int frame){
        int c = plus.getNChannels();
        int s = plus.getNSlices();
        int w = plus.getWidth();
        int h = plus.getHeight();
        ImageStack stack = plus.getStack();

        int frame_offset = c*s*frame;

        byte[] data = new byte[4 * w*h*s*c];
        FloatBuffer buffer = ByteBuffer.wrap(data).asFloatBuffer();
        for(int i = 0; i<c*s; i++){
            FloatProcessor proc = stack.getProcessor(frame_offset + 1 + i).convertToFloatProcessor();
            buffer.put( (float[])proc.getPixels());
        }

        return data;
    }
    public static ImagePlus toImage(byte[] data, int oc, int ow, int oh, int os, ImagePlus original){
        ImagePlus dup = original.createImagePlus();
        ImageStack stack = new ImageStack(ow, oh);
        int slices = oc*os;
        FloatBuffer buffer = ByteBuffer.wrap(data).asFloatBuffer();
        for(int i = 0; i<slices; i++){
            ImageProcessor proc = new FloatProcessor(ow, oh);
            float[] pixels = new float[ow*oh];
            buffer.get(pixels, 0, pixels.length);
            proc.setPixels(pixels);
            stack.addSlice(proc);
        }

        dup.setStack(stack, oc, os, 1);
        if(dup.getNSlices() != original.getNSlices() || dup.getHeight() != original.getHeight() || dup.getWidth() != original.getWidth()){
            Calibration c0 = original.getCalibration();
            Calibration c1 = dup.getCalibration();
            c1.pixelDepth = c0.pixelDepth*original.getNSlices() / dup.getNSlices();
            c1.pixelWidth = c0.pixelWidth*original.getWidth() / dup.getWidth();
            c1.pixelHeight = c0.pixelHeight*original.getHeight() / dup.getHeight();
            dup.setCalibration(c1);
        }

        return dup;
    }
    public static ImagePlus toImage(FloatPredictor.OutputMapper output, ImagePlus original){
        return toImage(output.bdata, output.oc, output.ow, output.oh, output.od, original);
    }

    public static void main(String[] args){
        String model = Paths.get(args[0]).toAbsolutePath().toString();
        String img = Paths.get(args[1]).toAbsolutePath().toString();

        SavedModelBundle bundle = SavedModelBundle.load(model);
        ImagePlus plus = new ImagePlus(img);
        byte[] data = getImageData(plus, 0);

        try(Session s = bundle.session();) {

            s.initialize();
            SessionFunction fun = bundle.function("serving_default");
            Signature sig = fun.signature();
            FloatPredictor predictor = new FloatPredictor(sig);
            predictor.setData(data, plus.getNChannels(), plus.getWidth(), plus.getHeight(), plus.getNSlices());
            List<FloatPredictor.OutputMapper> out = predictor.predict(fun);
            new ImageJ();
            toImage(data, plus.getNChannels(), plus.getWidth(), plus.getHeight(), plus.getNSlices(), plus).show();
            out.forEach( o ->{
                ImagePlus ip = toImage(o, plus);
                ip.setTitle(o.getName() + "-" + plus.getShortTitle());
                ip.show();
            });
        }
    }

}
