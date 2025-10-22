package org.orangepalantir;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.SessionFunction;
import org.tensorflow.Signature;
import org.tensorflow.op.math.Imag;

import org.tensorflow.DeviceSpec;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class PredictionApp {
    SavedModelBundle bundle;
    boolean broken = false;
    Session s;
    Signature sig;
    SessionFunction fun;
    public PredictionApp(String filename){
        bundle = SavedModelBundle.load(filename);
        s = bundle.session();
        s.initialize();
        fun = bundle.function("serving_default");
        sig = fun.signature();
    }

    public List<ImagePlus> predictImage(ImagePlus plus){
        List<ImagePlus> results = new ArrayList<>();
        int c = plus.getNChannels();
        int s = plus.getNSlices();
        int w = plus.getWidth();
        int h = plus.getHeight();

        for(int frame = 0; frame<plus.getNFrames(); frame++){

            ImageStack stack = plus.getStack();

            int frame_offset = c*s*frame;

            byte[] data = new byte[4 * w*h*s*c];
            FloatBuffer buffer = ByteBuffer.wrap(data).asFloatBuffer();
            for(int i = 0; i<c*s; i++){
                FloatProcessor proc = stack.getProcessor(frame_offset + 1 + i).convertToFloatProcessor();
                buffer.put( (float[])proc.getPixels());
            }
            FloatPredictor predictor = new FloatPredictor(sig);
            predictor.setData(data, c, w, h, s);
            List<FloatPredictor.OutputMapper> prediction = predictor.predict(fun);

            for(int i = 0; i<prediction.size(); i++){
                FloatPredictor.OutputMapper mapper = prediction.get(i);
                ImagePlus op = FloatRunner.toImage(mapper.bdata, mapper.oc, mapper.ow, mapper.oh, mapper.od, plus);

                if(frame == 0){
                    results.add(op);
                } else{
                    ImagePlus existing = results.get(i);
                    ImageStack exStack = existing.getStack();
                    ImageStack next = op.getStack();
                    for(int p = 0; p<next.size(); p++){
                        exStack.addSlice(next.getProcessor(p+1));
                    }
                    existing.setStack(stack, op.getNChannels(), op.getNSlices(), frame+1);
                }
            }


        }

        return results;
    }

    public static void main(String[] args){
        long start = System.currentTimeMillis();
        DeviceSpec gpuSpec = DeviceSpec.newBuilder().deviceType(DeviceSpec.DeviceType.GPU).deviceIndex(0).build();
        System.out.println(gpuSpec);
        //Loads image loads model and makes a prediction.
        ImagePlus plus = new ImagePlus(Paths.get(args[1]).toAbsolutePath().toString());
        ImagePlus iso = App.getIsoTropicFrame(plus, 0);
        PredictionApp app = new PredictionApp(args[0]);
        List<ImagePlus> predictions = app.predictImage(iso);
        
        int c = 0;
        for(ImagePlus p: predictions){
            IJ.save(p, "predicted-c"+ ( c++ ) + ".tiff");
        }
        long end = System.currentTimeMillis();
        System.out.println( ( (end - start)/1000.0 ) + " seconds" );

    }
}
