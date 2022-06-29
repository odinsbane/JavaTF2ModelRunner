package org.orangepalantir;

import ij.ImageJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.SessionFunction;
import org.tensorflow.Signature;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.FloatDataBuffer;
import org.tensorflow.types.TFloat32;

import java.awt.image.DataBufferFloat;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class App {

    static SavedModelBundle bundle;
    int batch_size = 4;
    int nBatches;
    int w;
    int h;
    int d;
    int c;
    String name;
    float[] batchBuffer;
    List<int[]> tiles = new ArrayList<>();
    ImagePlus volume;

    int vc;
    int vh;
    int vw;
    int vd;

    int frame = 0;
    public App(int batch_size, Signature sig){
        Map<String, Signature.TensorDescription> descriptions = sig.getInputs();
        this.batch_size = batch_size;
        for( String k: descriptions.keySet()){
            Signature.TensorDescription description = descriptions.get(k);
            Shape s = description.shape;
            name = k;
            c = (int)s.size(1);
            d = (int)s.size(2);
            h = (int)s.size(3);
            w = (int)s.size(4);
        }

        batchBuffer = new float[c*d*h*w*this.batch_size];
    }
    public int batches(){
        return nBatches;
    }

    public void setData(ImagePlus volume){
        System.out.println( volume);
        vc = volume.getNChannels();
        vh = volume.getHeight();
        vw = volume.getWidth();
        vd = volume.getNSlices();
        this.volume = volume;
        createVolumeTiles();
    }

    public void createVolumeTiles(){
        int strideZ = d/2;
        int strideY = h/2;
        int strideX = w/2;

        int nz = vd/strideZ;
        int ny = vh/strideY;
        int nx = vw/strideX;
        int x0, y0, z0;
        for(int i = 0; i<nz; i++){
            for(int j = 0; j<ny; j++){
                for(int k = 0; k<nx; k++){
                    x0 = strideX*k;
                    y0 = strideY*j;
                    z0 = strideZ*i;

                    if( x0 + w > vw){
                        x0 = vw - w;
                    }
                    if( y0 + h > vh){
                        y0 = vh - h;
                    }
                    if( z0 + d > vd){
                        z0 = vd - d;
                    }

                    tiles.add(new int[]{
                            z0, y0, x0
                    });
                }
            }
        }
        //ommit or duplicate.
        while(tiles.size()%batch_size != 0){
            tiles.remove(tiles.size() - 1);
        }
        nBatches = tiles.size()/batch_size;
        System.out.println(tiles.size() + " tiles " + nBatches + " batches");

    }

    class OutputChannel{
        List<float[]> pixels = new ArrayList<>();
        int oc, od, oh, ow;
        int pd, ph, pw;

        public OutputChannel(Tensor t){
            Shape s = t.shape();

            //these are per sample sizes.
            int channels = (int)s.size(1);
            int depth = (int)s.size(2);
            int height = (int)s.size(3);
            int width = (int)s.size(4);


            prepareGeometry(channels, depth, height, width);
        }

        public OutputChannel(int channels, int depth, int height, int width){
            prepareGeometry(channels, depth, height, width);
        }

        /**
         * Generates an output space relative to the input tile size and the input image size.
         *
         * @param channels
         * @param depth
         * @param height
         * @param width
         */
        private void prepareGeometry(int channels, int depth, int height, int width){
            //patch sizes.
            pd = depth;
            ph= height;
            pw = width;

            oc = channels;

            //total output size scales to relative input patch size and this patch size.
            od = vd*pd/d;
            oh = vh*ph/h;
            ow = vw*pw/w;

            System.out.println(Arrays.toString(new int[]{
                    pd, ph, pw, od, oh, ow
            }));

            int im_processors = volume.getNFrames()*od*channels;
            for(int i = 0; i<im_processors; i++){
                pixels.add(new float[oh*ow]);
            }
        }

        public void writeBatch(float[] data, int batch ){
            int floats_per_batch = oc*pd*ph*pw;
            for(int i = 0; i<batch_size; i++){
                int tile_no = batch*batch_size + i;
                int batch_offset = i*floats_per_batch;
                writeTile(data, tile_no, batch_offset);
            }
        }
        int[] scaledOrigin(int[] original) {
            //scale the coordinate based on the patch sizes.
            return new int[]{
                    original[0] * pd / d,
                    original[1] * ph / h,
                    original[2] * pw / w
            };
        }

        public void writeTile(float[] data, int tile, int batch_offset) {
            int[] origin = scaledOrigin(tiles.get(tile));
            int frame_offset = frame * oc * od;

            //full patch is written to the output.
            int dlow = 0;
            int dhigh = pd;
            int ylow =0;
            int yhigh = ph;
            int xlow = 0;
            int xhigh = pw;

            if(origin[0] > 0){
                dlow = pd/4;
            }
            if( origin[0] + pd < od){
                dhigh = 3*pd/4;
            }
            if(origin[1] > 0){
                ylow = ph/4;
            }
            if(origin[1] + ph < oh){
                yhigh = 3*ph/4;
            }
            if(origin[2] > 0){
                xlow = pw/4;
            }
            if(origin[2] + pw < ow){
                xhigh = 3*pw/4;
            }

            for (int i = 0; i < oc; i++) {
                int depth_offset = i + origin[0] * oc;
                for (int j = dlow; j < dhigh; j++) {
                    int nz = depth_offset + oc * j + frame_offset;
                    float[] p = pixels.get(nz);
                    for (int k = ylow; k < yhigh; k++) {
                        for (int m = xlow; m < xhigh; m++) {
                            int x = m + origin[2];
                            int y = k + origin[1];
                            int t = m + k*pw + j * ( pw * ph) + i*pw*ph*pd;
                            float f = data[t + batch_offset];
                            p[x + y*ow] = f;
                        }
                    }
                }
            }
        }

        public ImagePlus createImage(ImagePlus plus){

            ImagePlus dest = plus.createImagePlus();
            ImageStack stack = new ImageStack(ow, oh);
            for(float[] pxs: pixels){
                ImageProcessor p = new FloatProcessor(ow, oh, pxs);
                stack.addSlice(p);
            }
            dest.setStack(stack, oc, od, volume.getNFrames());
            return dest;
        }
    }

    public void getTile(float[] data, int tile, int batch_offset){
        int frame_offset = volume.getNChannels()*volume.getNSlices()*frame;

        int[] origin = tiles.get(tile);
        ImageStack stack = volume.getStack();
        int t = 0;
        for(int i = 0; i<c; i++){
            int channel_offset = i + origin[0]*c;
            for(int j = 0; j<d; j++){

                ImageProcessor p = stack.getProcessor(channel_offset + c*j + frame_offset + 1);

                for(int k = 0; k<h; k++){
                    for(int m = 0; m<w; m++){
                        int x = m + origin[2];
                        int y = k + origin[1];
                        batchBuffer[t + batch_offset] = p.getf(x, y);
                        t++;
                    }

                }
            }
        }

    }

    float[] getBatch(int n){
        int tile_index = n*batch_size;
        int bd = c*d*h*w;
        for(int i = 0; i<batch_size; i++){
            getTile(batchBuffer, i + tile_index, i*bd);

        }
        return batchBuffer;
    }

    public OutputChannel getOutputChannel(Tensor t){
        return new OutputChannel(t);
    }

    public static void main( String[] args ){
        long start = System.currentTimeMillis();
        SavedModelBundle bundle = SavedModelBundle.load(args[0]);
        new ImageJ();
        ImagePlus plus = new ImagePlus(Paths.get(args[1]).toAbsolutePath().toString());
        try( Session s = bundle.session();){

            s.initialize();
            SessionFunction fun = bundle.function("serving_default");
            Signature sig = fun.signature();
            App app = new App(4, sig);
            app.setData(plus);

            try(Tensor input = TFloat32.tensorOf(Shape.of(app.batch_size, app.c, app.d, app.h, app.w));){

                OutputChannel original = app.getOutputChannel(input);
                Map<String, Tensor> inputs = new HashMap<>();
                inputs.put(app.name, input);
                Map<String, OutputChannel> results = new HashMap<>();

                for(int frame = 0; frame<plus.getNFrames(); frame++){
                    app.frame = frame;
                    for(int i = 0; i<app.nBatches; i++){
                        float[] batch = app.getBatch(i);
                        FloatDataBuffer ibf = input.asRawTensor().data().asFloats().offset(0);
                        ibf.write(batch);
                        Map<String, Tensor> out = fun.call(inputs);
                        original.writeBatch(batch, i);
                        for(String key: out.keySet()){
                            final Tensor outTensor = out.get(key);
                            App.OutputChannel channel = results.computeIfAbsent(key, k->app.getOutputChannel(outTensor));

                            float[] batch_buffer = new float[(int)outTensor.size()];
                            FloatDataBuffer dbf = outTensor.asRawTensor().data().asFloats();
                            dbf.read(batch_buffer);
                            channel.writeBatch(batch_buffer, i);
                            outTensor.close();
                        }
                    }
                }

                ImagePlus o = original.createImage(plus);
                o.setTitle("input");
                o.show();
                for(String key: results.keySet()){
                    ImagePlus opp = results.get(key).createImage(plus);
                    opp.setTitle(key);
                    opp.show();
                }
            }
        }
        long end = System.currentTimeMillis();
        System.out.println( ( (end - start)/1000.0 ) + " seconds" );
    }
}
