package org.orangepalantir;

import org.tensorflow.SessionFunction;
import org.tensorflow.Signature;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.FloatDataBuffer;
import org.tensorflow.types.TFloat32;
import org.tensorflow.Result;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FloatPredictor {
    int batch_size = 4;
    int nBatches;
    int w;
    int h;
    int d;
    int c;
    String name;
    float[] batchBuffer;
    FloatBuffer data;

    List<int[]> tiles = new ArrayList<>();

    int vc;
    int vh;
    int vw;
    int vd;

    int frame = 0;
    public FloatPredictor( Signature signature){
        Map<String, Signature.TensorDescription> descriptions = signature.getInputs();
        for( String k: descriptions.keySet()){
            Signature.TensorDescription description = descriptions.get(k);
            Shape s = description.shape;
            name = k;
            c = (int)s.size(1);
            d = (int)s.size(2);
            h = (int)s.size(3);
            w = (int)s.size(4);
        }

        batchBuffer = new float[c*d*h*w*batch_size];
    }

    class OutputMapper{
        byte[] bdata;
        FloatBuffer pixels;
        int oc, od, oh, ow;
        int pd, ph, pw;
        String oname;
        public OutputMapper(Tensor t){
            Shape s = t.shape();
            //these are per sample sizes.
            int channels = (int)s.get(1);
            int depth = (int)s.get(2);
            int height = (int)s.get(3);
            int width = (int)s.get(4);


            prepareGeometry(channels, depth, height, width);
        }

        public void setName(String name){
            oname = name;
        }
        public String getName(){
            return oname;
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

            bdata = new byte[4 * od * oc * ow * oh];
            pixels = ByteBuffer.wrap(bdata).asFloatBuffer();
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
            double min = Double.MAX_VALUE;
            double max = -min;
            for (int i = 0; i < oc; i++) {
                int depth_offset = i + origin[0] * oc;
                for (int j = dlow; j < dhigh; j++) {
                    int nz = depth_offset + oc * j;
                    //float[] p = pixels.get(nz);
                    int proc_offset = ow*oh * nz;
                    for (int k = ylow; k < yhigh; k++) {
                        for (int m = xlow; m < xhigh; m++) {
                            int x = m + origin[2];
                            int y = k + origin[1];
                            int t = m + k*pw + j * ( pw * ph) + i*pw*ph*pd;
                            float v = data[t + batch_offset];
                            if(v < min) min = v;
                            if(v > max) max = v;
                            pixels.put(proc_offset + y*ow + x, data[t + batch_offset] );
                        }
                    }
                }
            }
            System.out.println("out: " + tile + ", " + min + ", " + max);
        }
    }
    /**
     * The data is an array of byte's representing an array of floats eg. 4 bytes per
     * pixel.
     * @param data
     * @param channels
     * @param width
     * @param height
     * @param slices
     */
    public void setData(byte[] data, int channels, int width, int height, int slices){
        vc = channels;
        vh = height;
        vw = width;
        vd = slices;
        this.data = ByteBuffer.wrap(data).asFloatBuffer();
        createVolumeTiles();
    }

    public void createVolumeTiles(){
        tiles.clear();
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

    float[] getBatch(int n){
        int tile_index = n*batch_size;
        int bd = c*d*h*w;
        for(int i = 0; i<batch_size; i++){
            bufferTile(i + tile_index, i*bd);
        }
        return batchBuffer;
    }

    public void bufferTile(int tile, int batch_offset){
        int[] origin = tiles.get(tile);
        System.out.print(".");
        int t = 0;

        float[] factors = new float[c];
        float[] means = new float[c];
        //Batch Normalize
        for (int i = 0; i < c; i++) {
            double mn = 0;
            double m2 = 0;
            int count = 0;

            for(int j = 0; j<d; j++) {
                //processor order is xycz
                int z = j + origin[0];
                int proc_offset = vw*vh * (i + vc*z);

                for (int k = 0; k < h; k++) {
                    int y = k + origin[1];

                    int px_offset = proc_offset + vw*y;

                    for (int m = 0; m < w; m++) {
                        int x = m + origin[2];

                        float f =  data.get(px_offset + x);
                        mn += f;
                        m2 += f*f;
                        count ++;
                    }
                }
            }

            mn = mn/count;
            float std = (float)Math.sqrt(m2/count - mn*mn);
            means[i] = (float)mn;
            factors[i] = std > 1.0e-3 ? 1f/std : 1;
            System.out.println("in, " + tile + ", " + mn + ", " + std);
        }

        for(int i = 0; i<c; i++){
            for(int j = 0; j<d; j++){
                /**
                 * Data is a float[] representing a single timepoint. If we want the pixel from
                 * channel ci, slice s, x and y.
                 * int proc_index = channels*s + ci
                 *
                 * int index = width*height ( channels * s + c ) + width*y + x
                 *
                 * float p = data[index]
                 *
                 */
                //ImageProcessor p = stack.getProcessor(channel_offset + c*j + frame_offset + 1);
                int z = j + origin[0];
                int proc_offset = vw*vh * ( vc * z + i);

                for(int k = 0; k<h; k++){

                    for(int m = 0; m<w; m++){
                        int x = m + origin[2];
                        int y = k + origin[1];
                        float f = data.get(proc_offset + vw*y + x);
                        batchBuffer[t + batch_offset] = (f - means[i])*factors[i];
                        t++;
                    }

                }
            }
        }

    }

    OutputMapper getOutputMapper(Tensor t){
        return new OutputMapper(t);
    }

    /**
     * Workhorse method. Goes through the whole image makes the predictions and returns the results
     * as 'float[]'s`
     *
     * @return
     */
    List<OutputMapper> predict(SessionFunction fun){
        Map<String, OutputMapper> results = new HashMap<>();
        try(Tensor input = TFloat32.tensorOf(Shape.of(batch_size, c, d, h, w));){
            Map<String, Tensor> inputs = new HashMap<>();
            inputs.put(name, input);
            for(int i = 0; i<nBatches; i++){
                float[] batch = getBatch(i);
                FloatDataBuffer ibf = input.asRawTensor().data().asFloats().offset(0);
                ibf.write(batch);
                Result out = fun.call(inputs);
                for(String key: out.keySet()){
                    final Tensor outTensor = out.get(key).get();
                    OutputMapper channel = results.computeIfAbsent(key, k->{
                        OutputMapper mapper = getOutputMapper(outTensor);
                        mapper.setName(k);
                        return mapper;
                    });

                    float[] batch_buffer = new float[(int)outTensor.size()];
                    FloatDataBuffer dbf = outTensor.asRawTensor().data().asFloats();
                    dbf.read(batch_buffer);
                    channel.writeBatch(batch_buffer, i);
                    outTensor.close();
                }
            }

        }

        return new ArrayList<>(results.values());

    }


}
