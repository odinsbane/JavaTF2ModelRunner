package org.orangepalantir;

import ij.ImageJ;
import ij.ImagePlus;
import ij.ImageStack;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.Socket;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class PredictionClient {
    Socket server;

    public PredictionClient(Socket s){
        server = s;
    }

    public void process(ImagePlus plus) throws IOException, ExecutionException, InterruptedException {
        ExecutorService sending = Executors.newFixedThreadPool(1);
        List<Future<Integer>> finishing = new ArrayList<>();
        OutputStream os = server.getOutputStream();
        DataOutputStream dos = new DataOutputStream(os);
        final InputStream in = server.getInputStream();
        DataInputStream din = new DataInputStream(in);
        dos.writeInt(plus.getNFrames());
        for(int i = 0; i<plus.getNFrames(); i++){
            final int frame = i;
            Future<Integer> future = sending.submit(()->{
                try {
                    System.out.println("writing frame: " + frame);
                    byte[] data = FloatRunner.getImageData(plus, frame);
                    dos.writeInt(plus.getNChannels());
                    dos.writeInt(plus.getWidth());
                    dos.writeInt(plus.getHeight());
                    dos.writeInt(plus.getNSlices());
                    os.write(data);
                    System.out.println("written: " + data.length);
                    return frame;
                } catch(IOException e){
                    throw new RuntimeException(e);
                }
            });
            finishing.add(future);
        }

        List<ImagePlus> pluses = new ArrayList<>();
        for(Future<Integer> result: finishing){
            int frame = result.get();
            int outputs = din.readInt();
            System.out.println("reading: " + outputs);
            if(frame == 0) {
                new ImageJ();
            }
            for(int i = 0; i<outputs; i++){
                int c = din.readInt();
                int w = din.readInt();
                int h = din.readInt();
                int s = din.readInt();
                byte[] buffer = new byte[c*w*h*s*4];
                int read = 0;
                while(read < buffer.length){
                    int r = din.read(buffer, read, buffer.length - read);
                    if(r<0) break;

                    read += r;
                }
                System.out.println( "read: " + read + " // " + Arrays.toString(new int[]{c, w, h, s}));
                ImagePlus op = FloatRunner.toImage(buffer, c, w, h, s, plus);

                if(frame == 0){
                    op.setTitle(i + " created from " + plus.getShortTitle());
                    op.show();
                    pluses.add(op);
                } else{
                    ImagePlus or = pluses.get(i);
                    ImageStack stack = or.getStack();
                    ImageStack fresh = op.getStack();
                    int nc = or.getNChannels();
                    int ns = or.getNSlices();
                    for(int j = 1; j<=fresh.size(); j++){
                        stack.addSlice(fresh.getProcessor(j));
                    }
                    or.setStack(stack,nc, ns, (frame + 1));
                    or.setOpenAsHyperStack(true);
                }

            }
        }
    }

    public static void main(String[] args) throws IOException, ExecutionException, InterruptedException {
        String img = Paths.get(args[0]).toAbsolutePath().toString();

        ImagePlus plus = new ImagePlus(img);
        Socket s = new Socket(args[1], 5050);

        PredictionClient client = new PredictionClient(s);
        client.process(plus);


    }
}
