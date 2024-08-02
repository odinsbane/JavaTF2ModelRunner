package org.orangepalantir;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.SessionFunction;
import org.tensorflow.Signature;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

/**
 * A simple socket server for creating predictions with a neural network.
 *
 * Creates a ServerSocket, and waits for a client.
 * When a client connects is reads an image worth of data then makes a neural network predictdion.
 */
public class PredictionServer implements AutoCloseable{
    SavedModelBundle bundle;
    boolean broken = false;
    Session s;
    Signature sig;
    SessionFunction fun;
    public PredictionServer(String filename){
        bundle = SavedModelBundle.load(filename);
        s = bundle.session();
        s.initialize();
        fun = bundle.function("serving_default");
        sig = fun.signature();
    }

    public void run() {
        ExecutorService receiver = Executors.newFixedThreadPool(1);
        ExecutorService predicts = Executors.newFixedThreadPool(1);
        ExecutorService sender = Executors.newFixedThreadPool(1);

        try (ServerSocket socket = new ServerSocket(5050);
             Socket client = socket.accept();
        ) {


            InputStream is = client.getInputStream();
            DataInputStream stream = new DataInputStream(is);
            FloatPredictor predictor = new FloatPredictor(sig);
            OutputStream os = client.getOutputStream();

            int frames = stream.readInt();

            System.out.println("Processing image with " + frames + " frames");

            List<Future<Callable<List<FloatPredictor.OutputMapper>>>> reading = new ArrayList<>();

            BlockingQueue<Callable<List<FloatPredictor.OutputMapper>>> toPredict = new ArrayBlockingQueue<>(1);
            BlockingQueue<List<FloatPredictor.OutputMapper>> toSend = new ArrayBlockingQueue(1);

            Future<?> receiving = receiver.submit(() -> {
                try {
                    for (int i = 0; i < frames; i++) {
                        int channels = stream.readInt();
                        int width = stream.readInt();
                        int height = stream.readInt();
                        int slices = stream.readInt();
                        byte[] bytes = new byte[4 * channels * width * height * slices];
                        int read = 0;
                        while (read < bytes.length) {
                            int r = is.read(bytes, read, bytes.length - read);
                            if (r < 0) break;
                            read += r;
                        }
                        toPredict.put(() -> {
                            predictor.setData(bytes, channels, width, height, slices);
                            return predictor.predict(fun);
                        });
                        System.out.println("read frame: " + i);
                    }
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            });
            Future<?> predicting = predicts.submit(() -> {
                for (int i = 0; i < frames; i++) {
                    try {
                        Callable<List<FloatPredictor.OutputMapper>> result = toPredict.take();
                        toSend.add(result.call());
                        System.out.println("predicted frame: " + i);
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                }
            });


            Future<?> sending = sender.submit(() -> {
                try {
                    DataOutputStream dos = new DataOutputStream(os);
                    for (int i = 0; i < frames; i++) {
                        List<FloatPredictor.OutputMapper> result = toSend.take();
                        System.out.println(result.size());
                        dos.writeInt(result.size());
                        for (FloatPredictor.OutputMapper op : result) {
                            System.out.println("  " + op.bdata.length);
                            dos.writeInt(op.oc);
                            dos.writeInt(op.ow);
                            dos.writeInt(op.oh);
                            dos.writeInt(op.od);
                            os.write(op.bdata);
                        }
                        System.out.println("written frame: " + i);
                    }
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            });
            try{
                receiving.get();
            } catch(Exception e){
                predicting.cancel(true);
                sending.cancel(true);
            }
            try{
                predicting.get();
            } catch(Exception e){
                sending.cancel(true);
            }
            sending.get();
        } catch (Exception e) {
            e.printStackTrace();
        }  finally{
            receiver.shutdown();
            sender.shutdown();
            predicts.shutdown();
        }
    }

    public static void main(String[] args) {
        String filename = Paths.get(args[0]).toAbsolutePath().toString();
        try(PredictionServer  server = new PredictionServer(filename);){
            while(!server.broken){
                server.run();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void close() throws Exception {
        s.close();
    }
}
