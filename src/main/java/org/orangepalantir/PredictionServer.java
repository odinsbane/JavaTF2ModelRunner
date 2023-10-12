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
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

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
            ExecutorService service = Executors.newFixedThreadPool(1);
            ExecutorService sending = Executors.newFixedThreadPool(1);

            for (int i = 0; i < frames; i++) {
                Future<Callable<List<FloatPredictor.OutputMapper>>> toRead = service.submit(() -> {
                    try {
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
                        System.out.println("reading " + channels + ", " + width + ", " + height + ", " + slices);
                        return () -> {
                            predictor.setData(bytes, width, height, slices, channels);
                            return predictor.predict(fun);
                        };
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                });
                reading.add(toRead);
            }


            try {
                List<Future<?>> sent = new ArrayList<>();

                for (Future<Callable<List<FloatPredictor.OutputMapper>>> finished : reading) {
                    List<FloatPredictor.OutputMapper> out = finished.get().call();
                    Future<?> done = sending.submit(() -> {
                        try {
                            DataOutputStream dos = new DataOutputStream(os);
                            System.out.println(out.size());
                            dos.writeInt(out.size());
                            System.out.println("writing outputs.");
                            for (FloatPredictor.OutputMapper op : out) {
                                System.out.println("  " + op.bdata.length);
                                dos.writeInt(op.oc);
                                dos.writeInt(op.ow);
                                dos.writeInt(op.oh);
                                dos.writeInt(op.od);
                                os.write(op.bdata);
                            }
                        } catch (Exception e) {
                            System.out.println("unable to process");
                            e.printStackTrace();
                            broken = true;
                        }
                    });
                    sent.add(done);
                }
                for(Future<?> d : sent){
                    d.get();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
            System.out.println("finished client");
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ExecutionException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
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
