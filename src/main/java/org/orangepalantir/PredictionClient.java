package org.orangepalantir;

import ij.ImageJ;
import ij.ImagePlus;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.Socket;
import java.nio.file.Paths;
import java.util.Arrays;

public class PredictionClient {
    Socket server;
    public PredictionClient(Socket s){
        server = s;
    }

    public void process(ImagePlus plus) throws IOException {
        byte[] data = FloatRunner.getImageData(plus, 0);
        OutputStream os = server.getOutputStream();
        DataOutputStream dos = new DataOutputStream(os);
        dos.writeInt(plus.getNChannels());
        dos.writeInt(plus.getWidth());
        dos.writeInt(plus.getHeight());
        dos.writeInt(plus.getNSlices());
        os.write(data);
        System.out.println("written: " + data.length);
        InputStream in = server.getInputStream();
        DataInputStream din = new DataInputStream(in);
        int outputs = din.readInt();
        System.out.println("reading: " + outputs);
        new ImageJ();
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
            op.setTitle(i + " created from " + plus.getShortTitle());
            op.show();

        }

    }

    public static void main(String[] args) throws IOException {
        String img = Paths.get(args[0]).toAbsolutePath().toString();

        ImagePlus plus = new ImagePlus(img);
        Socket s = new Socket("localhost", 5050);

        PredictionClient client = new PredictionClient(s);
        client.process(plus);


    }
}
