# JavaTF2ModelRunner

## App
a demo of using tensorflow to run a model on an image.

To run this, compile with maven as normal.

    mvn package

Then execute the main class.

mvn exec:java -D"exec.mainClass"="org.orangepalantir.App" -Dexec.args="$MODEL $IMAGE"

## FloatPredictor

Abstracts away the ImagePlus. Images are a float array with associated channels, slices, height and width.
## PredictionServer and PredictionClient

A very simple socket server client implementation for sending and recieving images and predictions via a socket.

## Troubleshooting

This error seems to happen when I don't have enough memory
for the model.

>Failed to get convolution algorithm. This is probably because cuDNN failed to initialize,

To address that I set the env variable `TF_FORCE_GPU_ALLOW_GROWTH`