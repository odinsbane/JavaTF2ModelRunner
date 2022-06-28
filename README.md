# JavaTF2ModelRunner
a demo of using tensorflow to run a model on an image.

To run this, compile with maven as normal.

    mvn package

Then execute the main class.

mvn exec:java -D"exec.mainClass"="org.orangepalantir.App" -Dexec.args="$MODEL $IMAGE"


## Troubleshooting

This error seems to happen when I don't have enough memory
for the model.

>Failed to get convolution algorithm. This is probably because cuDNN failed to initialize,