#!/bin/bash

model=$1
#starts a server that will make predictions using the provided model
mvn exec:java -D"exec.mainClass"="org.orangepalantir.PredictionServer" -Dexec.args="$model"

