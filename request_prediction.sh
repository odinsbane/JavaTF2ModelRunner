#!/bin/bash

image=$1
host=localhost
#starts a server that will make predictions using the provided model
mvn exec:java -D"exec.mainClass"="org.orangepalantir.PredictionClient" -Dexec.args="$image $host"

