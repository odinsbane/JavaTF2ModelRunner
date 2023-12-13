#!/bin/bash

image=$1
host=$2
#requests a prediction from a running server
mvn exec:java -D"exec.mainClass"="org.orangepalantir.PredictionClient" -Dexec.args="$image $host"

