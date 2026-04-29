#!/bin/bash

module add maven-3.9.0

mvn clean install

nice -n 19 java -cp target/neuralNetwork-1.0-SNAPSHOT.jar cz.muni.fi.pv021.impl.Main
