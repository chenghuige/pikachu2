#!/bin/bash
g++ $1 -std=gnu++11 -I /usr/include/eigen3 -I /usr/include/google/tensorflow -I /usr/include/google/tensorflow/tensorflow/contrib/makefile/downloads/absl -I /usr/include/google/tensorflow/tensorflow/contrib/makefile/downloads/protobuf/src -I /usr/include/google/tensorflow/tensorflow/contrib/makefile/downloads/googletest/googletest/include/ -ltensorflow_cc -ltensorflow_framework
