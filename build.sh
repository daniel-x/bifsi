#!/bin/bash
mkdir -p build
g++ -std=c++17 -Wall -g -O3 src/*.cpp -o build/test
