#!/bin/bash

mkdir ../data
cd ../data
wget http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip
unzip ModelNet10.zip
rm -rf __MACOSX
rm ModelNet10.zip