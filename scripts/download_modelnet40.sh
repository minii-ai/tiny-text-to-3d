#!/bin/bash

mkdir ../data
cd ../data
wget http://modelnet.cs.princeton.edu/ModelNet40.zip
unzip ModelNet40.zip
rm -rf __MACOSX
rm ModelNet40.zip