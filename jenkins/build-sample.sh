#!/bin/bash
##
#
# Script to build xgboost jar files.
# Source tree is supposed to be ready by Jenkins before starting this script.
#
###
set -e

BUILD_ARG="$1" 
CUDA_CLASSIFIER="$2"
# Set CUDA_CLASSIFIER="" if '$CUDA_CLASSIFIER != cuda10'
if [ "$CUDA_CLASSIFIER" == "cuda10" ];then
    echo "cuda classifier using cuda10"
elif [ "$CUDA_CLASSIFIER" == "cuda10-1" ];then
    echo "cuda classifier using cuda10-1"
elif [ "$CUDA_CLASSIFIER" == ""];then
    echo "cuda classifier using default empty, meaning cuda9.2"
else
    echo "unsupported cuda classifier $CUDA_CLASSIFIER, using default empty"
    CUDA_CLASSIFIER=""
fi
BUILD_ARG="$BUILD_ARG -Dcuda.classifier=$CUDA_CLASSIFIER"

if [ "$REPO_TYPE" == "Local" ];then
    BUILD_ARG="$BUILD_ARG -s $WORKSPACE/jenkins/settings.xml -P apt-sh04-repo"
elif [ "$REPO_TYPE" == "Sonatype" ];then
    BUILD_ARG="$BUILD_ARG -P sonatype-repo"
else
   echo "Dependency gpuwa"
fi

# set .m2 dir and force update snapshot dependencies
BUILD_ARG="$BUILD_ARG -Dmaven.repo.local=$WORKSPACE/.m2 -U"
echo "mvn package $BUILD_ARG"

cd examples/apps/scala
mvn $BUILD_ARG clean package
cd -
