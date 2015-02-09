/* 
 * File:   caffeLink.hpp
 * Author: kerhy
 *
 * Created on November 4, 2014, 9:56 PM
 */

#ifndef CAFFELINK_HPP
#define	CAFFELINK_HPP

#include "WolframLibrary.h"



extern "C" {

/** 
 * Sets flag for data type in net. Doubles: no conversion double->float, more
 * memory. Float: conversion and copy double->float, less memory.
 * @param useDoublesPar \c true for \c doubles
 * @param useGPU \c true for GPU mode
 * @param devID id of cuda device
 */
void initCaffeLink_(bool useDoublesPar, bool useGPU, int deviceID);

/**
 * Returns \c true if net data type is \c double, \c false otherwise.
 * @return
 */
bool isUsingDouble();

/**
 * Calls \c CLnets.prepareNetString with correct type.
 * Creates new net from protobuffer parameters in \n string.
 * @param paramStr protobuffer parameters
 * @return \c true on success, \c false otherwise
 */
bool prepareNetString_(char* paramStr);

/**
 * Calls \c CLnets.prepareNetFile with correct type.
 * Creates new net from protobuffer parameters in file.
 * @param path
 * @return \c true on success, \c false otherwise
 */
bool prepareNetFile_(char* path);

/**
 * Calls \c CLnets.loadNet with correct type.
 * Loads previously learned net from caffemodel file. The caffemodel must agree
 * with prepared net. This function should be called only after some net was
 * alredy initialized (prepared).
 * @param path
 */
void loadNet_(char* path);

/**
 * Calls \c CLnets.testNet with correct type.
 * Runs net in test mode.
 */
void testNet_();

/**
 * Parses solver parameters from file or string.
 * Calls \c CLnets.trainNet with correct type.
 * @param param whole solver protobuffer or path to protobuffer file
 * @param paramIsFile whether \c contains path or protobuffer
 * @param trainMode TRAIN_NEW, TRAIN_SNAPSHOT or TRAIN_WEIGHTS
 * @param path path to snapshot or weights if selected in \c trainMode
 * @return \c true on success, \c false otherwise
 */
bool trainNet_(char *param, bool paramIsFile, int trainMode, char* path);

/**
 * Calls \c CLnets.exportNet with correct type.
 * Exports net to given file path.
 * @param path
 * @return \c true on success, \c false otherwise
 */
bool exportNet_(char* path);

/**
 * Calls \c CLnets.printNetInfo with correct type.
 * Prints net and layer parameters: layer names, top, bottom counts...
 */
void printNetInfo_();

/**
 * Calls \c CLnets.getLayerNum with correct type.
 * Returns number of layers.
 * @return 
 */
int getLayerNum_();

/**
 * Returns index of layer with given name.
 * @param name layer name
 * @return 
 */
int getLayerIdx_(char* name);

/**
 * Returns array of size num. layers + 1 with starting indices of parameter
 * data of each layer. Last cell contains size of parameter data array. 
 * @return 
 */
int* getParamDataLUT();

/** 
 * Calculates blob size using given function and stores it to given pointer. 
 * Returns false on failure
 * or if size == 0.
 * 
 * @param blobSize num * ch * w * h
 * @param getBlSize f. p. relevant to blob type
 * @param layerIdx layer index
 * @param blobIdx blob index
 * @return \c true on success, \c false otherwise
 */
bool getBlobSize_(int *blobSize,
        bool (*getBlSize)(int**, int*, int), int layerIdx, int blobIdx);

/**
 * Stores dimensions of all top blobs in layer \c layerIdx into pointer \c dims.
 * @param dims array of blob dimensons
 * @param dimSize length of \c dims
 * @param layerIdx layer index
 * @return \c true on success, \c false otherwise
 */
bool getTopBlobSize_(int** dims, int* dimSize, int layerIdx);

/**
 * Stores dimensions of all bottom blobs in layer \c layerIdx into pointer \c dims.
 * @param dims array of blob dimensons
 * @param dimSize length of \c dims
 * @param layerIdx layer index
 * @return \c true on success, \c false otherwise
 */
bool getBottomBlobSize_(int** dims, int* dimSize, int layerIdx);

/**
 * Stores dimensions of all param blobs in layer \c layerIdx into pointer \c dims.
 * @param dims array of blob dimensons
 * @param dimSize length of \c dims
 * @param layerIdx layer index
 * @return \c true on success, \c false otherwise
 */
bool getParamBlobSize_(int** dims, int* dimSize, int layerIdx);

/**
 * Stores dimensions of input blobs into pointer \c dims.
 * @param dims array of blob dimensons
 * @param dimSize length of \c dims
 * @param layerIdx not used
 * @return \c true on success, \c false otherwise
 */
bool getInputSize_(int** dims, int* dimSize, int layerIdx);

/**
 * Stores pointer to cpu data into \c data of top blob \c blobIdx in
 * layer \c layerIdx.
 * @param data
 * @param layerIdx layer index
 * @param blobIdx blob index
 * @return \c true on success, \c false otherwise 
 */
bool getTopBlob_(double** data, int layerIdx, int blobIdx);

/**
 * Stores pointer to cpu data into \c data of bottom blob \c blobIdx in
 * layer \c layerIdx.
 * @param data
 * @param layerIdx layer index
 * @param blobIdx blob index
 * @return \c true on success, \c false otherwise 
 */
bool getBottomBlob_(double** data, int layerIdx, int blobIdx);

/**
 * Stores pointer to cpu data into \c data of param blob \c blobIdx in
 * layer \c layerIdx.
 * @param data
 * @param layerIdx layer index
 * @param blobIdx blob index
 * @return \c true on success, \c false otherwise 
 */
bool getParamBlob_(double** data, int layerIdx, int blobIdx);

/**
 * Sets \c data as cpu data in top blob \c blobIdx in layer \c layerIdx.
 * @param data
 * @param layerIdx layer index
 * @param blobIdx blob index
 * @return \c true on success, \c false otherwise  
 */
bool setTopBlob_(double** data, int layerIdx, int blobIdx);

/**
 * Sets \c data as cpu data in bottom blob \c blobIdx in layer \c layerIdx.
 * @param data
 * @param layerIdx layer index
 * @param blobIdx blob index
 * @return \c true on success, \c false otherwise  
 */
bool setBottomBlob_(double** data, int layerIdx, int blobIdx);

/**
 * Sets \c data as cpu data in param blob \c blobIdx in layer \c layerIdx.
 * @param data
 * @param layerIdx layer index
 * @param blobIdx blob index
 * @return \c true on success, \c false otherwise  
 */
bool setParamBlob_(double** data, int layerIdx, int blobIdx);

/**
 *  * Calls \c CLnets.setInput with correct type.
 * Inserts input for net - whole batch of images. InputData is float or double
 * 1D array with size of B * K * H * W where B is batch size (num. of images), K
 * is num of channels in each image and H and W are dimensions of all images.
 * Data layout: {img01, img02, ...}, img01={ch01, ch02, ...},
 * ch01={row1, row2, ...}, row1 = {col1, col2, ...}
 * @param inputData
 * @return \c true on success, \c false otherwise
 */
bool setInput_(double **inputData);

}

#endif	/* CAFFELINK_HPP */

