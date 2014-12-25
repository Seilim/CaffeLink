
#include <cstdlib>
#include <cstdio>

#include "WolframLibrary.h"
#include "caffe/caffe.hpp"

#include "build_utils.hpp"
#include "caffeLink.hpp"


/** Mathematica librarylink wrapper for \c setTopBlob_.*/
extern "C" DLLEXPORT int setTopBlob(LIB_LINK_ARGS)
{
    MTensor blobMT;
    int layerIdx;
    int blobIdx;
    double *data;
    
    if(Argc != 3){
        printf("ERR: %s takes 3 argument: Real tensor data, Integer layer index"
                "and Integer blob index\n",
                __FUNCTION__);
        libData->Message(MSG_WRONG_ARGS);
        return LIBRARY_FUNCTION_ERROR;
    }

    blobMT = MArgument_getMTensor(Args[0]);
    layerIdx = MArgument_getInteger(Args[1]);
    blobIdx = MArgument_getInteger(Args[2]);
    
    data = libData->MTensor_getRealData(blobMT);
    if(!setTopBlob_(&data, layerIdx, blobIdx))
        return LIBRARY_FUNCTION_ERROR;

    return LIBRARY_NO_ERROR;
}

/** Mathematica librarylink wrapper for \c setBottomBlob_.*/
extern "C" DLLEXPORT int setBottomBlob(LIB_LINK_ARGS)
{
    MTensor blobMT;
    int layerIdx;
    int blobIdx;
    double *data;
    
    if(Argc != 3){
        printf("ERR: %s takes 3 argument: Real tensor data, Integer layer index"
                "and Integer blob index\n",
                __FUNCTION__);
        libData->Message(MSG_WRONG_ARGS);
        return LIBRARY_FUNCTION_ERROR;
    }

    blobMT = MArgument_getMTensor(Args[0]);
    layerIdx = MArgument_getInteger(Args[1]);
    blobIdx = MArgument_getInteger(Args[2]);
    
    data = libData->MTensor_getRealData(blobMT);
    if(!setBottomBlob_(&data, layerIdx, blobIdx))
        return LIBRARY_FUNCTION_ERROR;

    return LIBRARY_NO_ERROR;
}

/** Mathematica librarylink wrapper for \c setParamBlob_.*/
extern "C" DLLEXPORT int setParamBlob(LIB_LINK_ARGS)
{
    MTensor blobMT;
    int layerIdx;
    int blobIdx;
    double *data;
    
    if(Argc != 3){
        printf("ERR: %s takes 3 argument: Real tensor data, Integer layer index"
                "and Integer blob index\n",
                __FUNCTION__);
        libData->Message(MSG_WRONG_ARGS);
        return LIBRARY_FUNCTION_ERROR;
    }

    blobMT = MArgument_getMTensor(Args[0]);
    layerIdx = MArgument_getInteger(Args[1]);
    blobIdx = MArgument_getInteger(Args[2]);
    
    data = libData->MTensor_getRealData(blobMT);
    if(!setParamBlob_(&data, layerIdx, blobIdx))
        return LIBRARY_FUNCTION_ERROR;

    return LIBRARY_NO_ERROR;
}

/** Mathematica librarylink wrapper for \c setInput_.*/
extern "C" DLLEXPORT int setInput(LIB_LINK_ARGS)
{
    MTensor blobMT;
    double *data;
    
    if(Argc != 1){
        printf("ERR: %s takes 1 argument: Real tensor data, with size"
                "of input blob\n",
                __FUNCTION__);
        libData->Message(MSG_WRONG_ARGS);
        return LIBRARY_FUNCTION_ERROR;
    }

    blobMT = MArgument_getMTensor(Args[0]);
    
    data = libData->MTensor_getRealData(blobMT);
    if(!setInput_(&data))
        return LIBRARY_FUNCTION_ERROR;  
    
    return LIBRARY_NO_ERROR;
}
