
#include <cstdlib>
#include <cstdio>

#include "WolframLibrary.h"
#include "caffe/caffe.hpp"

#include "build_utils.hpp"
#include "caffeLink.hpp"

/** Mathematica librarylink wrapper for \c exportNet_.*/
extern "C" DLLEXPORT int exportNet(LIB_LINK_ARGS)
{
    char* path;   
    
    if (Argc != 1) {
        printf("ERR: %s takes 1 argument: UTF8String path to net data\n",
                __FUNCTION__);
        libData->Message(MSG_WRONG_ARGS);
        return LIBRARY_FUNCTION_ERROR;
    }
    
    path = MArgument_getUTF8String(Args[0]);    
    exportNet_(path);

    return LIBRARY_NO_ERROR;
}

/** Prints root directory in stdout. */
extern "C" DLLEXPORT int printWorkingPath(LIB_LINK_ARGS)
{
    if (Argc != 0) {
        printf("WRN: %s takes 0 arguments\n", __FUNCTION__);
    }
    
    system("pwd");
    return LIBRARY_NO_ERROR;
}

/** Mathematica librarylink wrapper for \c printNetInfo_.*/
extern "C" DLLEXPORT int printNetInfo(LIB_LINK_ARGS)
{
    if (Argc != 0) {
        printf("WRN: %s takes 0 arguments\n", __FUNCTION__);
    }
    
    printNetInfo_();
    return LIBRARY_NO_ERROR;
}

/** Mathematica librarylink wrapper for \c getLayerNum_.*/
extern "C" DLLEXPORT int getLayerNum(LIB_LINK_ARGS)
{
    if (Argc != 0) {
        printf("WRN: %s takes 0 arguments\n", __FUNCTION__);
    }
    
    MArgument_setInteger(Res, getLayerNum_());
    return LIBRARY_NO_ERROR;
}

bool getBlobSize(WolframLibraryData libData,
        MTensor *dimsMT, int layerIdx, bool (*getBlSize)(int**, int*, int))
{
    int err, i;
    int *dims, dimSize;
    mint *data, mtSize;
    
    if(!getBlSize(&dims, &dimSize, layerIdx))
        return false;

    mtSize = dimSize;
    err = libData->MTensor_new(MType_Integer, 1, &mtSize, dimsMT);
    if (err) {
        printf("ERR %d, could not create new MTensor\n", err);
        return false;
    }
    
    data = libData->MTensor_getIntegerData(*dimsMT);
    for(i = 0; i < dimSize; i++)
        data[i] = dims[i];
    
    if(dimSize != 0)
        free(dims);

    return true;
}

/** Mathematica librarylink wrapper for \c getTopBlobSize_.*/
extern "C" DLLEXPORT int getTopBlobSize(LIB_LINK_ARGS)
{
    MTensor dimsMT;
    int layerIdx;
    
    if(Argc != 1){
        printf("ERR: %s takes 1 argument: Integer layer index\n",
                __FUNCTION__);
        libData->Message(MSG_WRONG_ARGS);
        return LIBRARY_FUNCTION_ERROR;
    }

    layerIdx = MArgument_getInteger(Args[0]);
    if (!getBlobSize(libData, &dimsMT, layerIdx, &getTopBlobSize_))
        return LIBRARY_FUNCTION_ERROR;
    
    MArgument_setMTensor(Res, dimsMT);
    return LIBRARY_NO_ERROR;   
}

/** Mathematica librarylink wrapper for \c getBottomBlobSize_.*/
extern "C" DLLEXPORT int getBottomBlobSize(LIB_LINK_ARGS)
{
    MTensor dimsMT;
    int layerIdx;
    
    if(Argc != 1){
        printf("ERR: %s takes 1 argument: Integer layer index\n",
                __FUNCTION__);
        libData->Message(MSG_WRONG_ARGS);
        return LIBRARY_FUNCTION_ERROR;
    }

    layerIdx = MArgument_getInteger(Args[0]);
    if (!getBlobSize(libData, &dimsMT, layerIdx, &getBottomBlobSize_))
        return LIBRARY_FUNCTION_ERROR;
    
    MArgument_setMTensor(Res, dimsMT);
    return LIBRARY_NO_ERROR;   
}

/** Mathematica librarylink wrapper for \c getParamBlobSize_.*/
extern "C" DLLEXPORT int getParamBlobSize(LIB_LINK_ARGS)
{
    MTensor dimsMT;
    int layerIdx;
    
    if(Argc != 1){
        printf("ERR: %s takes 1 argument: Integer layer index\n",
                __FUNCTION__);
        libData->Message(MSG_WRONG_ARGS);
        return LIBRARY_FUNCTION_ERROR;
    }

    layerIdx = MArgument_getInteger(Args[0]);
    if (!getBlobSize(libData, &dimsMT, layerIdx, &getParamBlobSize_))
        return LIBRARY_FUNCTION_ERROR;
    
    MArgument_setMTensor(Res, dimsMT);
    return LIBRARY_NO_ERROR;   
}

bool getBlob(WolframLibraryData libData, MTensor *blobMT,
        int layerIdx, int blobIdx,
        bool (*getBl)(double**, int, int), bool (*getBlSize)(int**, int*, int))
{
    int err, i, blobSize;
    int *dims, dimSize;
    mint mtSize;
    double *data;
    double *dataD;
    float *dataF;

    if (isUsingDouble()) {
        if (!getBl(&dataD, layerIdx, blobIdx))
            return false;        
    } else {
        if (!getBl((double**) &dataF, layerIdx, blobIdx))
            return false;
    }
    
    if(!getBlobSize_(&blobSize, getBlSize, layerIdx, blobIdx))
        return false;
    
    mtSize = blobSize;
    err = libData->MTensor_new(MType_Real, 1, &mtSize, blobMT);
    if (err){
        printf("ERR %d, could not create new MTensor\n", err);
        return false;
    }
    
    data = libData->MTensor_getRealData(*blobMT);
    
    if (isUsingDouble()) {
        for(i = 0; i < mtSize; i++)
            data[i] = dataD[i];
    } else {
        for(i = 0; i < mtSize; i++)
            data[i] = dataF[i];
    }    
       
    return true;
}

/** Mathematica librarylink wrapper for \c getTopBlob_.*/
extern "C" DLLEXPORT int getTopBlob(LIB_LINK_ARGS)
{
    MTensor blobMT;
    int layerIdx;
    int blobIdx;
    
    if(Argc != 2){
        printf("ERR: %s takes 2 arguments: Integer layer index "
                "and Integer blob index\n",
                __FUNCTION__);
        libData->Message(MSG_WRONG_ARGS);
        return LIBRARY_FUNCTION_ERROR;
    }

    layerIdx = MArgument_getInteger(Args[0]);
    blobIdx = MArgument_getInteger(Args[1]);
    if (!getBlob(libData, &blobMT, layerIdx, blobIdx, &getTopBlob_, &getTopBlobSize_))
        return LIBRARY_FUNCTION_ERROR;
    
    MArgument_setMTensor(Res, blobMT);
    return LIBRARY_NO_ERROR;   
}

/** Mathematica librarylink wrapper for \c getBottomBlob_.*/
extern "C" DLLEXPORT int getBottomBlob(LIB_LINK_ARGS)
{
    MTensor blobMT;
    int layerIdx;
    int blobIdx;
    
    if(Argc != 2){
        printf("ERR: %s takes 2 arguments: Integer layer index "
                "and Integer blob index\n",
                __FUNCTION__);
        libData->Message(MSG_WRONG_ARGS);
        return LIBRARY_FUNCTION_ERROR;
    }

    layerIdx = MArgument_getInteger(Args[0]);
    blobIdx = MArgument_getInteger(Args[1]);
    if (!getBlob(libData, &blobMT, layerIdx, blobIdx, &getBottomBlob_, &getBottomBlobSize_))
        return LIBRARY_FUNCTION_ERROR;
    
    MArgument_setMTensor(Res, blobMT);
    return LIBRARY_NO_ERROR;   
}

/** Mathematica librarylink wrapper for \c getParamBlob_.*/
extern "C" DLLEXPORT int getParamBlob(LIB_LINK_ARGS)
{
    MTensor blobMT;
    int layerIdx;
    int blobIdx;
    
    if(Argc != 2){
        printf("ERR: %s takes 2 arguments: Integer layer index "
                "and Integer blob index\n",
                __FUNCTION__);
        libData->Message(MSG_WRONG_ARGS);
        return LIBRARY_FUNCTION_ERROR;
    }

    layerIdx = MArgument_getInteger(Args[0]);
    blobIdx = MArgument_getInteger(Args[1]);
    if (!getBlob(libData, &blobMT, layerIdx, blobIdx, &getParamBlob_, &getParamBlobSize_))
        return LIBRARY_FUNCTION_ERROR;
    
    MArgument_setMTensor(Res, blobMT);
    return LIBRARY_NO_ERROR;   
}
