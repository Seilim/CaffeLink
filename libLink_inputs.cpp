
#include <cstdlib>
#include <cstdio>

#include "WolframLibrary.h"
#include "caffe/caffe.hpp"

#include "build_utils.hpp"
#include "caffeLink.hpp"
#include "libLink_inputs.hpp"

static MTensor inputBlobMT;
static MTensor *paramDataMT;

extern "C" bool initParamDataMT()
{
    int i;
    int *pd;
    pd = getParamDataLUT();

    paramDataMT = (MTensor*) malloc(sizeof(MTensor) * pd[getLayerNum_()]);
    if (!paramDataMT) {
        printf("%s: allocation failed\n", __FUNCTION__);
        return false;
    }
    
    for (i = 0; i < pd[getLayerNum_()]; i++)
        paramDataMT[i] = NULL;
    
    return true;
}

extern "C" void freeParamDataMT(WolframLibraryData libData)
{
    int i;
    int *pd;
    pd = getParamDataLUT();
    if(!pd || !paramDataMT)
        return;

    for(i = 0; i < pd[getLayerNum_()]; i++)
        if(paramDataMT[i])
            libData->MTensor_free(paramDataMT[i]);
    
    free(paramDataMT);
}

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
    int parDataIdx;
    
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
    if(!setParamBlob_(&data, layerIdx, blobIdx)){
        libData->MTensor_free(blobMT);
        return LIBRARY_FUNCTION_ERROR;
    }
    
    if (!isUsingDouble())
        /* MTensor (doubles) was converted to float, hence is not needed */
        libData->MTensor_free(blobMT);
    else {
        /* free previous MTensor and store the new one */
        parDataIdx = getParamDataLUT()[layerIdx] + blobIdx;
        if (paramDataMT[parDataIdx])
            libData->MTensor_free(paramDataMT[parDataIdx]);
        paramDataMT[parDataIdx] = blobMT;
    }

    return LIBRARY_NO_ERROR;
}

/** Mathematica librarylink wrapper for \c setTopBlob_ called by layer name.*/
extern "C" DLLEXPORT int setTopBlobLName(LIB_LINK_ARGS)
{
    MTensor blobMT;
    char* name;
    int layerIdx;
    int blobIdx;
    double *data;
    
    if(Argc != 3){
        printf("ERR: %s takes 3 argument: Real tensor data, UTF8String layer"
                "name and Integer blob index\n",
                __FUNCTION__);
        libData->Message(MSG_WRONG_ARGS);
        return LIBRARY_FUNCTION_ERROR;
    }

    blobMT = MArgument_getMTensor(Args[0]);
    name = MArgument_getUTF8String(Args[1]);
    layerIdx = getLayerIdx_(name);
    if(layerIdx == -1)
        return LIBRARY_FUNCTION_ERROR;
    
    blobIdx = MArgument_getInteger(Args[2]);
    
    data = libData->MTensor_getRealData(blobMT);
    if(!setTopBlob_(&data, layerIdx, blobIdx))
        return LIBRARY_FUNCTION_ERROR;

    return LIBRARY_NO_ERROR;
}

/** Mathematica librarylink wrapper for \c setBottomBlob_ called by layer name.*/
extern "C" DLLEXPORT int setBottomBlobLName(LIB_LINK_ARGS)
{
    MTensor blobMT;
    char* name;
    int layerIdx;
    int blobIdx;
    double *data;
    
    if(Argc != 3){
        printf("ERR: %s takes 3 argument: Real tensor data, UTF8String layer"
                "name and Integer blob index\n",
                __FUNCTION__);
        libData->Message(MSG_WRONG_ARGS);
        return LIBRARY_FUNCTION_ERROR;
    }

    blobMT = MArgument_getMTensor(Args[0]);
    name = MArgument_getUTF8String(Args[1]);
    layerIdx = getLayerIdx_(name);
    if(layerIdx == -1)
        return LIBRARY_FUNCTION_ERROR;
    
    blobIdx = MArgument_getInteger(Args[2]);
    
    data = libData->MTensor_getRealData(blobMT);
    if(!setBottomBlob_(&data, layerIdx, blobIdx))
        return LIBRARY_FUNCTION_ERROR;

    return LIBRARY_NO_ERROR;
}

/** Mathematica librarylink wrapper for \c setParamBlob_ called by layer name.*/
extern "C" DLLEXPORT int setParamBlobLName(LIB_LINK_ARGS)
{
    MTensor blobMT;
    char* name;
    int layerIdx;
    int blobIdx;
    double *data;
    int parDataIdx;
    
    if(Argc != 3){
        printf("ERR: %s takes 3 argument: Real tensor data, UTF8String layer"
                "name and Integer blob index\n",
                __FUNCTION__);
        libData->Message(MSG_WRONG_ARGS);
        return LIBRARY_FUNCTION_ERROR;
    }

    blobMT = MArgument_getMTensor(Args[0]);
    name = MArgument_getUTF8String(Args[1]);
    layerIdx = getLayerIdx_(name);
    if(layerIdx == -1){
        libData->MTensor_free(blobMT);
        return LIBRARY_FUNCTION_ERROR;
    }
    
    blobIdx = MArgument_getInteger(Args[2]);

    data = libData->MTensor_getRealData(blobMT);
    if (!setParamBlob_(&data, layerIdx, blobIdx)) {
        libData->MTensor_free(blobMT);
        return LIBRARY_FUNCTION_ERROR;
    }
    
    if (!isUsingDouble())
        /* MTensor (doubles) was converted to float, hence is not needed */
        libData->MTensor_free(blobMT);
    else {
        /* free previous MTensor and store the new one */
        parDataIdx = getParamDataLUT()[layerIdx] + blobIdx;        
        if (paramDataMT[parDataIdx])           
            libData->MTensor_free(paramDataMT[parDataIdx]);
        paramDataMT[parDataIdx] = blobMT;
    }

    return LIBRARY_NO_ERROR;
}

/** Mathematica librarylink wrapper for \c setInput_.*/
extern "C" DLLEXPORT int setInput(LIB_LINK_ARGS)
{
    double *data;
    
    if(Argc != 1){
        printf("ERR: %s takes 1 argument: Real tensor data, with size"
                "of input blob\n",
                __FUNCTION__);
        libData->Message(MSG_WRONG_ARGS);
        return LIBRARY_FUNCTION_ERROR;
    }

    if(inputBlobMT)
        libData->MTensor_free(inputBlobMT);
    inputBlobMT = MArgument_getMTensor(Args[0]);
    
    data = libData->MTensor_getRealData(inputBlobMT);
    if(!setInput_(&data))
        return LIBRARY_FUNCTION_ERROR;  
    
    return LIBRARY_NO_ERROR;
}
