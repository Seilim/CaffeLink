
#include <cstdlib>
#include <vector>
#include <cstring>
#include <google/protobuf/text_format.h>

#include "WolframLibrary.h"
#include "caffe/caffe.hpp"

#include "caffeLink.hpp"
#include "build_utils.hpp"
#include "utils.hpp"
#include "CLnets.hpp"


static bool useDoubles;
static bool inited = false;

static CLnets<double> netsD;
static CLnets<float> netsF;

static float *inputDataF;
static float **paramDataF;
static int *pd;

extern "C" bool initParamDataF();
extern "C" void freeParamDataF();

extern "C" void initCaffeLink_(bool useDoublesPar, bool useGPU, int devID)
{
    if (!inited) {
        useDoubles = useDoublesPar;
        inited = true;
        printf("CaffeLink initiation:\n");
    } else
        printf("CaffeLink already initiated:\n");

    if (useDoubles)
        printf(" Net data type: double\n");
    else
        printf(" Net data type: float\n");
    
    if(useGPU){
        caffe::Caffe::set_mode(caffe::Caffe::GPU);
        caffe::Caffe::SetDevice(devID);
        printf(" Mode: GPU, dev ID: %d\n", devID);
    }
    else{
        printf(" Mode: CPU\n");
        caffe::Caffe::set_mode(caffe::Caffe::CPU);
    }
    
    caffe::Caffe::set_phase(caffe::Caffe::TEST);
}

extern "C" bool isUsingDouble()
{
    return useDoubles;
}

extern "C" bool prepareNetString_(char* paramStr)
{
    if (useDoubles) {
        if (netsD.prepareNetString(paramStr))
            return netsD.initParamDataLUT(&pd);
    } else {
        freeParamDataF();
        if (netsF.prepareNetString(paramStr) && netsF.initParamDataLUT(&pd))
            return initParamDataF();
    }

    return false;
}

extern "C" bool prepareNetFile_(char* path)
{
    if (useDoubles) {
        if (netsD.prepareNetFile(path))
            return netsD.initParamDataLUT(&pd);

    } else {
        freeParamDataF();
        if (netsF.prepareNetFile(path) && netsF.initParamDataLUT(&pd))
            return initParamDataF();
    }

    return false;
}

extern "C" bool initParamDataF()
{   
    int i;
    paramDataF = (float**) malloc(sizeof(float*) * pd[getLayerNum_()]);
    if (!paramDataF) {
        printf("%s: allocation failed\n", __FUNCTION__);
        return false;
    }

    for (i = 0; i < pd[getLayerNum_()]; i++)
        paramDataF[i] = NULL;
    
    return true;
}

extern "C" void freeParamDataF()
{
    int i;
    if(!pd || !paramDataF)
        return;
    
    for(i = 0; i < pd[getLayerNum_()]; i++)
        if(paramDataF[i])
            free(paramDataF[i]);
       
    free(paramDataF);
}

extern "C" void loadNet_(char* path)
{ 
    printf("net data src: %s\n", path);
    if(useDoubles)
        netsD.loadNet(path);
    else
        netsF.loadNet(path);
}

extern "C" void testNet_()
{
    caffe::Caffe::set_phase(caffe::Caffe::TEST);
    
    if(useDoubles)
        netsD.testNet();
    else
        netsF.testNet();    
}

extern "C"
bool trainNet_(char *param, bool paramIsFile, int trainMode, char* path)
{
    bool success;
    caffe::SolverParameter solverParam; 
    
    if (paramIsFile) {
        success = caffe::ReadProtoFromTextFile(param, &solverParam);
    } else {
        success = google::protobuf::TextFormat::ParseFromString(param, &solverParam);
        if (!success) {
            printf("ERR in %s: Could not create solver params\n", __FUNCTION__);
            return false;
        }
    }
    
    if(useDoubles)
        netsD.trainNet(&solverParam, trainMode, path);
    else
        netsF.trainNet(&solverParam, trainMode, path);
    
    return true;
}

extern "C" bool exportNet_(char* path)
{
    if(useDoubles)
        return netsD.exportNet(path);
    else
        return netsF.exportNet(path);
}

extern "C" void printNetInfo_()
{
    /*
     * Top and bottom blobs can be accesed via layer index throught net in
     *     net.top_vecs()[laIdx].
     * Learned parameters can be acceses from layers or directly from net:
     *     net.layers[laIdx].get().blobs()
     *     net.params()
     * but vector in net.params() is a list of all parameter blobs so indexing
     * is not simple since some layers has more than one par. bl. (filters
     * and bias...) and some has none.
     * 
     * Blob is caffe's ultimate data storage. Data are interpeted as 4D array
     * stored in 1D:
     * [image, channel, row, collumn].
     * Which would be in 1D array as:
     * blob.data = {img1, img2, ...},
     *   img1 = {A,B,...},
     *   channel A = {row1, row2, ...},
     *   row1 = {col1, col2, ...}
     */

    if (useDoubles)
        netsD.printNetInfo();
    else
        netsF.printNetInfo();
}

extern "C" int getLayerNum_()
{
    if (useDoubles)
        return netsD.getLayerNum();
    else
        return netsF.getLayerNum();
}

extern "C" int getLayerIdx_(char* name)
{
    if (useDoubles)
        return netsD.getLayerIdx(name);
    else
        return netsF.getLayerIdx(name);
}

extern "C" int* getParamDataLUT()
{
    return pd;
}

extern "C" bool getBlobSize_(int *blobSize, bool (*getBlSize)(int**, int*, int),
        int layerIdx, int blobIdx)
{
    int i;
    int *dims, dimSize;
    if(!getBlSize(&dims, &dimSize, layerIdx) || dimSize == 0)
        return false;
    
    *blobSize = 1;
    for(i = blobIdx * 4; i < (blobIdx + 1) * 4; i++)
        *blobSize *= dims[i];
    
    free(dims); 
    return true;
}

extern "C" bool getTopBlobSize_(int** dims, int* dimSize, int layerIdx)
{
    if (useDoubles)
        return netsD.getTopBlobSize(dims, dimSize, layerIdx);
    else
        return netsF.getTopBlobSize(dims, dimSize, layerIdx);
}

extern "C" bool getBottomBlobSize_(int** dims, int* dimSize, int layerIdx)
{
    if (useDoubles)
        return netsD.getBottomBlobSize(dims, dimSize, layerIdx);
    else
        return netsF.getBottomBlobSize(dims, dimSize, layerIdx);
}

extern "C" bool getParamBlobSize_(int** dims, int* dimSize, int layerIdx)
{
    if (useDoubles)
        return netsD.getParamBlobSize(dims, dimSize, layerIdx);
    else
        return netsF.getParamBlobSize(dims, dimSize, layerIdx);
}

extern "C" bool getInputSize_(int** dims, int* dimSize, int layerIdx)
{
    if (useDoubles)
        return netsD.getInputSize(dims, dimSize);
    else
        return netsF.getInputSize(dims, dimSize);
}

extern "C" bool getTopBlob_(double** data, int layerIdx, int blobIdx)
{
    if (useDoubles)
        return netsD.getTopBlob(data, layerIdx, blobIdx);
    else
        return netsF.getTopBlob((float**) data, layerIdx, blobIdx);
}

extern "C" bool getBottomBlob_(double** data, int layerIdx, int blobIdx)
{
    if (useDoubles)
        return netsD.getBottomBlob(data, layerIdx, blobIdx);
    else
        return netsF.getBottomBlob((float**) data, layerIdx, blobIdx);
}

extern "C" bool getParamBlob_(double** data, int layerIdx, int blobIdx)
{
    if (useDoubles)
        return netsD.getParamBlob(data, layerIdx, blobIdx);
    else
        return netsF.getParamBlob((float**) data, layerIdx, blobIdx);
}

extern "C" bool setTopBlob_(double** data, int layerIdx, int blobIdx)
{
    if (useDoubles)
        return netsD.setTopBlob(data, layerIdx, blobIdx);
    else {
        float* tmp;
        int blobSize;
        if (!getBlobSize_(&blobSize, *getTopBlobSize_, layerIdx, blobIdx))
            return false;
        if (!doublesToFloats(*data, &tmp, blobSize)) {
            return false;
        }
        return netsF.setTopBlob(&tmp, layerIdx, blobIdx);
    }
}

extern "C" bool setBottomBlob_(double** data, int layerIdx, int blobIdx)
{
    if (useDoubles)
        return netsD.setBottomBlob(data, layerIdx, blobIdx);
    else {
        float* tmp;
        int blobSize;
        if (!getBlobSize_(&blobSize, *getBottomBlobSize_, layerIdx, blobIdx))
            return false;
        if (!doublesToFloats(*data, &tmp, blobSize)) {
            return false;
        }
        return netsF.setBottomBlob(&tmp, layerIdx, blobIdx);
    }
}

extern "C" bool setParamBlob_(double** data, int layerIdx, int blobIdx)
{
    if (useDoubles)
        return netsD.setParamBlob(data, layerIdx, blobIdx);
    else {
        int blobSize;        
        int paramDataIdx = getParamDataLUT()[layerIdx] + blobIdx;
        
        if (!getBlobSize_(&blobSize, *getParamBlobSize_, layerIdx, blobIdx))
            return false;
        if (!doublesToFloats(*data, &paramDataF[paramDataIdx], blobSize)) {
            return false;
        }
        return netsF.setParamBlob(&paramDataF[paramDataIdx], layerIdx, blobIdx);
    }
}

extern "C" bool setInput_(double **inputData)
{
    if(useDoubles)
        return netsD.setInput(inputData);
    else{
        int blobSize;        
        if (!getBlobSize_(&blobSize, *getInputSize_, 0, 0))
            return false;
        if (!doublesToFloats(*inputData, &inputDataF, blobSize)) {
            return false;
        }
        return netsF.setInput(&inputDataF);
    }
}