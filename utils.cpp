
#include <vector>
#include <opencv/cv.h>
#include <opencv/highgui.h>
extern "C" {
#include <cblas.h>
}

#include "caffe/caffe.hpp"
#include "WolframLibrary.h"

#include "build_utils.hpp"

extern "C" bool doublesToFloats(double* in, float** out, long size)
{
    int i;
    float *tmp;
    tmp = (float*) realloc(*out, sizeof(float) * size);

    if (!tmp) {
        printf("%s: allocation failed\n", __FUNCTION__);
        return false;
    }
    *out = tmp;

    for (i = 0; i < size; i++)
        (*out)[i] = in[i];
    return true;
}

/** Simple function to test library link. Gets one tensor on input and returns
 * second with squared elements of the first.
 * In Mathematica: tll = LibraryLoadFunction["libcaffeLink", "testLibLink",
 * {{Real, 2}}, {Real, 2}]
 * tll[{{3, 4, 5}, {1, 2, 6}}] --> {{9., 16., 25.}, {1., 4., 36.}} */
extern "C" DLLEXPORT int testLibLink(LIB_LINK_ARGS)
{
    int err; // error code
    MTensor m1; // input tensor
    MTensor m2; // output tensor

    mint const* dims; // dimensions of the tensor

    double* data1; // actual data of the input tensor
    double* data2; // data for the output tensor

    mint i; // bean counters
    mint j;

    m1 = MArgument_getMTensor(Args[0]);
    dims = libData->MTensor_getDimensions(m1);
    err = libData->MTensor_new(MType_Real, 2, dims, &m2);
    if(err)
        return LIBRARY_MEMORY_ERROR;
    data1 = libData->MTensor_getRealData(m1);
    data2 = libData->MTensor_getRealData(m2);

    for (i = 0; i < dims[0]; i++) {
        for (j = 0; j < dims[1]; j++) {
            data2[i * dims[1] + j] = data1[i * dims[1] + j] * data1[i * dims[1] + j];
        }
    }

    MArgument_setMTensor(Res, m2);
    return LIBRARY_NO_ERROR;
}

extern "C" void cblasTest_()
{    
#ifdef CBLAS_TEST
    
    // check Atlas version, only if you build this with atlas (-latlas)
    //    void ATL_buildinfo(void);
    //    ATL_buildinfo();

    printf("Testing cblast_sgemm...\n");

    const int M = 10;
    const int N = 8;
    const int K = 5;

    float *A = new float[M * K];
    float *B = new float[K * N];
    float *C = new float[M * N];

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            M, N, K, 1.0, A, K, B, N, 0.0, C, N);

    delete A;
    delete B;
    delete C;

    printf("Success, cblast_sgemm is ok.\n");
#else
    printf("To test cblast_sgemm, rebuild caffeLink with CBLAS_TEST defined.\n");
#endif
}

/** In case your kernel crushes in CPU mode, you can try calling this
 * function cblas_test and if it crushes again, then it is the mentioned
 * cblas_sggem. */
extern "C" DLLEXPORT int cblasTest(LIB_LINK_ARGS)
{
    cblasTest_();
    return LIBRARY_NO_ERROR;
}


