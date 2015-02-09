
#include <cstdio>

#include "caffe/caffe.hpp"
#include "WolframLibrary.h"

#include "build_utils.hpp"
#include "caffeLink.hpp"
#include "libLink_inputs.hpp"


/* Necessary for Mathematica library link. */
extern "C" DLLEXPORT mint WolframLibrary_getVersion()
{
    return WolframLibraryVersion;
}

/* Necessary for Mathematica library link. */
extern "C" DLLEXPORT int WolframLibrary_initialize(WolframLibraryData libData)
{
    return 0;
}

/* Necessary for Mathematica library link. */
extern "C" DLLEXPORT void WolframLibrary_uninitialize(WolframLibraryData libData)
{
}

/** Mathematica librarylink wrapper for \c initCaffeLink_.*/
extern "C" DLLEXPORT int initCaffeLink(LIB_LINK_ARGS)
{
    mbool useDoubles;
    mbool useGPU;
    mint devID;
    
    if(Argc != 3){
        printf("ERR: %s takes 3 arguments: \"Boolean\" (True for net using "
                "doubles, false for floats), \"Boolean\" (True for GPU, false "
                "for CPU mode) and Integer (device ID in case of GPU mode)\n",
                __FUNCTION__);
        libData->Message(MSG_WRONG_ARGS);
        return LIBRARY_FUNCTION_ERROR;
    }
    
    useDoubles = MArgument_getBoolean(Args[0]);
    useGPU = MArgument_getBoolean(Args[1]);
    devID = MArgument_getInteger(Args[2]);
    
    initCaffeLink_(useDoubles, useGPU, devID);
    
    return LIBRARY_NO_ERROR;
}

/** Mathematica librarylink wrapper for \c prepareNetFile_.*/
extern "C" DLLEXPORT int prepareNetFile(LIB_LINK_ARGS)
{
    char* path;
    
    if (Argc != 1) {
        printf("ERR: %s takes 1 argument: UTF8String path to net protobuffer file\n",
                __FUNCTION__);
        libData->Message(MSG_WRONG_ARGS);
        return LIBRARY_FUNCTION_ERROR;
    }
    
    path = MArgument_getUTF8String(Args[0]);
    
    if(isUsingDouble())
        freeParamDataMT(libData);
    
    if(!prepareNetFile_(path))
        return LIBRARY_FUNCTION_ERROR;
    
    if(isUsingDouble())
        initParamDataMT();
    
    return LIBRARY_NO_ERROR;
}

/** Mathematica librarylink wrapper for \c prepareNetString_.*/
extern "C" DLLEXPORT int prepareNetString(LIB_LINK_ARGS)
{
    char* str;   
    
    if (Argc != 1) {
        printf("ERR: %s takes 1 argument: UTF8String net protobuffer parameters\n",
                __FUNCTION__);
        libData->Message(MSG_WRONG_ARGS);
        return LIBRARY_FUNCTION_ERROR;
    }
    
    str = MArgument_getUTF8String(Args[0]);

    if (isUsingDouble())
        freeParamDataMT(libData);

    if (!prepareNetString_(str))
        return LIBRARY_FUNCTION_ERROR;

    if (isUsingDouble())
        initParamDataMT();
    
    return LIBRARY_NO_ERROR;
}

/** Mathematica librarylink wrapper for \c loadNet_.*/
extern "C" DLLEXPORT int loadNet(LIB_LINK_ARGS)
{
    char* path; 
    
    if (Argc != 1) {
        printf("ERR: %s takes 1 arguments: UTF8String path to net data\n",
                __FUNCTION__);
        libData->Message(MSG_WRONG_ARGS);
        return LIBRARY_FUNCTION_ERROR;
    }

    path = MArgument_getUTF8String(Args[0]); 
    loadNet_(path);

    return LIBRARY_NO_ERROR;
}

/** Mathematica librarylink wrapper for \c testNet_.*/
extern "C" DLLEXPORT int testNet(LIB_LINK_ARGS)
{
    if (Argc != 0) {
        printf("WRN: %s takes 0 arguments\n", __FUNCTION__);
    }

    testNet_();

    return LIBRARY_NO_ERROR;
}

/** Mathematica librarylink wrapper for \c trainNet_, TRAIN_NEW version.*/
extern "C" DLLEXPORT int trainNetString(LIB_LINK_ARGS)
{
    char* str;

    if (Argc != 1) {
        printf("WRN: %s takes 1 argument: UTF8String solver "
                "protobuffer parameters\n", __FUNCTION__);
        libData->Message(MSG_WRONG_ARGS);
        return LIBRARY_FUNCTION_ERROR;
    }

    str = MArgument_getUTF8String(Args[0]);
    if(!trainNet_(str, false, TRAIN_NEW, (char*) ""))
        return LIBRARY_FUNCTION_ERROR;

    return LIBRARY_NO_ERROR;
}

/** Mathematica librarylink wrapper for \c trainNet_, TRAIN_NEW version.*/
extern "C" DLLEXPORT int trainNetFile(LIB_LINK_ARGS)
{
    char* str;

    if (Argc != 1) {
        printf("WRN: %s takes 1 argument: UTF8String path to solver "
                "protobuffer file\n", __FUNCTION__);
        libData->Message(MSG_WRONG_ARGS);
        return LIBRARY_FUNCTION_ERROR;
    }

    str = MArgument_getUTF8String(Args[0]);
    if(!trainNet_(str, true, TRAIN_NEW, (char*) ""))
        return LIBRARY_FUNCTION_ERROR;

    return LIBRARY_NO_ERROR;
}

/** Mathematica librarylink wrapper for \c trainNet_, TRAIN_SNAPSHOT version.*/
extern "C" DLLEXPORT int trainNetSnapshotString(LIB_LINK_ARGS)
{
    char* str;
    char* path;

    if (Argc != 2) {
        printf("WRN: %s takes 2 argument: UTF8String solver "
                "protobuffer parameters and UTF8String path to snapshot\n",
                __FUNCTION__);
        libData->Message(MSG_WRONG_ARGS);
        return LIBRARY_FUNCTION_ERROR;
    }

    str = MArgument_getUTF8String(Args[0]);
    path = MArgument_getUTF8String(Args[1]);
    if(!trainNet_(str, false, TRAIN_SNAPSHOT, path))
        return LIBRARY_FUNCTION_ERROR;
    
    return LIBRARY_NO_ERROR;
}

/** Mathematica librarylink wrapper for \c trainNet_, TRAIN_SNAPSHOT version.*/
extern "C" DLLEXPORT int trainNetSnapshotFile(LIB_LINK_ARGS)
{
    char* str;
    char* path;

    if (Argc != 2) {
        printf("WRN: %s takes 2 argument: UTF8String path to solver "
                "protobuffer file and UTF8String path to snapshot\n",
                __FUNCTION__);
        libData->Message(MSG_WRONG_ARGS);
        return LIBRARY_FUNCTION_ERROR;
    }

    str = MArgument_getUTF8String(Args[0]);
    path = MArgument_getUTF8String(Args[1]);
    if(!trainNet_(str, true, TRAIN_SNAPSHOT, path))
        return LIBRARY_FUNCTION_ERROR;
    
    return LIBRARY_NO_ERROR;
}

/** Mathematica librarylink wrapper for \c trainNet_, TRAIN_WEIGHTS version.*/
extern "C" DLLEXPORT int trainNetWeightsString(LIB_LINK_ARGS)
{
    char* str;
    char* path;

    if (Argc != 2) {
        printf("WRN: %s takes 2 argument: UTF8String solver "
                "protobuffer parameters and UTF8String path to weights\n",
                __FUNCTION__);
        libData->Message(MSG_WRONG_ARGS);
        return LIBRARY_FUNCTION_ERROR;
    }

    str = MArgument_getUTF8String(Args[0]);
    path = MArgument_getUTF8String(Args[1]);
    if(!trainNet_(str, false, TRAIN_WEIGHTS, path))
        return LIBRARY_FUNCTION_ERROR;
    
    return LIBRARY_NO_ERROR;
}

/** Mathematica librarylink wrapper for \c trainNet_, TRAIN_WEIGHTS version.*/
extern "C" DLLEXPORT int trainNetWeightsFile(LIB_LINK_ARGS)
{
    char* str;
    char* path;

    if (Argc != 2) {
        printf("WRN: %s takes 2 argument: UTF8String to solver "
                "protobuffer file and UTF8String path to weights\n",
                __FUNCTION__);
        libData->Message(MSG_WRONG_ARGS);
        return LIBRARY_FUNCTION_ERROR;
    }

    str = MArgument_getUTF8String(Args[0]);
    path = MArgument_getUTF8String(Args[1]);
    if(!trainNet_(str, true, TRAIN_WEIGHTS, path))
        return LIBRARY_FUNCTION_ERROR;
    
    return LIBRARY_NO_ERROR;
}
