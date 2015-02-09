
#include <vector>
#include <fstream>

#include <google/protobuf/text_format.h>
#include "caffe/caffe.hpp"

#include "build_utils.hpp"
#include "CLnets.hpp"

template <typename Dtype>
CLnets<Dtype>::CLnets()
{
    this->net = NULL;
}

template <typename Dtype>
bool CLnets<Dtype>::prepareNetString(char* paramStr)
{    
    caffe::NetParameter param;    
    bool success = google::protobuf::TextFormat::ParseFromString(paramStr, &param);
    if(!success) {
        printf("ERR in %s: Could not creat net\n", __FUNCTION__);
        return false;
    }
    
    if(this->net)
        delete this->net;
    
    this->net = new caffe::Net<Dtype>(param);
    
    return true;
}

template <typename Dtype>
bool CLnets<Dtype>::prepareNetFile(char* path)
{
    if(this->net)
        delete this->net;
    
    this->net = new caffe::Net<Dtype>(path);

    return true;
}

template <typename Dtype>
bool CLnets<Dtype>::initParamDataLUT(int** pd)
{
    long unsigned int li;
    std::string name;
    caffe::Layer<Dtype>* layer;
    long unsigned int layerNum = net->layers().size();
    
    int paramBlobCnt = 0;
    int *tmp;
    tmp = (int*) realloc(*pd, sizeof(int) * (layerNum + 1));

    if (!tmp) {
        printf("%s: allocation failed\n", __FUNCTION__);
        return false;
    }
    *pd = tmp;

    for (li = 0; li < layerNum; li++) {
        (*pd)[li] = paramBlobCnt;

        name = net->layer_names()[li];
        layer = net->layer_by_name(name).get();
        
        paramBlobCnt += layer->blobs().size();
    }
    /* stores total count of parameter blobs in last cell */
    (*pd)[layerNum] = paramBlobCnt;
    
    return true;
}

template <typename Dtype>
void CLnets<Dtype>::loadNet(char* path)
{
    this->net->CopyTrainedLayersFrom(path);
}

template <typename Dtype>
void CLnets<Dtype>::trainNet(caffe::SolverParameter *param,
        int trainMode, char* path)
{
    // possibly overide mode
    if (param->solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
        caffe::Caffe::set_mode(caffe::Caffe::GPU);
        caffe::Caffe::SetDevice(param->device_id());
        printf("Solver uses GPU with device id %d.\n", param->device_id());
    } else {
        printf("Solver uses CPU.\n");
    }

    printf("Starting Optimization\n");
    caffe::shared_ptr<caffe::Solver<Dtype> >
            solver(caffe::GetSolver<Dtype>(*param));

    switch (trainMode) {
    case TRAIN_NEW:
        solver->Solve();
        break;
    case TRAIN_SNAPSHOT:
        solver->Solve(path);
        break;
    case TRAIN_WEIGHTS:
        solver->net()->CopyTrainedLayersFrom(path);
        solver->Solve();
        break;
    }

    printf("Optimization Done.\n");
}

template <typename Dtype>
bool CLnets<Dtype>::exportNet(char* path)
{
    std::ofstream fCaffeModel(path, std::ios::binary);     
    if(fCaffeModel.fail()){
        printf("ERR in %s: Could not open/write "
                "to a file: %s\n", __FUNCTION__, path);
        return false;
    }    
    
    caffe::NetParameter netPar;
    this->net->ToProto(&netPar, false);    
    netPar.SerializeToOstream(&fCaffeModel);
    
    fCaffeModel.close();
    return true;
}

template <typename Dtype>
void CLnets<Dtype>::testNet()
{
    Dtype loss;
    net->ForwardPrefilled(&loss);
//    printf("loss: %.3f\n", loss);
}

template <typename Dtype>
void CLnets<Dtype>::printNetInfo()
{
    long unsigned int i;
    const std::vector<std::string>& names = net->blob_names();
    
    long unsigned int bi;
    long unsigned int li;
    int parIdx = 0;
    std::string name;
    caffe::Blob<Dtype>* bl;
    caffe::Layer<Dtype>* layer;
    std::vector<caffe::Blob<Dtype>*> laTopVecs, laBotVecs;
    long unsigned int layerNum = net->layers().size();

    printf("Each layer has bottom blobs as input and top blobs as output. Some\n"
            "layers have learneable parameter - weights etc.\n"
            "Names (%lu): ", names.size());
    for (i = 0; i < names.size(); i++)
        printf("%s%s", names[i].c_str(), (i < names.size() - 1) ? " " : "\n");
    
    printf("\n");    
    
    for (li = 0; li < layerNum; li++) {        

        name = net->layer_names()[li];
        layer = net->layer_by_name(name).get();
        laTopVecs = net->top_vecs()[li];
        laBotVecs = net->bottom_vecs()[li];
        /* print layer info*/
        printf("%2lu: %8s, %14s, par.: %2lu, top: %2lu, bot.: %2lu\n",
                li, name.c_str(), layer->type_name().c_str(),
                layer->blobs().size(), laTopVecs.size(), laBotVecs.size());
        /*  and its parameter blobs */
        for (bi = 0; bi < layer->blobs().size(); bi++) {
            bl = layer->blobs()[bi].get();
            printf("%12s par[%lu (%2d)]: (%4d, %4d, %4d, %4d)\n", "",
                    bi, parIdx++,
                    bl->num(), bl->channels(), bl->height(), bl->width());
        }

        /*  and its bottom blobs */
        for (bi = 0; bi < laBotVecs.size(); bi++) {
            bl = laBotVecs[bi];
            printf("%12s bot[%lu,  %2lu]: (%4d, %4d, %4d, %4d)\n", "", bi,
                    li, bl->num(), bl->channels(), bl->height(), bl->width());
        }
        /*  and its top blobs */
        for (bi = 0; bi < laTopVecs.size(); bi++) {
            bl = laTopVecs[bi];
            printf("%12s top[%lu,  %2lu]: (%4d, %4d, %4d, %4d)\n", "", bi,
                    li, bl->num(), bl->channels(), bl->height(), bl->width());
        }
    }    
    
    printf("\ninput blobs info:\n  vector len: %lu\n", net->input_blobs().size());
    for (bi = 0; bi < net->input_blobs().size(); bi++) {
        bl = net->input_blobs()[bi];
            printf("%6s input[%lu]: (%4d, %4d, %4d, %4d)\n", "", bi,
                    bl->num(), bl->channels(), bl->height(), bl->width());
    }

}

template <typename Dtype>
int CLnets<Dtype>::getLayerIdx(char* name)
{
    long unsigned int li;
    long unsigned int layerNum = net->layers().size();
    std::string otherName;   
    
    for (li = 0; li < layerNum; li++) {
        otherName = net->layer_names()[li];
        
        if(strcmp(name, otherName.c_str()) == 0)
            return li;
    }
    
    printf("ERR in %s: Wrong layer name\n", __FUNCTION__);
    return -1;
}

template <typename Dtype>
bool CLnets<Dtype>::getLayBlobSize(int** dims, int* dimSize,
        std::vector<caffe::Blob<Dtype>*> blobs)
{
    unsigned int i, di;
    
    *dimSize = blobs.size() * 4;
    if(*dimSize == 0){
        return true;
    }
    
    *dims = NULL;
    *dims = (int*) malloc(sizeof(int) * *dimSize);
    
    if(!*dims){
        printf("ERR in %s: Allocation failed\n", __FUNCTION__);
        return false;
    }
    
    di = 0;
    for(i = 0; i < blobs.size(); i++){
        (*dims)[di++] = blobs[i]->num();
        (*dims)[di++] = blobs[i]->channels();
        (*dims)[di++] = blobs[i]->height();
        (*dims)[di++] = blobs[i]->width();                
    }        
    
    return true;
}  

template <typename Dtype>
bool CLnets<Dtype>::getTopBlobSize(int** dims, int* dimSize, unsigned int layerIdx)
{   
    if(layerIdx >= net->layers().size()){
        printf("ERR in %s: Wrong layer index\n", __FUNCTION__);
        return false;
    }
    
    return getLayBlobSize(dims, dimSize, net->top_vecs()[layerIdx]);
}

template <typename Dtype>
bool CLnets<Dtype>::getBottomBlobSize(int** dims, int* dimSize, unsigned int layerIdx)
{  
    if(layerIdx >= net->layers().size()){
        printf("ERR in %s: Wrong layer index\n", __FUNCTION__);
        return false;
    }
    
    return getLayBlobSize(dims, dimSize, net->bottom_vecs()[layerIdx]);
}

template <typename Dtype>
bool CLnets<Dtype>::getParamBlobSize(int** dims, int* dimSize, unsigned int layerIdx)
{    
    unsigned int i, di;
    std::vector<caffe::shared_ptr<caffe::Blob<Dtype> > > parBl;
    if(layerIdx >= net->layers().size()){
        printf("ERR in %s: Wrong layer index\n", __FUNCTION__);
        return false;
    }
    
    parBl = net->layers()[layerIdx].get()->blobs();
    
    *dimSize = parBl.size() * 4;    
    if(*dimSize == 0)
        return true;
    
    *dims = NULL;
    *dims = (int*) malloc(sizeof(int) * *dimSize);
    
    if(!*dims){
        printf("ERR in %s: Allocation failed\n", __FUNCTION__);
        return false;
    }
    
    di = 0;
    for(i = 0; i < parBl.size(); i++){
        (*dims)[di++] = parBl[i].get()->num();
        (*dims)[di++] = parBl[i].get()->channels();
        (*dims)[di++] = parBl[i].get()->height();
        (*dims)[di++] = parBl[i].get()->width();                
    }        

    return true;
}

template <typename Dtype>
bool CLnets<Dtype>::getInputSize(int** dims, int* dimSize)
{   
    if(net->input_blobs().size() < 1){
        printf("ERR in %s: net has no input blobs and probaly uses data layer\n",
                __FUNCTION__);
        return false;
    }
    
    return getLayBlobSize(dims, dimSize, net->input_blobs());
}

template <typename Dtype>
bool CLnets<Dtype>::getTopBlob(Dtype** data, unsigned int layerIdx, unsigned int blobIdx)
{
    if (layerIdx >= net->layers().size()) {
        printf("ERR in %s: Wrong layer index\n", __FUNCTION__);
        return false;
    }

    if (blobIdx >= net->top_vecs()[layerIdx].size()
            || net->top_vecs()[layerIdx].size() == 0) {
        printf("ERR in %s: Wrong blob index\n", __FUNCTION__);
        return false;
    }
    
    *data = (Dtype*) net->top_vecs()[layerIdx][blobIdx]->cpu_data();
    return true;
}

template <typename Dtype>
bool CLnets<Dtype>::getBottomBlob(Dtype** data, unsigned int layerIdx, unsigned int blobIdx)
{
    if (layerIdx >= net->layers().size()) {
        printf("ERR in %s: Wrong layer index\n", __FUNCTION__);
        return false;
    }

    if (blobIdx >= net->bottom_vecs()[layerIdx].size()
            || net->bottom_vecs()[layerIdx].size() == 0) {
        printf("ERR in %s: Wrong blob index\n", __FUNCTION__);
        return false;
    }
    
    *data = (Dtype*) net->bottom_vecs()[layerIdx][blobIdx]->cpu_data();
    return true;
}

template <typename Dtype>
bool CLnets<Dtype>::getParamBlob(Dtype** data, unsigned int layerIdx, unsigned int blobIdx)
{
    if (layerIdx >= net->layers().size()) {
        printf("ERR in %s: Wrong layer index\n", __FUNCTION__);
        return false;
    }

    if (blobIdx >= net->layers()[layerIdx].get()->blobs().size()
            || net->layers()[layerIdx].get()->blobs().size() == 0) {
        printf("ERR in %s: Wrong blob index\n", __FUNCTION__);
        return false;
    }
    
    *data = (Dtype*) net->layers()[layerIdx].get()->blobs()[blobIdx].get()->cpu_data();
    return true;
}

template <typename Dtype>
bool CLnets<Dtype>::setTopBlob(Dtype** data, unsigned int layerIdx, unsigned int blobIdx)
{
    if (layerIdx >= net->layers().size()) {
        printf("ERR in %s: Wrong layer index\n", __FUNCTION__);
        return false;
    }

    if (blobIdx >= net->top_vecs()[layerIdx].size()
            || net->top_vecs()[layerIdx].size() == 0) {
       printf("ERR in %s: Wrong blob index\n", __FUNCTION__);
        return false;
    }
    
    net->top_vecs()[layerIdx][blobIdx]->set_cpu_data(*data);
    return true;
}

template <typename Dtype>
bool CLnets<Dtype>::setBottomBlob(Dtype** data, unsigned int layerIdx, unsigned int blobIdx)
{
    if (layerIdx >= net->layers().size()) {
        printf("ERR in %s: Wrong layer index\n", __FUNCTION__);
        return false;
    }

    if (blobIdx >= net->bottom_vecs()[layerIdx].size()
            || net->bottom_vecs()[layerIdx].size() == 0) {
        printf("ERR in %s: Wrong blob index\n", __FUNCTION__);
        return false;
    }
    
    net->bottom_vecs()[layerIdx][blobIdx]->set_cpu_data(*data);
    return true;
}

template <typename Dtype>
bool CLnets<Dtype>::setParamBlob(Dtype** data, unsigned int layerIdx, unsigned int blobIdx)
{
    if (layerIdx >= net->layers().size()) {
        printf("ERR in %s: Wrong layer index\n", __FUNCTION__);
        return false;
    }

    if (blobIdx >= net->layers()[layerIdx].get()->blobs().size()
            || net->layers()[layerIdx].get()->blobs().size() == 0) {
        printf("ERR in %s: Wrong blob index\n", __FUNCTION__);
        return false;
    }
    
    net->layers()[layerIdx].get()->blobs()[blobIdx].get()->set_cpu_data(*data);
    return true;
}

template <typename Dtype>
bool CLnets<Dtype>::setInput(Dtype **inputData)
{    
    if(net->input_blobs().size() < 1){
        printf("ERR in %s: net has no input blobs and probaly uses data layer\n",
                __FUNCTION__);
        return false;
    }
    
    net->input_blobs()[0]->set_cpu_data(*inputData);
    return true;
}

INSTANTIATE_CLASS(CLnets);
