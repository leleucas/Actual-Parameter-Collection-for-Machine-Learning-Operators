#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <fstream>
#include <iostream>
#include <time.h>
#include <cudnn.h>
#include <string>
#include <vector>
#include <algorithm>
#include <cfloat>
#include "json/json.h"
using namespace std;

#define ErrChk(code)                                                           \
  { Assert((code), __FILE__, __LINE__); }
inline void Assert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    printf("CUDA Runtime Error: %s:%d:'%s'\n", file, line,
           cudaGetErrorString(code));
    // exit(EXIT_FAILURE);
  }
}
inline void Assert(cudnnStatus_t code, const char *file, int line) {
  if (code != CUDNN_STATUS_SUCCESS) {
     printf("cuDNN API Error: %s:%d:'%s'\n", file, line,
            cudnnGetErrorString(code));
    // exit(EXIT_FAILURE);
  }
}

inline int _ConvertSMVer2Cores(int major, int minor) { //根据GPU Arch确定每个SM上有多少个SP
  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM; // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = { { 0x30, 192 },
                                      { 0x32, 192 },
                                      { 0x35, 192 },
                                      { 0x37, 192 },
                                      { 0x50, 128 },
                                      { 0x52, 128 },
                                      { 0x53, 128 },
                                      { 0x60, 64 },
                                      { 0x61, 128 },
                                      { 0x62, 128 },
                                      { 0x70, 64 },
                                      { 0x72, 64 },
                                      { 0x75, 64 },
                                      { -1, -1 } };

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  printf("MapSMtoCores for SM %d.%d is undefined."
         "  Default to use %d Cores/SM\n",
         major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores; //如果没找到 就选择0x75对应的64
}

int main(int argc, char **argv) {

  printf("%s Starting...\n\n", argv[0]);
  printf("CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

  int deviceCount = 0;
  ErrChk(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    printf("There are no available device(s) that support CUDA\n");
  } else {
    printf("Detected %d CUDA Capable device(s)\n", deviceCount);
  }

  int dev = 7;
  ErrChk(cudaSetDevice(dev));

  // cuDNN implementation

  cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;

  Json::Reader reader;
  Json::FastWriter fwriter;
  Json::Value op_samples;
  Json::Value result_json;

  assert(argc == 3);
  std::string batchsize_str = argv[1];
  std::string kernelsize_str = argv[2];
  
  std::ifstream i("./data/conv2d_fakepara"+batchsize_str+"_"+kernelsize_str+".json");
  if (!reader.parse(i, op_samples, false))
  {
      printf("parse failed! \n");
  }
  i.close();

  std::ofstream result_csv("./data/conv2d_perf"+batchsize_str+"_"+kernelsize_str+".csv");
  result_csv << "Batch_sizeN,input_channel,image_size_H,image_size_W,output_channel,kernel_sizeR,kernel_sizeS,strideU,strideV,pad_h,pad_w,algo0,algo1,algo2,algo3,algo4,algo5,algo6,algo7,algoMin\n";
  
  int BATCH = op_samples["N"].size();
  for (int sample_iter = 0; sample_iter < BATCH; sample_iter++) {
    int N = op_samples["N"][sample_iter].asInt();
    int C = op_samples["C_in"][sample_iter].asInt();//输入通道
    int H = op_samples["H"][sample_iter].asInt();
    int W = op_samples["W"][sample_iter].asInt();

    int K = op_samples["C_out"][sample_iter].asInt();//输出通道
    int R = op_samples["kernel_R"][sample_iter].asInt();//filter长
    int S = op_samples["kernel_S"][sample_iter].asInt();//filter宽

    int U = op_samples["strideU"][sample_iter].asInt();//stride
    int V = op_samples["strideV"][sample_iter].asInt();//stride
    int pad_h = op_samples["pad_h"][sample_iter].asInt();
    int pad_w = op_samples["pad_w"][sample_iter].asInt();

    int P = floor((H + 2 * pad_h - R) / U) + 1;//输出
    int Q = floor((W + 2 * pad_w - S) / V) + 1;//输出

    std::string config = "["+std::to_string(N) +", "+std::to_string(C)+", "+std::to_string(H)+", "+std::to_string(W)+"]"+" / kernel: ["+std::to_string(K)+", "+std::to_string(R)+", "+std::to_string(S)+"] / stride:"+to_string(U)+", pad, "+std::to_string(pad_h)+"\n";
    std::string config_csv = std::to_string(N) +","+std::to_string(C)+","+std::to_string(H)+","+std::to_string(W)+","+std::to_string(K)+","+std::to_string(R)+","+std::to_string(S)+","+to_string(U)+","+to_string(V)+","+std::to_string(pad_h)+","+std::to_string(pad_w);
    result_csv << config_csv << ",";

    long kernel_size = (long)K * (long)C * (long)R * (long)S;
    long input_size = (long)N * (long)C * (long)H * (long)W;
    long output_size = (long)N * (long)K * (long)P * (long)Q;

    std::vector<float> result_vec(8, FLT_MAX);

    if ((1.0*input_size*sizeof(float) + 1.0*kernel_size*sizeof(float) + 1.0*output_size*sizeof(float))/1024/1024/1024 >= 32) {
       for (int w = 0; w < 8; w++) {
	result_csv << "NOT SUPPORT!,";
       }
       result_csv<<"NOT SUPPORT!\n";
       printf("%d\n", sample_iter);
       continue;
    }

    float *h_Var0 = (float *)malloc(sizeof(float) * input_size);
    if (h_Var0 == NULL) {
      printf("Error in alloc h_Var0 %f %d GB\n", 1.0*input_size*sizeof(float)/1024/1024/1024, input_size);
      exit(EXIT_FAILURE);
    }
    float *h_Var1 = (float *)malloc(sizeof(float) * kernel_size);
    if (h_Var1 == NULL) {
      printf("Error in alloc h_Var1 %f GB\n", 1.0*kernel_size*sizeof(float)/1024/1024/1024);
      exit(EXIT_FAILURE);
    }
    float *h_Var2 = (float *)malloc(sizeof(float) * output_size);
    if (h_Var2 == NULL) {
      printf("Error in alloc h_Var2 %f GB\n", 1.0*output_size*sizeof(float)/1024/1024/1024);
      exit(EXIT_FAILURE);
    }
    for (long i = 0; i < input_size; ++i) {
      h_Var0[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
      // h_Var0[i] = static_cast <float> (1);
    }
    // printf("pass 1.4\n");
    for (long i = 0; i < kernel_size; ++i) {
      h_Var1[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
      // h_Var1[i] = static_cast <float> (1);
    }
    for (long i = 0; i < output_size; ++i) {
      h_Var2[i] = static_cast<float>(0);
    }
    float *y, *filter, *x;
    
    ErrChk(cudaMalloc((void **)&x, sizeof(float) * input_size));
    ErrChk(cudaMalloc((void **)&filter, sizeof(float) * kernel_size));
    ErrChk(cudaMalloc((void **)&y, sizeof(float) * output_size));


    ErrChk(cudaMemcpy(x, h_Var0, sizeof(float) * input_size, cudaMemcpyHostToDevice));
    ErrChk(cudaMemcpy(filter, h_Var1, sizeof(float) * kernel_size, cudaMemcpyHostToDevice));
    ErrChk(cudaMemcpy(y, h_Var2, sizeof(float) * output_size, cudaMemcpyHostToDevice));


    cudnnHandle_t handle;
    ErrChk(cudnnCreate(&handle));

    float one = 1.0, zero = 0.0;

    cudnnTensorDescriptor_t yDesc;
    ErrChk(cudnnCreateTensorDescriptor(&yDesc));
    cudnnStatus_t code = cudnnSetTensor4dDescriptor(yDesc, format, CUDNN_DATA_FLOAT, N, K, P, Q);
    if (code != CUDNN_STATUS_SUCCESS) {
       for (int w = 0; w < 8; w++) {
	result_csv << "NOT SUPPORT!,";
       }
       result_csv<<"NOT SUPPORT!\n";
       printf("%d\n", sample_iter);
       cudaFree(y);
       cudaFree(filter);
       cudaFree(x);
       free(h_Var0);
       free(h_Var1);
       free(h_Var2);
       continue;
    }
    ErrChk(code);

    cudnnFilterDescriptor_t filterDesc;
    ErrChk(cudnnCreateFilterDescriptor(&filterDesc));
    code = cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT,
                                      CUDNN_TENSOR_NCHW, K, C, R, S);
    if (code != CUDNN_STATUS_SUCCESS) {
       for (int w = 0; w < 8; w++) {
	result_csv << "NOT SUPPORT!,";
       }
       result_csv<<"NOT SUPPORT!\n";
       cudaFree(y);
       cudaFree(filter);
       cudaFree(x);
       free(h_Var0);
       free(h_Var1);
       free(h_Var2);
       ErrChk(cudnnDestroyTensorDescriptor(yDesc));
       printf("%d\n", sample_iter);
       continue;
    }
    ErrChk(code);

    cudnnTensorDescriptor_t xDesc;
    ErrChk(cudnnCreateTensorDescriptor(&xDesc));
    code = cudnnSetTensor4dDescriptor(xDesc, format, CUDNN_DATA_FLOAT, N, C, H, W);
    if (code != CUDNN_STATUS_SUCCESS) {
       for (int w = 0; w < 8; w++) {
	result_csv << "NOT SUPPORT!,";
       }
       result_csv<<"NOT SUPPORT!\n";
       printf("%d\n", sample_iter);
       cudaFree(y);
       cudaFree(filter);
       cudaFree(x);
       free(h_Var0);
       free(h_Var1);
       free(h_Var2);
       ErrChk(cudnnDestroyTensorDescriptor(yDesc));
       ErrChk(cudnnDestroyFilterDescriptor(filterDesc));
       continue;
    }
    ErrChk(code);

    cudnnConvolutionDescriptor_t convDesc;
    ErrChk(cudnnCreateConvolutionDescriptor(&convDesc));
    code = cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, U, V, 1, 1,
                                          CUDNN_CROSS_CORRELATION,
                                          CUDNN_DATA_FLOAT);
    if (code != CUDNN_STATUS_SUCCESS) {
       for (int w = 0; w < 8; w++) {
	result_csv << "NOT SUPPORT!,";
       }
       result_csv<<"NOT SUPPORT!\n";
       cudaFree(y);
       cudaFree(filter);
       cudaFree(x);
       free(h_Var0);
       free(h_Var1);
       free(h_Var2);
       ErrChk(cudnnDestroyTensorDescriptor(yDesc));
       ErrChk(cudnnDestroyFilterDescriptor(filterDesc));
       ErrChk(cudnnDestroyTensorDescriptor(xDesc));
       printf("%d\n", sample_iter);
       continue;
    }
    ErrChk(code);

    ErrChk(cudnnSetConvolutionMathType(convDesc, CUDNN_FMA_MATH)); // turn off tensor core


    float *extra;

    // cudnnConvolutionFwdAlgo_t fwd_algo = fwd_algo_perf[0].algo;
    for (int algo_i = 0; algo_i < 8; algo_i++)
    {
      try
      {
        cudnnConvolutionFwdAlgo_t fwd_algo = cudnnConvolutionFwdAlgo_t(algo_i);
        size_t fwd_workspace_size;
        cudnnStatus_t retStatus = cudnnGetConvolutionForwardWorkspaceSize(
            handle, xDesc, filterDesc, convDesc, yDesc, fwd_algo,
            &fwd_workspace_size);
        if (retStatus != CUDNN_STATUS_SUCCESS || (float)fwd_workspace_size/1024/1024 > 30000)
        {
          continue;
        }
        ErrChk(cudaMalloc((void **)&extra, fwd_workspace_size));

        float time_sum = 0.0;
        int count = 0;
        for (int i = 0; i < 10; i++)
        {
          cudaEvent_t start, stop;
          cudaEventCreate(&start);
          cudaEventCreate(&stop) ;
          cudaEventRecord(start, 0) ;
          ErrChk(cudnnConvolutionForward(handle, &one, xDesc, x, filterDesc, filter, convDesc, fwd_algo, extra, fwd_workspace_size, &zero, yDesc, y));
          cudaEventRecord(stop, 0) ;
          cudaEventSynchronize(stop);
          float elapsedTime;
          cudaEventElapsedTime(&elapsedTime, start, stop);
          if (i >= 1)
          {
            count++;
            time_sum += elapsedTime;
          }
        }
        
        result_vec[algo_i] = double(time_sum)/count;
        
        cudaFree(extra);
      }
      catch(const char *msg)
      {
        printf("%s", msg);
      }
      
    }
    for (int i = 0; i < 8; i++)
    {
      if (result_vec[i] == FLT_MAX)
        result_csv << "NOT SUPPORT!,";
      else {
        result_csv << std::to_string(result_vec[i])+",";
      }
    }
    
    int minElementIndex = std::min_element(result_vec.begin(), result_vec.end()) - result_vec.begin();
    result_csv << std::to_string(minElementIndex);
    result_csv << "\n";


    cudaFree(y);
    cudaFree(filter);
    cudaFree(x);
    free(h_Var0);
    free(h_Var1);
    free(h_Var2);
    ErrChk(cudnnDestroyTensorDescriptor(xDesc));
    ErrChk(cudnnDestroyTensorDescriptor(yDesc));
    ErrChk(cudnnDestroyFilterDescriptor(filterDesc));
    ErrChk(cudnnDestroyConvolutionDescriptor(convDesc));
    printf("%d\n", sample_iter);
  }

  result_csv.close();
  exit(EXIT_SUCCESS);
}
