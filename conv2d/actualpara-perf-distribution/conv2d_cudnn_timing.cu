#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <iostream>
#include <time.h>
#include <cudnn.h>
#include <string>
#include <vector>
#include <algorithm>
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

  int dev = 0;
  for (dev = 0; dev < deviceCount; ++dev) {

    ErrChk(cudaSetDevice(dev));

    cudaDeviceProp deviceProp;
    ErrChk(cudaGetDeviceProperties(&deviceProp, dev));

    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
    printf("GPU Max Clock rate: %0.2f (GHz)\n", deviceProp.clockRate * 1e-6f);

    float freq = deviceProp.clockRate * 1e-6f; // GHz
    int nCore = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
    int nSM = deviceProp.multiProcessorCount;
    printf("(%2d) Multiprocessors, (%2d) CUDA Cores/MP: %d CUDA Cores\n", nSM,
           nCore, nSM * nCore);

    float peakPerf = nSM * nCore * freq * 2;

    printf("GPU Peak Performance is %0.2f GFlops.\n", peakPerf);
  }

  // cuDNN implementation

  // ErrChk(cudaSetDevice(dev));

  cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;

  Json::Reader reader;
  Json::FastWriter fwriter;
  Json::Value op_samples;
  Json::Value result_json;
  
  std::ifstream i("./conv2d_all.json");
  if (!reader.parse(i, op_samples, false))
  {
      printf("parse failed! \n");
  }
  i.close();

  std::ofstream result_csv("./conv2d_results_2080ti.csv");
  result_csv << "Batch_sizeN,input_channel,image_size_H,image_size_W,output_channel,kernel_sizeR,kernel_sizeS,strideU,strideV,pad_h,pad_w,algo0,algo1,algo2,algo3,algo4,algo5,algo6,algo7,algoSelected,algoMin\n";
  int BATCH = op_samples["input_size"].size();
  for (int sample_iter = 0; sample_iter < BATCH; sample_iter++)
  {
    int N = op_samples["input_size"][sample_iter][0].asInt();
    int C = op_samples["input_size"][sample_iter][1].asInt();//输入通道
    int H = op_samples["input_size"][sample_iter][2].asInt();
    int W = op_samples["input_size"][sample_iter][3].asInt();

    int K = op_samples["kernel_size"][sample_iter][0].asInt();//输出通道
    int R = op_samples["kernel_size"][sample_iter][2].asInt();//filter长
    int S = op_samples["kernel_size"][sample_iter][3].asInt();//filter宽

    int U = op_samples["stride"][sample_iter][0].asInt();//stride
    int V = op_samples["stride"][sample_iter][1].asInt();//stride
    int pad_h = op_samples["padding"][sample_iter][0].asInt();
    int pad_w = op_samples["padding"][sample_iter][1].asInt();

    int P = floor((H + 2 * pad_h - R) / U) + 1;//输出
    int Q = floor((W + 2 * pad_w - S) / V) + 1;//输出

    std::string config = "["+std::to_string(N) +", "+std::to_string(C)+", "+std::to_string(H)+", "+std::to_string(W)+"]"+" / kernel: ["+std::to_string(K)+", "+std::to_string(R)+", "+std::to_string(S)+"] / stride:"+to_string(U)+", pad, "+std::to_string(pad_h)+"\n";
    std::string config_csv = std::to_string(N) +","+std::to_string(C)+","+std::to_string(H)+","+std::to_string(W)+","+std::to_string(K)+","+std::to_string(R)+","+std::to_string(S)+","+to_string(U)+","+to_string(V)+","+std::to_string(pad_h)+","+std::to_string(pad_w)+",";
    result_csv << config_csv;

    int kernel_size = K * C * R * S;
    int input_size = N * C * H * W ;
    int output_size = N * K * P * Q;


    // alloc host memory
    float *h_Var0 = (float *)malloc(sizeof(float) * input_size);
    if (h_Var0 == NULL) {
      printf("Error in alloc h_Var0\n");
      exit(EXIT_FAILURE);
    }
    float *h_Var1 = (float *)malloc(sizeof(float) * kernel_size);
    if (h_Var1 == NULL) {
      printf("Error in alloc h_Var1\n");
      exit(EXIT_FAILURE);
    }
    float *h_Var2 = (float *)malloc(sizeof(float) * output_size);
    if (h_Var2 == NULL) {
      printf("Error in alloc h_Var2\n");
      exit(EXIT_FAILURE);
    }

    // generate dummy data for test
    for (int i = 0; i < input_size; ++i) {
      h_Var0[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
      // h_Var0[i] = static_cast <float> (1);
    }
    for (int i = 0; i < kernel_size; ++i) {
      h_Var1[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
      // h_Var1[i] = static_cast <float> (1);
    }
    for (int i = 0; i < output_size; ++i) {
      h_Var2[i] = static_cast<float>(0);
    }
    float *y, *filter, *x;

    ErrChk(cudaMalloc((void **)&y, sizeof(float) * output_size));
    ErrChk(cudaMalloc((void **)&filter, sizeof(float) * kernel_size));
    ErrChk(cudaMalloc((void **)&x, sizeof(float) * input_size));

    ErrChk(cudaMemcpy(x, h_Var0, sizeof(float) * input_size, cudaMemcpyHostToDevice));
    ErrChk(cudaMemcpy(filter, h_Var1, sizeof(float) * kernel_size, cudaMemcpyHostToDevice));
    ErrChk(cudaMemcpy(y, h_Var2, sizeof(float) * output_size, cudaMemcpyHostToDevice));

    /*  2. cuDNN preparation  */
    cudnnHandle_t handle;
    ErrChk(cudnnCreate(&handle));

    float one = 1.0, zero = 0.0;

    cudnnTensorDescriptor_t yDesc;
    ErrChk(cudnnCreateTensorDescriptor(&yDesc));
    ErrChk(
        cudnnSetTensor4dDescriptor(yDesc, format, CUDNN_DATA_FLOAT, N, K, P, Q));

    cudnnFilterDescriptor_t filterDesc;
    ErrChk(cudnnCreateFilterDescriptor(&filterDesc));
    ErrChk(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT,
                                      CUDNN_TENSOR_NCHW, K, C, R, S));
    cudnnTensorDescriptor_t xDesc;
    ErrChk(cudnnCreateTensorDescriptor(&xDesc));
    ErrChk(cudnnSetTensor4dDescriptor(xDesc, format, CUDNN_DATA_FLOAT, N, C, H, W));

    cudnnConvolutionDescriptor_t convDesc;
    ErrChk(cudnnCreateConvolutionDescriptor(&convDesc));
    ErrChk(cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, U, V, 1, 1,
                                          CUDNN_CROSS_CORRELATION,
                                          CUDNN_DATA_FLOAT));
    ErrChk(cudnnSetConvolutionMathType(convDesc, CUDNN_FMA_MATH)); // turn off tensor core

    const int n_requestedAlgo = 10;
    int n_returnedAlgo;
    cudnnConvolutionFwdAlgoPerf_t fwd_algo_perf[n_requestedAlgo];
    ErrChk(cudnnFindConvolutionForwardAlgorithm(
        handle, xDesc, filterDesc, convDesc, yDesc, n_requestedAlgo,
        &n_returnedAlgo, fwd_algo_perf));
    printf("algo choice: %d, time: %f\n", fwd_algo_perf[0].algo, fwd_algo_perf[0].time);
    if (n_returnedAlgo == 0) {
      printf("No cudnnConvolutionFwdAlgoPerf_t found");
      exit(1);
    }
    float *extra;

    // cudnnConvolutionFwdAlgo_t fwd_algo = fwd_algo_perf[0].algo;
    std::vector<float> result_vec(8, MAXFLOAT);
    for (int algo_i = 0; algo_i < 8; algo_i++)
    {
      try
      {
        cudnnConvolutionFwdAlgo_t fwd_algo = cudnnConvolutionFwdAlgo_t(algo_i);
        printf("algo choice: %d\t", fwd_algo);
        size_t fwd_workspace_size;
        cudnnStatus_t retStatus = cudnnGetConvolutionForwardWorkspaceSize(
            handle, xDesc, filterDesc, convDesc, yDesc, fwd_algo,
            &fwd_workspace_size);
        printf("worksapce: %.2fMB\t", (float)fwd_workspace_size/1024/1024);
        if (retStatus != CUDNN_STATUS_SUCCESS || (float)fwd_workspace_size/1024/1024 > 30000)
        {
          printf("\n");
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
          // printf("algo time cost: %f\n", double(end - start)/CLOCKS_PER_SEC);
        }
        
        result_vec[algo_i] = double(time_sum)/count;
        printf("algo time avg cost: %lf ms\n", double(time_sum)/count);
        
        cudaFree(extra);
      }
      catch(const char *msg)
      {
        printf("%s", msg);
      }
      
    }
    for (int i = 0; i < 8; i++)
    {
      if (result_vec[i] == MAXFLOAT)
        result_csv << "NOT SUPPORT!,";
      else
        result_csv << std::to_string(result_vec[i])+",";
    }
    
    int minElementIndex = std::min_element(result_vec.begin(), result_vec.end()) - result_vec.begin();
    result_json[std::to_string(minElementIndex)] = result_json[std::to_string(minElementIndex)].asString()+config;
    result_csv << std::to_string(fwd_algo_perf[0].algo)+",";
    result_csv << std::to_string(minElementIndex)+",";
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

  std::ofstream result_txt;
  for (int out_i = 0; out_i < 8; out_i++)
  {
    result_txt.open("./output/best_algo/algo_"+std::to_string(out_i)+".txt", ios::out);
    result_txt << result_json[std::to_string(out_i)].asString();
    result_txt.close();
  }
  result_csv.close();
  exit(EXIT_SUCCESS);
}