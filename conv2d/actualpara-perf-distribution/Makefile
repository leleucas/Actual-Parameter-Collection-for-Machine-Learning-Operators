#GENCODE_FLAGS = -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70
GENCODE_FLAGS = -gencode arch=compute_70,code=sm_70

all: conv2d_cudnn_timing conv2d_nsight_profile

conv2d_timing:conv2d_cudnn_timing.o
	/usr/local/cuda-11.6/bin/nvcc $^ -o $@ -lcudnn ${GENCODE_FLAGS} -L/usr/local/lib /usr/local/lib/libjsoncpp.a

conv2d_cudnn_timing.o:conv2d_cudnn_timing.cu
	/usr/local/cuda-11.6/bin/nvcc -c $< -o $@ --std=c++11 ${GENCODE_FLAGS}

conv2d_nsight_profile:conv2d_nsight_profile.o
	/usr/local/cuda-11.6/bin/nvcc $^ -o $@ -lcudnn ${GENCODE_FLAGS} -L/usr/local/lib /usr/local/lib/libjsoncpp.a

conv2d_nsight_profile.o:conv2d_nsight_profile.cu
	/usr/local/cuda-11.6/bin/nvcc -c $< -o $@ --std=c++11 ${GENCODE_FLAGS}

clean:
	rm -f conv2d_cudnn_timing.o conv2d_timing conv2d_nsight_profile conv2d_nsight_profile.o
