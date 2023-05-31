#GENCODE_FLAGS = -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70
GENCODE_FLAGS = -gencode arch=compute_70,code=sm_70

all: gen_fakepara conv2d_nsight_profile

gen_fakepara:gen_fakepara.o
	nvcc $^ -o $@ -lcudnn ${GENCODE_FLAGS} -I/home/jianglijuan/jsoncpp-master/include -L/home/jianglijuan/jsoncpp-master/build/lib/ /home/jianglijuan/jsoncpp-master/build/lib/libjsoncpp.a

gen_fakepara.o:gen_fakepara.cu
	nvcc -c $< -o $@ --std=c++11 ${GENCODE_FLAGS} -I/home/jianglijuan/jsoncpp-master/include


conv2d_nsight_profile:conv2d_nsight_profile.o
	nvcc $^ -o $@ -lcudnn ${GENCODE_FLAGS} -I/home/jianglijuan/jsoncpp-master/include -L/home/jianglijuan/jsoncpp-master/build/lib/ /home/jianglijuan/jsoncpp-master/build/lib/libjsoncpp.a

conv2d_nsight_profile.o:conv2d_nsight_profile.cu
	nvcc -c $< -o $@ --std=c++11 ${GENCODE_FLAGS} -I/home/jianglijuan/jsoncpp-master/include


clean:
	rm -f gen_fakepara.o gen_fakepara conv2d_nsight_profile conv2d_nsight_profile.o
