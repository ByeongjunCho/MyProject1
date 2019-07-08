#include <cuda.h>
#include <cudnn.h>
#include <iostream>
#include <device_launch_parameters.h>
/* 입력데이터 1개
   입력데이터 NCHW = 1x2x4x4
   최종 출력 NCHW = 1x2x1x1
   컨볼루션 필터 NCHW = 1x2x3x3, stride 1x1, padding 1x1, same convolution
   풀링 strides 2x2, padding 0x0
   Fully connected layer NCHW = 2x1x2x2
   예측 클래스의 수 2개*/

using namespace std;

void checkCUDNN(cudnnStatus_t status)
{
	if (status != CUDNN_STATUS_SUCCESS)
		cout << "[ERROR] CUDNN " << status << endl;
}

void checkCUDA(cudaError_t error)
{
	if (error != CUDA_SUCCESS)
		cout << "[ERROR] CUDA " << error << endl;
}

void print(char* title, float*src, int filter_num, int h, int w)
{
	cout << title << endl;
	for (int i = 0; i < filter_num; i++) {
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				printf("%.0f ", src[i*h*w + y*w + x]);
			}
			cout << endl;
		}
		cout << endl;
	}
}

void convNCHW()   // [N-count 갯수, channel, height, width] 형식의 convolution
{
	const int batch_count = 1;// 입력 데이터 개수
	const int in_channel = 2; // 입력 데이터 채널 개수
	const int in_height = 4; // 입력 데이터의 세로 길이
	const int in_width = 4; // 입력 데이터 가로 길이
	const int out_channel = 2; // 출력 클래스 수
	const int filter_width = 3; // convolution 가로 길이
	const int filter_height = 3; // convolution 세로 길이
	const int filter_num = 1; // 컨볼루션 필터 개수
	const int padding_w = 1; // 컨볼루션 패딩 가로. 필터의 가로 세로 길이가 3이고 패딩이 1,1이면 same convolution이 된다.
	const int padding_h = 1; // 컨볼루션 패딩 세로
	const int stride_horizontal = 1; // stride
	const int stride_vertical = 1; // stride
	const int pool_window_w = 2;
	const int pool_window_h = 2;
	const int pool_stride_horizontal = 2; // pooling시 strides->1/2만큼 크기 감소
	const int pool_stride_vertical = 2;
	const int pool_padding_horizontal = 0;
	const int pool_padding_vertical = 0;
	const int pool_w = in_width / pool_stride_horizontal;
	const int pool_h = in_height / pool_stride_vertical;
	const int src_len = batch_count * filter_num * in_height * in_width;
	const int pool_len = batch_count * filter_num * pool_w * pool_h;
	float inData[batch_count][in_channel][in_height][in_width]; // host 입력 데이터
	float outData[batch_count][filter_num][in_height][in_width]; // host 출력
	float *hostArray = new float[src_len];
	float *inData_d; // device 입력 데이터
	float *outData_d, *outData1_d; // device 출력 데이터
	float *filterData_d; // device 컨볼루션 필터 데이터
	float *filterData2_d; // device FCN 필터 데이터
	float *biasData_d;
	void *workSpace; // cudnn이 작업 중 사용할 버퍼 메모리

					 // 입력 데이터 셋팅
	for (int i = 0; i < in_channel; i++) {
		for (int y = 0; y < in_height; y++) {
			for (int x = 0; x < in_width; x++) {
				inData[0][i][y][x] = i * in_channel * in_height * in_width + y * in_width + x;
			}
		}
	}

	// 필터(가중치) 셋팅
	float filterData[filter_num][in_channel][filter_height][filter_width] =
	{ {
		{ { 0.0f,0.0f,0.0f },{ 0.0f,1.0f,0.0f },{ 0.0f,0.0f,0.0f } },
		{ { 0.0f,0.0f,0.0f },{ 0.0f,0.0f,1.0f },{ 0.0f,0.0f,0.0f } }
		} };

	// Fully connected Layer 가중치
	float filterData2[out_channel][filter_num][pool_h][pool_w] =
	{ { { { 0.1f,0.1f },{ 0.1f,0.1f } } },{ { { 0.2f,0.2f },{ 0.2f,0.2f } } }
	};

	float biasData[filter_num] = { -20 };

	cout << "입력 데이터" << endl;
	for (int i = 0; i < in_channel; i++) {
		for (int y = 0; y < in_height; y++) {
			for (int x = 0; x < in_width; x++) {
				printf("%.0f ", inData[0][i][y][x]);
			}
			cout << endl;
		}
		cout << endl;
	}

	cout << "가중치" << endl;
	for (int n = 0; n < filter_num; n++) {
		for (int i = 0; i < in_channel; i++) {
			for (int y = 0; y < filter_height; y++) {
				for (int x = 0; x < filter_width; x++) {
					printf("%.1f ", filterData[n][i][y][x]);
				}
				cout << endl;
			}
			cout << endl;
		}
	}

	// GPU 메모리 할당
	checkCUDA(cudaMalloc((void**)&inData_d, sizeof(inData))); // 입력 데이터 할당
	checkCUDA(cudaMalloc((void**)&outData_d, sizeof(outData))); //출력 데이터
	checkCUDA(cudaMalloc((void**)&outData1_d, sizeof(outData)));// 출력데이터
	checkCUDA(cudaMalloc((void**)&filterData_d, sizeof(filterData))); // 컨볼루션 필터 가중치 
	checkCUDA(cudaMalloc((void**)&biasData_d, sizeof(biasData)));  // 바이어스 메모리
	checkCUDA(cudaMalloc((void**)&filterData2_d, sizeof(filterData2))); //fully connected 가중치 메모리 할당

	// CPU 데이터를 GPU 메모리로 복사
	// NHWC와 NCHW중 선택  
	// NHWC => number, height, width, channel
	// NCHW => number, channel,height, width
	checkCUDA(cudaMemcpy(inData_d, inData, sizeof(inData), cudaMemcpyHostToDevice));// NCHW형식 입력 데이터 cpu=>gpu 복사
	checkCUDA(cudaMemcpy(filterData_d, filterData, sizeof(filterData), cudaMemcpyHostToDevice));
	checkCUDA(cudaMemcpy(filterData2_d, filterData2, sizeof(filterData2), cudaMemcpyHostToDevice));
	checkCUDA(cudaMemcpy(biasData_d, biasData, sizeof(biasData), cudaMemcpyHostToDevice));

	// CUDNN 배열
	cudnnHandle_t cudnnHandle; // cudnn을 사용하기 위한 핸들러
	cudnnTensorDescriptor_t inTensorDesc, outTensorDesc, biasTensorDesc, poolOutTensorDeesc, sftTensorDesc; // 데이터 구조체 선언
	cudnnFilterDescriptor_t filterDesc, filterDesc2; // 필터 구조체 선언
	cudnnConvolutionDescriptor_t convDesc; //컨볼루션 구조체 선언
	cudnnConvolutionDescriptor_t convDesc2; // 컨볼루션 구조체 선언
	cudnnPoolingDescriptor_t poolDesc; // 풀링 구조체 선언
	cudnnActivationDescriptor_t actDesc; // 활성함수 구조체 선언

										 // 할당
	checkCUDNN(cudnnCreate(&cudnnHandle));
	checkCUDNN(cudnnCreateTensorDescriptor(&inTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&outTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&poolOutTensorDeesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&sftTensorDesc));
	checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
	checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc2));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc2));
	checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
	checkCUDNN(cudnnCreateActivationDescriptor(&actDesc));

	//초기화
	//inData_NCHW 정보 - 구조가 [Number][Channel][Height][Width] 형태임을 알려줌
	checkCUDNN(cudnnSetTensor4dDescriptor(inTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_count, in_channel, in_height, in_width)); // 입력 1x2x4x4
	checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filter_num, in_channel, filter_height, filter_width)); // 컨볼루션 필터 1x2x3x3
	checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc2, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_channel, filter_num, pool_h, pool_w)); // 풀 링크 필터 2x1x2x2

																																	   // 컨볼루션의 padding, stride, convolution mode 설정
	checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, // 이전에 만든 convolution descriptor(input/output)를 다룬다
		padding_h,   // padding 파라미터 설정
		padding_w,
		stride_vertical, // strides 파라미터 설정
		stride_horizontal,
		1, // filter height dilation  => 일반적인 convolution 연산
		1, // filter width dilation
		CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT)); // Input. Selects between CUDNN_CONVOLUTION and CUDNN_CROSS_CORRELATION.
	// fully connected network 셋업
	checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc2, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

	// 바이어스 셋업
	checkCUDNN(cudnnSetTensor4dDescriptor(biasTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, filter_num, 1, 1));

	// pooling 셋업
	checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc, // pooling descriptor 
		CUDNN_POOLING_MAX,  // set pooling mode
		CUDNN_PROPAGATE_NAN,  // maxpooling Nan propagation mode
		pool_window_h, // pooling window height
		pool_window_w, // pooling windowl width
		pool_padding_vertical, // padding parameter
		pool_padding_horizontal,
		pool_stride_vertical,  // padding ㅣ stride
		pool_stride_horizontal));

	// 활성함수 Relu셋업
	checkCUDNN(cudnnSetActivationDescriptor(actDesc, // input/output handle descriptor
		CUDNN_ACTIVATION_RELU,  // activation mode
		CUDNN_PROPAGATE_NAN,  // non propagation mode
		0));  // float number. 

	int out_n, out_c, out_h, out_w;
	// 입력데이터를 위에서 셋팅한 대로 컨볼루션 했을때 출력 데이터의 구조 알아내기
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, // 컨볼루션
		inTensorDesc,  // 입력 데이터
		filterDesc,    // 필터
		&out_n,
		&out_c,
		&out_h,
		&out_w));
	printf("conv out shape (n x c x h x w) = (%d x %d x %d x %d)\n", out_n, out_c, out_h, out_w);
	checkCUDNN(cudnnSetTensor4dDescriptor(outTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));
	// input 1x2x4x4 를 1x2x3x3 필터로 same padding convolution 결과 1x1x4x4가 나온다.

	// 풀링 결과 구조 확인
	checkCUDNN(cudnnGetPooling2dForwardOutputDim(poolDesc, outTensorDesc, &out_n, &out_c, &out_h, &out_w));
	printf("pool out shape (n x c x h x w) = (%d x %d x %dx %d)\n", out_n, out_c, out_h, out_w);
	// 풀링 결과 구조 셋업
	checkCUDNN(cudnnSetTensor4dDescriptor(poolOutTensorDeesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));

	// FCN 결과 구조 확인
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc2, poolOutTensorDeesc, filterDesc2, &out_n, &out_c, &out_h, &out_w));
	printf("conv2 out shape (n x c x h x w) = (%d x %d x %d x %d)\n", out_n, out_c, out_h, out_w);
	checkCUDNN(cudnnSetTensor4dDescriptor(sftTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));

	// 입력과 필터, 컨볼루션 패딩, 스트라이드가 위와 같이 주어졌을 때 가장 빠른 알고리즘이 무엇인지 알아내기
	cudnnConvolutionFwdAlgo_t algo;
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
		inTensorDesc,
		filterDesc,
		convDesc,
		outTensorDesc,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		0,
		&algo));

	cout << "Fastest algorithm for conv0 = " << algo << endl;

	cudnnConvolutionFwdAlgo_t algo2;
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
		poolOutTensorDeesc,  // pooling 된 결과값
		filterDesc2,		 // pooling된 값 필터
		convDesc2,			 // convolution descriptor
		sftTensorDesc,		// output tensor
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		0,
		&algo2));

	cout << "Fastest algorithm for conv1 = " << algo2 << endl;

	// 위에서 알아낸 가장 빠른 알고리즘을 사용할 경우 계산과정에서 필요한 버퍼 데이터의 크기를 알아내기
	size_t sizeInBytes = 0;
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
		inTensorDesc,
		filterDesc,
		convDesc,
		outTensorDesc,
		algo,
		&sizeInBytes));

	cout << "sizeInBytes = " << sizeInBytes << endl;
	// 계산과정에서 버퍼 데이터가 필요한 경우가 있다면 메모리 할당
	if (sizeInBytes != 0)checkCUDA(cudaMalloc(&workSpace, sizeInBytes));

	float alpha = 1.0f;
	float beta = 0.0f;
	// 컨볼루션 시작
	// alpah와 beta는 "output = alpha * Op(Input) + beta * output" 에 사용됨
	// 일반 컨보루션은 output = 1 * inputs
	// 그래서          output = 1 * Op(Inputs) + 0 * output이 되도락 alpha와 beta를 1,0으로 셋팅함
	checkCUDNN(cudnnConvolutionForward(cudnnHandle,
		&alpha,     
		inTensorDesc,   // 입력 데이터 구조체
		inData_d,   // 입력 데이터 
		filterDesc,   // 필터 구조체
		filterData_d, // device 컨볼루션 필터 데이터
		convDesc,  // 컨볼루션 구조체
		algo,  
		workSpace,  // cudnn이 작업중 사용할 버퍼 메모리
		sizeInBytes, // 알고리즘을 사용할 때 필요한 버퍼 데이터 크기
		&beta,  
		outTensorDesc,   // 출력 구조체
		outData_d)); // device 출력 데이터

	checkCUDA(cudaMemcpy(hostArray, outData_d, sizeof(float)* src_len, cudaMemcpyDeviceToHost));
	print("conv out", hostArray, filter_num, in_height, in_width);

	// Add bias
	beta = 1.0f;
	checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, biasTensorDesc, biasData_d, &beta, outTensorDesc, outData_d));
	checkCUDA(cudaMemcpy(hostArray, outData_d, sizeof(float) * src_len, cudaMemcpyDeviceToHost));
	print("Add bias out", hostArray, filter_num, in_height, in_width);

	// Activation - Relu
	beta = 0.0f;
	checkCUDNN(cudnnActivationForward(cudnnHandle, actDesc, &alpha, outTensorDesc, outData_d, &beta, outTensorDesc, outData1_d));
	checkCUDA(cudaMemcpy(hostArray, outData1_d, sizeof(float) * src_len, cudaMemcpyDeviceToHost));
	print("Activaton - Relu out", hostArray, filter_num, in_height, in_width);

	// Pooling
	checkCUDNN(cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, outTensorDesc, outData1_d, &beta, poolOutTensorDeesc, outData_d));
	checkCUDA(cudaMemcpy(hostArray, outData_d, sizeof(float)* pool_len, cudaMemcpyDeviceToHost));
	print("pool out", hostArray, filter_num, pool_h, pool_w);

	// FC
	checkCUDNN(cudnnConvolutionForward(cudnnHandle,
		&alpha,
		poolOutTensorDeesc,
		outData_d,
		filterDesc2,
		filterData2_d,
		convDesc2,
		algo2,
		workSpace,
		sizeInBytes,
		&beta,
		sftTensorDesc,
		outData1_d));

	checkCUDA(cudaMemcpy(hostArray, outData1_d, sizeof(float)*out_channel, cudaMemcpyDeviceToHost));
	print("FCN out", hostArray, out_channel, 1, 1);

	// Softmax
	cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, sftTensorDesc, outData1_d, &beta, sftTensorDesc, outData_d);
	checkCUDA(cudaMemcpy(hostArray, outData_d, sizeof(float) * out_channel, cudaMemcpyDeviceToHost));
	print("Softmax out", hostArray, out_channel, 1, 1);

	// 메모리 해제
	checkCUDNN(cudnnDestroyTensorDescriptor(inTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(outTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(biasTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(poolOutTensorDeesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(sftTensorDesc));

	checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
	checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc2));
	checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
	checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
	checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc2));
	checkCUDNN(cudnnDestroyActivationDescriptor(actDesc));
	checkCUDNN(cudnnDestroy(cudnnHandle));

	checkCUDA(cudaFree(inData_d));
	checkCUDA(cudaFree(outData_d));
	checkCUDA(cudaFree(filterData_d));
	checkCUDA(cudaThreadSynchronize());
}

void convNHWC()
{
	const int batch_count = 1;// 입력 데이터 개수
	const int in_channel = 2; // 입력 데이터 채널 개수
	const int in_height = 4; // 입력 데이터의 세로 길이
	const int in_width = 4; // 입력 데이터 가로 길이
	const int out_channel = 2; // 출력 클래스 수
	const int filter_width = 3; // convolution 가로 길이
	const int filter_height = 3; // convolution 세로 길이
	const int filter_num = 1; // 컨볼루션 필터 개수
	const int padding_w = 1; // 컨볼루션 패딩 가로. 필터의 가로 세로 길이가 3이고 패딩이 1,1이면 same convolution이 된다.
	const int padding_h = 1; // 컨볼루션 패딩 세로
	const int stride_horizontal = 1; // stride
	const int stride_vertical = 1; // stride
	const int pool_window_w = 2;
	const int pool_window_h = 2;
	const int pool_stride_horizontal = 2; // pooling시 strides->1/2만큼 크기 감소
	const int pool_stride_vertical = 2;
	const int pool_padding_horizontal = 0;
	const int pool_padding_vertical = 0;
	const int pool_w = in_width / pool_stride_horizontal;
	const int pool_h = in_height / pool_stride_vertical;
	const int src_len = batch_count * filter_num * in_height * in_width;
	const int pool_len = batch_count * filter_num * pool_w * pool_h;
	float inData_NCHW[batch_count][in_channel][in_height][in_width]; // host 입력 데이터 NCHW
	float inData_NHWC[batch_count][in_height][in_width][in_channel]; // host 입력 데이터 NHWC
	float outData[batch_count][filter_num][in_height][in_width]; // host 출력
	float *hostArray = new float[src_len];
	float *inData_d; // device 입력 데이터
	float *outData_d, *outData1_d; // device 출력 데이터
	float *filterData_d; // device 컨볼루션 필터 데이터
	float *filterData2_d; // device FCN 필터 데이터
	float *biasData_d;
	void *workSpace; // cudnn이 작업 중 사용할 버퍼 메모리

	// 입력 데이터 셋팅
	for (int i = 0; i < in_channel; i++) {
		for (int y = 0; y < in_height; y++) {
			for (int x = 0; x < in_width; x++) {
				inData_NCHW[0][i][y][x] = i * in_channel * in_height * in_width + y * in_width + x;
			}
		}
	}

	// 입력 데이터 형태 변환
	for (int i = 0; i < in_channel; i++) {
		for (int y = 0; y < in_height; y++) {
			for (int x = 0; x < in_width; x++) {
				inData_NHWC[0][y][x][i] = inData_NCHW[0][i][y][x];
			}
		}
	}

	// 필터(가중치) 셋팅
	float filterData[filter_num][in_channel][filter_height][filter_width] =  //1x2x3x3
	{ {
		{ { 0.0f,0.0f,0.0f },{ 0.0f,1.0f,0.0f },{ 0.0f,0.0f,0.0f } },
		{ { 0.0f,0.0f,0.0f },{ 0.0f,0.0f,1.0f },{ 0.0f,0.0f,0.0f } }
		} };

	// Fully connected Layer 가중치
	float filterData2[out_channel][filter_num][pool_h][pool_w] =
	{ { { { 0.1f,0.5f },{ 0.3f,0.8f } } },{ { { 0.7f,0.2f },{ 0.6f,0.4f } } }
	};

	float biasData[filter_num] = { -20 };

	cout << "입력 데이터 NCHW" << endl;
	for (int i = 0; i < in_channel; i++) {
		for (int y = 0; y < in_height; y++) {
			for (int x = 0; x < in_width; x++) {
				printf("%.0f ", inData_NCHW[0][i][y][x]);
			}
			cout << endl;
		}
		cout << endl;
	}

	cout << "입력 데이터 변환 NHWC" << endl;
	for (int y = 0; y < in_height; y++) {
		for (int x = 0; x < in_width; x++) {
			for (int i = 0; i < in_channel; i++) {
				printf("%.0f ", inData_NHWC[0][y][x][i]);
			}
			cout << endl;
		}
		cout << endl;
	}

	cout << "가중치" << endl;
	for (int n = 0; n < filter_num; n++) {
		for (int i = 0; i < in_channel; i++) {
			for (int y = 0; y < filter_height; y++) {
				for (int x = 0; x < filter_width; x++) {
					printf("%.1f ", filterData[n][i][y][x]);
				}
				cout << endl;
			}
			cout << endl;
		}
	}

	// GPU 메모리 할당
	checkCUDA(cudaMalloc((void**)&inData_d, sizeof(inData_NCHW))); // 입력 데이터 할당
	checkCUDA(cudaMalloc((void**)&outData_d, sizeof(outData))); //출력 데이터
	checkCUDA(cudaMalloc((void**)&outData1_d, sizeof(outData)));// 출력데이터
	checkCUDA(cudaMalloc((void**)&filterData_d, sizeof(filterData))); // 컨볼루션 필터 가중치 
	checkCUDA(cudaMalloc((void**)&biasData_d, sizeof(biasData)));  // 바이어스 메모리
	checkCUDA(cudaMalloc((void**)&filterData2_d, sizeof(filterData2)));; //fully connected 가중치 메모리 할당

	// CPU 데이터를 GPU 메모리로 복사
	// NHWC와 NCHW중 선택  
	// NHWC => number, height, width, channel
	// NCHW => number, channel,height, width
	checkCUDA(cudaMemcpy(inData_d, inData_NHWC, sizeof(inData_NHWC), cudaMemcpyHostToDevice));// NCHW형식 입력 데이터 cpu=>gpu 복사
	checkCUDA(cudaMemcpy(filterData_d, filterData, sizeof(filterData), cudaMemcpyHostToDevice));
	checkCUDA(cudaMemcpy(filterData2_d, filterData2, sizeof(filterData2), cudaMemcpyHostToDevice));
	checkCUDA(cudaMemcpy(biasData_d, biasData, sizeof(biasData), cudaMemcpyHostToDevice));

	// CUDNN 배열
	cudnnHandle_t cudnnHandle; // cudnn을 사용하기 위한 핸들러
	cudnnTensorDescriptor_t inTensorDesc, outTensorDesc, biasTensorDesc, poolOutTensorDeesc, sftTensorDesc; // 데이터 구조체 선언
	cudnnFilterDescriptor_t filterDesc, filterDesc2; // 필터 구조체 선언
	cudnnConvolutionDescriptor_t convDesc; //컨볼루션 구조체 선언
	cudnnConvolutionDescriptor_t convDesc2; // 컨볼루션 구조체 선언
	cudnnPoolingDescriptor_t poolDesc; // 풀링 구조체 선언
	cudnnActivationDescriptor_t actDesc; // 활성함수 구조체 선언

										 // 할당
	checkCUDNN(cudnnCreate(&cudnnHandle));
	checkCUDNN(cudnnCreateTensorDescriptor(&inTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&outTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&poolOutTensorDeesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&sftTensorDesc));
	checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
	checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc2));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc2));
	checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
	checkCUDNN(cudnnCreateActivationDescriptor(&actDesc));

	//초기화
	//inData_NCHW 정보 - 구조가 [Number][Channel][Height][Width] 형태임을 알려줌
	//checkCUDNN(cudnnSetTensor4dDescriptor(inTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_count, in_channel, in_height, in_width)); // 입력 1x2x4x4
	//inData_NHWC 정보 - 구조가 [Number][Height][Width][Channel] 형태임을 알려줌
	checkCUDNN(cudnnSetTensor4dDescriptor(inTensorDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, batch_count, in_channel, in_height, in_width));
	checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filter_num, in_channel, filter_height, filter_width)); // 컨볼루션 필터 1x2x3x3
	checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc2, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_channel, filter_num, pool_h, pool_w)); // 풀 링크 필터 2x1x2x2

																																	   // 컨볼루션의 padding, stride, convolution mode 설정
	checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,    // 이전에 만든 convolution descriptor(input/output)를 다룬다
		padding_h,   // padding 파라미터 설정
		padding_w,
		stride_vertical, // strides 파라미터 설정
		stride_horizontal,
		1, // filter height dilation  => 일반적인 convolution 연산
		1, // filter width dilation
		CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT)); // Input. Selects between CUDNN_CONVOLUTION and CUDNN_CROSS_CORRELATION.
													 // fully connected network 셋업
	checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc2, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

	// 바이어스 셋업
	checkCUDNN(cudnnSetTensor4dDescriptor(biasTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, filter_num, 1, 1));

	// pooling 셋업
	checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc, // pooling descriptor 
		CUDNN_POOLING_MAX,  // set pooling mode
		CUDNN_PROPAGATE_NAN,  // maxpooling Nan propagation mode
		pool_window_h, // pooling window height
		pool_window_w, // pooling windowl width
		pool_padding_vertical, // padding parameter
		pool_padding_horizontal,
		pool_stride_vertical,  // padding ㅣ stride
		pool_stride_horizontal));

	// 활성함수 Relu셋업
	checkCUDNN(cudnnSetActivationDescriptor(actDesc, // input/output handle descriptor
		CUDNN_ACTIVATION_RELU,  // activation mode
		CUDNN_PROPAGATE_NAN,  // non propagation mode
		0));  // float number. 

	int out_n, out_c, out_h, out_w;
	// 입력데이터를 위에서 셋팅한 대로 컨볼루션 했을때 출력 데이터의 구조 알아내기
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, // 컨볼루션
		inTensorDesc,  // 입력 데이터
		filterDesc,    // 필터
		&out_n,
		&out_c,
		&out_h,
		&out_w));
	printf("conv out shape (n x c x h x w) = (%d x %d x %d x %d)\n", out_n, out_c, out_h, out_w);
	checkCUDNN(cudnnSetTensor4dDescriptor(outTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));
	// input 1x2x4x4 를 1x2x3x3 필터로 same padding convolution 결과 1x1x4x4가 나온다.

	// 풀링 결과 구조 확인
	checkCUDNN(cudnnGetPooling2dForwardOutputDim(poolDesc, outTensorDesc, &out_n, &out_c, &out_h, &out_w));
	printf("pool out shape (n x c x h x w) = (%d x %d x %dx %d)\n", out_n, out_c, out_h, out_w);
	// 풀링 결과 구조 셋업
	checkCUDNN(cudnnSetTensor4dDescriptor(poolOutTensorDeesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));

	// FCN 결과 구조 확인
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc2, poolOutTensorDeesc, filterDesc2, &out_n, &out_c, &out_h, &out_w));
	printf("conv2 out shape (n x c x h x w) = (%d x %d x %d x %d)\n", out_n, out_c, out_h, out_w);
	checkCUDNN(cudnnSetTensor4dDescriptor(sftTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));

	// 입력과 필터, 컨볼루션 패딩, 스트라이드가 위와 같이 주어졌을 때 가장 빠른 알고리즘이 무엇인지 알아내기
	cudnnConvolutionFwdAlgo_t algo;
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
		inTensorDesc,
		filterDesc,
		convDesc,
		outTensorDesc,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		0,
		&algo));

	cout << "Fastest algorithm for conv0 = " << algo << endl;

	cudnnConvolutionFwdAlgo_t algo2;
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
		poolOutTensorDeesc,  // pooling 된 결과값
		filterDesc2,		 // pooling된 값 필터
		convDesc2,			 // convolution descriptor
		sftTensorDesc,		// output tensor
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		0,
		&algo2));

	cout << "Fastest algorithm for conv1 = " << algo2 << endl;

	// 위에서 알아낸 가장 빠른 알고리즘을 사용할 경우 계산과정에서 필요한 버퍼 데이터의 크기를 알아내기
	size_t sizeInBytes = 0;
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
		inTensorDesc,
		filterDesc,
		convDesc,
		outTensorDesc,
		algo,
		&sizeInBytes));

	cout << "sizeInBytes = " << sizeInBytes << endl;
	// 계산과정에서 버퍼 데이터가 필요한 경우가 있다면 메모리 할당
	if (sizeInBytes != 0)checkCUDA(cudaMalloc(&workSpace, sizeInBytes));

	float alpha = 1.0f;
	float beta = 0.0f;
	// 컨볼루션 시작
	// alpah와 beta는 "output = alpha * Op(Input) + beta * output" 에 사용됨
	// 일반 컨보루션은 output = 1 * inputs
	// 그래서          output = 1 * Op(Inputs) + 0 * output이 되도록 alpha와 beta를 1,0으로 셋팅함
	checkCUDNN(cudnnConvolutionForward(cudnnHandle,
		&alpha,
		inTensorDesc,
		inData_d,
		filterDesc,
		filterData_d,
		convDesc,
		algo,
		workSpace,
		sizeInBytes,
		&beta,
		outTensorDesc,
		outData_d));

	checkCUDA(cudaMemcpy(hostArray, outData_d, sizeof(float)* src_len, cudaMemcpyDeviceToHost));
	print("conv out", hostArray, filter_num, in_height, in_width);

	// Add bias
	beta = 1.0f;
	checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, biasTensorDesc, biasData_d, &beta, outTensorDesc, outData_d));
	checkCUDA(cudaMemcpy(hostArray, outData_d, sizeof(float) * src_len, cudaMemcpyDeviceToHost));
	print("Add bias out", hostArray, filter_num, in_height, in_width);

	// Activation - Relu
	beta = 0.0f;
	checkCUDNN(cudnnActivationForward(cudnnHandle, actDesc, &alpha, outTensorDesc, outData_d, &beta, outTensorDesc, outData1_d));
	checkCUDA(cudaMemcpy(hostArray, outData1_d, sizeof(float) * src_len, cudaMemcpyDeviceToHost));
	print("Activaton - Relu out", hostArray, filter_num, in_height, in_width);

	// Pooling
	checkCUDNN(cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, outTensorDesc, outData1_d, &beta, poolOutTensorDeesc, outData_d));
	checkCUDA(cudaMemcpy(hostArray, outData_d, sizeof(float)* pool_len, cudaMemcpyDeviceToHost));
	print("pool out", hostArray, filter_num, pool_h, pool_w);

	// FC
	checkCUDNN(cudnnConvolutionForward(cudnnHandle,
		&alpha,
		poolOutTensorDeesc,
		outData_d,
		filterDesc2,
		filterData2_d,
		convDesc2,
		algo2,
		workSpace,
		sizeInBytes,
		&beta,
		sftTensorDesc,
		outData1_d));

	checkCUDA(cudaMemcpy(hostArray, outData1_d, sizeof(float)*out_channel, cudaMemcpyDeviceToHost));
	print("FCN out", hostArray, out_channel, 1, 1);

	// Softmax
	cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, sftTensorDesc, outData1_d, &beta, sftTensorDesc, outData_d);
	checkCUDA(cudaMemcpy(hostArray, outData_d, sizeof(float) * out_channel, cudaMemcpyDeviceToHost));
	print("Softmax out", hostArray, out_channel, 1, 1);

	// 메모리 해제
	checkCUDNN(cudnnDestroyTensorDescriptor(inTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(outTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(biasTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(poolOutTensorDeesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(sftTensorDesc));

	checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
	checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc2));
	checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
	checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
	checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc2));
	checkCUDNN(cudnnDestroyActivationDescriptor(actDesc));
	checkCUDNN(cudnnDestroy(cudnnHandle));

	checkCUDA(cudaFree(inData_d));
	checkCUDA(cudaFree(outData_d));
	checkCUDA(cudaFree(filterData_d));
	checkCUDA(cudaThreadSynchronize());
}

int main()
{
	convNCHW();
	//convNHWC();
	return 0;
}

