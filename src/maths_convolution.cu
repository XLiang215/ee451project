#include "maths_convolution.h"
#include <cublas.h>
#include<cuda_runtime_api.h>
#include<cuda.h>

__global__ void convolution(float *I, const float* __restrict__ M, float *P,int channels, int width, int height, int TILE_WIDTH, int maskLength)
{
  __shared__ float N_ds[7][7];
  int k;
  int w_l = TILE_WIDTH + maskLength - 1;
  for (k = 0; k < channels; k++) {					// First batch loading
    int dest = threadIdx.y * TILE_WIDTH + threadIdx.x,
       destY = dest / w_l, destX = dest % w_l,
       srcY = blockIdx.y * TILE_WIDTH + destY - maskLength/2,
       srcX = blockIdx.x * TILE_WIDTH + destX - maskLength/2,
       src = (srcY * width + srcX) * channels + k;
    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
       N_ds[destY][destX] = I[src];
    else
       N_ds[destY][destX] = 0.0;

       for (int iter=1; iter <= (w_l * w_l) / (TILE_WIDTH*TILE_WIDTH); iter++)
      {					// Second batch loading
        dest = threadIdx.y * TILE_WIDTH + threadIdx.x + iter*(TILE_WIDTH * TILE_WIDTH);
          destY = dest / w_l, destX = dest % w_l;
          srcY  = blockIdx.y * TILE_WIDTH + destY - maskLength/2;
          srcX = blockIdx.x * TILE_WIDTH + destX - maskLength/2;
          src = (srcY * width + srcX) * channels + k;
          if (destY < w_l && destX < w_l)
          {
              if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
                  N_ds[destY][destX] = I[src];
              else
                  N_ds[destY][destX] = 0.0;
          }
      }
    __syncthreads();

    float accum = 0;
    int y, x;
    for (y = 0; y < maskLength; y++)
       for (x = 0; x < maskLength; x++)
          accum += N_ds[threadIdx.y + y][threadIdx.x + x] * M[y * maskLength + x];
    y = blockIdx.y * TILE_WIDTH + threadIdx.y;
    x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    if ((y >= (maskLength - 1) / 2) && (x >= (maskLength - 1) / 2) && (y < height - (maskLength - 1) / 2) && (x < width - (maskLength - 1) / 2))
      P[((y-(maskLength - 1) / 2) * maskLength + x-(maskLength - 1) / 2) * channels + k] = accum;
    __syncthreads();
  }
}


Array3Dd convolution(Array3Dd X, const Array2Dd &Ker, string shape) // 采用数组来求卷积，而不是用vector，速度要快10%！
{

	if (shape != "valid" && shape != "full")
	{
		cout << "wrong convolution shape control!" << endl << "convolution() failed!" << endl;
		Array3Dd temp;
		return temp;
	}

	if (X.size() <= 0)
	{
		cout << "Array3Dd is wrong!" << endl << "convolution() failed!" << endl;
		Array3Dd temp;
		return temp;
	}

	int Ker_row = Ker.at(0).size();
	int Ker_col = Ker.size();

	if (shape == "full")
	{
		X.expand_to_full_size(Ker_col, Ker_row);
	}

	int X_page = X.size();
	int X_row = X.at(0).at(0).size();
	int X_col = X.at(0).size();

	int i, j, k;

	if (shape == "valid" && (X_row < Ker_row || X_col < Ker_col))
	{
		cout << "X size is smaller than Ker size!" << endl << "convolution() failed!" << endl;
		Array3Dd temp;
		return temp;
	}

    
	int conv_row = X.at(0).at(0).size() - Ker.at(0).size() + 1; // 创建卷积结果输出变量conv并初始化为0
	int conv_col = X.at(0).size() - Ker.size() + 1;
	Array3Dd convn(X_page, conv_col, conv_row, 0);

	double *arr_X = new double[X_page * X_row * X_col]();
	double *arr_Ker = new double[Ker_row * Ker_col]();

	
	for (i = 0; i < X_page; i++) //vector 转 数组
	{
		for (j = 0; j < X_row; j++)
		{
			for (k = 0; k < X_col; k++)
			{
				
				arr_X[i * (X_row * X_col) + j * X_col + k] = X.at(i).at(k).at(j); // 对arr_X赋值

				
				if ((i == 0) && (j < Ker_row) && (k < Ker_col)) // 对arr_Ker赋值
				{
					arr_Ker[j * Ker_col + k] = Ker.at(Ker_col - 1 - k).at(Ker_row - 1 - j);// x,y向同时翻转
				}
			}
		}
	}


	int maskLength = 5;

    int imageChannels = X_page;
    int imageWidth = X_row;
    int imageHeight = X_col;

    int TILE_WIDTH = 3;

	float * hostOutputImageData;

	float * deviceInputImageData;
	float * deviceOutputImageData;
	float * deviceMaskData;

	hostOutputImageData = (float *) malloc(sizeof(float)*(imageWidth - maskLength + 1)*(imageHeight - maskLength + 1)*imageChannels);
		
	cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **) &deviceOutputImageData, (imageWidth - maskLength + 1) * (imageHeight - maskLength + 1) * imageChannels * sizeof(float));
	cudaMalloc((void **) &deviceMaskData, maskLength * maskLength * sizeof(float));

	cudaMemcpy(deviceInputImageData, //copy image to device
		     arr_X,
		     imageWidth * imageHeight * imageChannels * sizeof(float),
		     cudaMemcpyHostToDevice);
	cudaMemcpy(deviceMaskData,       //copy mask to device
		     arr_Ker,
		     maskLength * maskLength * sizeof(float),
		     cudaMemcpyHostToDevice);

    
    dim3 dimGrid(((imageWidth-1)/TILE_WIDTH)+1, ((imageHeight-1)/TILE_WIDTH)+1,1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	convolution<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceMaskData, deviceOutputImageData, 
										imageChannels, imageWidth, imageHeight, TILE_WIDTH, maskLength);
    
	
    cudaMemcpy(hostOutputImageData, //copy result to host
	         deviceOutputImageData,
	         (imageWidth - maskLength + 1) * (imageHeight - maskLength + 1) * imageChannels * imageChannels * sizeof(float),
	         cudaMemcpyDeviceToHost);

    int id = 0;
    for (i = 0; i < X_page; i++) {
		for (j = 0; j < conv_row; j++) {
			for (k = 0; k < conv_col; k++) {
				convn.at(i).at(k).at(j) = hostOutputImageData[id++];
			}
		}
	}

	delete[] arr_X;
	delete[] arr_Ker;
	free(hostOutputImageData);
	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);
	cudaFree(deviceMaskData);

	return convn;
}



Array2Dd convolution(const Array3Dd &X, const Array3Dd &Ker, string shape) 
{

	if (shape != "valid" && shape != "full")
	{
		cout << "wrong convolution shape control!" << endl << "convolution() failed!" << endl;
		Array2Dd temp;
		return temp;
	}

	int page_X = X.size();
	int page_Ker = Ker.size();

	if (page_X != page_Ker)
	{
		cout << "page size not equal!" << endl << "convolution() failed!" << endl;
		Array2Dd temp;
		return temp;
	}

	Array2Dd sum;

	for (int i = 0; i < page_X; ++i)
	{
		sum.add(convolution(X.at(i), Ker.at(i), shape));
	}

	return sum;
}


Array2Dd convolution(Array2Dd X, Array2Dd Ker, string shape) // 采用数组来求卷积，而不是用vector，速度要快30倍！
{
	if (shape != "valid" && shape != "full")
	{
		cout << "wrong convolution shape control!" << endl << "convolution() failed!" << endl;
		Array2Dd temp;
		return temp;
	}

	int Ker_row = Ker.at(0).size();
	int Ker_col = Ker.size();

	if (shape == "full")
	{
		X.expand_to_full_size(Ker_col, Ker_row);
	}

	int X_row = X.at(0).size();
	int X_col = X.size();

	if (shape == "valid" && (X_row < Ker_row || X_col < Ker_col))
	{
		cout << "X size is smaller than Ker size!" << endl << "convolution() failed!" << endl;
		Array2Dd temp;
		return temp;
	}

   
	int conv_row = X.at(0).size() - Ker.at(0).size() + 1;  // 创建卷积结果输出变量conv并初始化为0
	int conv_col = X.size() - Ker.size() + 1;
	Array2Dd conv(conv_col, conv_row, 0);

	double *arr_X = new double[X_row * X_col]();
	double *arr_Ker = new double[Ker_row * Ker_col]();

	int i, j;

	for (i = 0; i < X_row; i++)
	{
		for (j = 0; j < X_col; j++)
		{	
			arr_X[i * X_col + j] = X.at(j).at(i); // 对arr_X赋值  
			if ((i < Ker_row) && (j < Ker_col)) // 对arr_Ker赋值
			{
				arr_Ker[i * Ker_col + j] = Ker.at(Ker_col - 1 - j).at(Ker_row - 1 - i); // x,y向同时翻转
			}
		}
	}

	int row, col;
	for (i = 0; i < conv_row; i++)
	{
		for (j = 0; j < conv_col; j++)
		{
            double sum_ij = 0; // 计算卷积矩阵第(i,j)点的值
			for (row = i; row < i + Ker_row; row++)
			{
				for (col = j; col < j + Ker_col; col++)
				{
					sum_ij += arr_X[row * X_col + col] * arr_Ker[(row - i) * Ker_col + (col - j)];
				}
			}
			conv.at(j).at(i) = sum_ij;
		}
	}

	delete[] arr_X;
	delete[] arr_Ker;

	return conv;
}

