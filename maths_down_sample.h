#pragma once
#include <vector>
#include "vector_array.h"
#include <iostream>
#include "Array.h"

using namespace std;


// ����ÿ���ػ����͵Ķ�����Կ����
// https://blog.csdn.net/yangqingse/article/details/79841889
typedef enum
{
	MeanPooling = 0,
	MaxPooling,
	StochasticPooling
}down_sample_type;

Array3Dd down_sample(const Array3Dd &vector_array, const int sample_num, down_sample_type sample_type);

Array3Dd down_sample_mean_pooling(const Array3Dd &vector_array, const int sample_num);

Array3Dd down_sample_max_pooling(const Array3Dd &vector_array, const int sample_num);

Array3Dd down_sample_stochastic_pooling(const Array3Dd &vector_array, const int sample_num);

Array3Dd up_sample(const Array3Dd &vector_array3D, const int sample_num, down_sample_type sample_type);

Array3Dd up_sample_mean_pooling(const Array3Dd &vector_array3D, const int sample_num);

