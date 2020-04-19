#include "maths_matrix.h"
#include <iostream>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>  

using namespace std;
using namespace cv;


// ��ͼƬ�Ծ������ʽ��ʾ���������ڲ鿴ͼƬÿһ���ص�ֵ��
void show_image_64FC1_as_matrix_double(const Mat &img)
{
	int row, col;
	row = img.rows;
	col = img.cols;

	//Ϊ��ָ�����ռ� 
	double **arr = new double *[row];
	for (int i = 0; i < row; i++)
		arr[i] = new double[col];//Ϊÿ�з���ռ䣨ÿ������col��Ԫ�أ� 

	for (int i = 0; i < row; i++)
	{
		const double* pData = img.ptr<double>(i);	//��i+1�е�����Ԫ��
		for (int j = 0; j < col; j++)
		{
			arr[i][j] = pData[j];
		}
	}

	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{

			cout.setf(ios::left); // ���ö��뷽ʽ
			cout.width(8); //����������
			cout.fill('0'); //������Ŀո���0���
			cout << arr[i][j] << ' ';
		}
		cout << endl;
	}

	delete[] arr;
}
