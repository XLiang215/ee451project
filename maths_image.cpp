#include <iostream>
#include "maths.h"
#include "maths_image.h"


using namespace cv;

// �˺�����array��ά���鲻����new�����ģ�������Ϊ�ڴ��ַ�����������Ի�Խ�硣
Mat matrix_double_to_Mat_64FC1(double *array, int row, int col)
{
	Mat img(row, col, CV_64FC1);
	double *ptmp = NULL;
	for (int i = 0; i < row; i++)
	{
		ptmp = img.ptr<double>(i);

		for (int j = 0; j < col; ++j)
		{
			ptmp[j] = *array++;
		}
	}
	return img;
}


Mat vector_vector_double_to_Mat_64FC1(const vector<vector<double> > &array)
{
	int row = array.at(0).size();
	int col = array.size();

	Mat img(row, col, CV_64FC1);
	double *ptmp = NULL;
	for (int i = 0; i < row; i++)
	{
		ptmp = img.ptr<double>(i);

		for (int j = 0; j < col; ++j)
		{
			ptmp[j] = array.at(j).at(i);
		}
	}
	return img;
}


vector<Mat> vector_array2D_to_vector_Mat_64FC1(const vector<array2D> &vector_array)
{
	int size = vector_array.size();

	vector<Mat> vector_Mat_ret(size);

	for (int i = 0; i < size; i++)
	{
		vector_Mat_ret.at(i) = vector_vector_double_to_Mat_64FC1(vector_array.at(i));
	}
	
	return vector_Mat_ret;
}


// ��ͼƬ����ʽ�Ѿ�����ʾ����
// ���ô˺���ʱ����һ����������д��array[0],��������array
void show_matrix_double_as_image_64FC1(double *array, int row, int col, int time_msec)
{
	Mat image = matrix_double_to_Mat_64FC1(array, row, col);

	// ��ʾͼƬ   
	imshow("ͼƬ", image);
	// �ȴ�time_msec�󴰿��Զ��ر�    
	waitKey(time_msec);
}


// ��ͼƬ����ʽ��vector������ʾ����
void show_vector_vector_double_as_image_64FC1(const vector<vector<double> > &array, int time_msec)
{
	Mat image = vector_vector_double_to_Mat_64FC1(array);

	// ��ʾͼƬ   
	imshow("ͼƬ", image);
	// �ȴ�time_msec�󴰿��Զ��ر�    
	waitKey(time_msec);

	destroyWindow("ͼƬ");
}


// ��ָ���ļ�����������ȡͼƬ
void read_batch_images(string file_addr, string image_suffix, int begin_num, int end_num, vector<Mat> &data_set)
{
    for (int repe = 0; repe < 600; repe ++) {
        if (repe % 100 == 99)
            cout<<"reading samples "<<(repe + 1) * 100<<" / 60000"<<endl;
        for (int i = begin_num; i <= end_num; i++) {
            stringstream ss; // intתstring
            string image_name;
            ss << i;
            ss >> image_name;
            image_name = image_name + "." + image_suffix;
            string image_addr_name = file_addr + "/" + image_name;
            //		cout << "reading " << image_name << " from " << image_addr_name << endl;

            // ��ȡ�Ҷ�ͼ
            Mat image = imread(image_addr_name, 0);

            data_set.push_back(image);

            if (image.data == NULL) {
                cout << "[warning: no image!]" << endl;
            }
        }
    }
}


void images_convert_to_64FC1(vector<Mat> &data_set)
{
	vector<Mat>::iterator it;
	for (it = data_set.begin(); it != data_set.end(); it++)
	{
		(*it).convertTo(*it, CV_64FC1, 1 / 255.0);//����dstΪĿ��ͼ�� CV_64FC1ΪҪת��������
	}
}


/*
	scale�������ڵ������ͼ������ռ�����������ı���
*/
void show_curve_image(vector<double>data_x, vector<double>data_y, float scale, int msec)
{
	// https://blog.csdn.net/hu_guan_jie/article/details/50987520

	if (data_x.size() != data_y.size())
	{
		cout << "data_x size is not same as data_y size in show_curve_image()!" << endl;
		return;
	}

	int point_num = data_x.size();

	Mat img = Mat::zeros(800, 800, CV_8UC3);//��������

	vector<Point> curvePoint;//���ڱ���point��vector
	Point tmpPoint;

	for (int i = 0; i < point_num; ++i)
	{
		tmpPoint = cvPoint((int)data_x.at(i)*scale, (int)data_y.at(i)*scale);;
		curvePoint.push_back(tmpPoint);
	}

	vector<Point>::iterator it;
	it = curvePoint.begin();

	Point pointPre = cvPoint(curvePoint.at(1).x, 800 - curvePoint.at(1).y);//��ʼ��
	while (it != curvePoint.end())
	{
		Point pointTmp = (*it);
		pointTmp = cvPoint(pointTmp.x, 800 - pointTmp.y);//���귭ת
		line(img, pointPre, pointTmp, cvScalarAll(255), 4);
		pointPre = pointTmp;

		it++;
	}

	imshow("curve", img);
	waitKey(msec);
}


