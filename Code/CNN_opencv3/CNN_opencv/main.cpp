#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <random>
#include <ctime>
#include <windows.h>    //微秒级计时相关函数
#include "CNN.h"
#include "minst.h"

using namespace cv;
using namespace std;


void write_data_to_mat(float **array, Mat &mat)
{
	for (int i = 0; i < mat.rows; i++)
	{
		float *p = mat.ptr<float>(i);
		for (int j = 0; j < mat.cols; j++)
		{
			p[j] = array[i][j];
		}
	}
}


void minst_cnn_test(void)
{
#if 0
	LabelArr trainLabel = read_Lable("Minst/train-labels.idx1-ubyte");
	ImgArr trainImg = read_Img("Minst/train-images.idx3-ubyte");
	LabelArr testLabel = read_Lable("Minst/t10k-labels.idx1-ubyte");
	ImgArr testImg = read_Img("Minst/t10k-images.idx3-ubyte");

	vector<Mat> traindata_list;
	vector<Mat> traindata_label;
	vector<Mat> testdata_list;
	vector<Mat> testdata_label;

	int train_num = trainImg->ImgNum;
	int test_num = testImg->ImgNum;
	int outSize = testLabel->LabelPtr[0].l;

	int row = trainImg->ImgPtr[0].r;
	int col = trainImg->ImgPtr[0].c;

	printf("row=%d, col=%d, outSize=%d\n", row, col, outSize);

	Mat tmp(row, col, CV_32FC1);   //二维数组转换为Mat
	for (int i = 0; i < train_num; i++)
	{
		write_data_to_mat(trainImg->ImgPtr[i].ImgData, tmp);
		Mat labeltmp(1, outSize, CV_32FC1, trainLabel->LabelPtr[i].LabelData);

		traindata_list.push_back(tmp.clone());
		traindata_label.push_back(labeltmp.clone());
	}

	for (int i = 0; i < test_num; i++)
	{
		write_data_to_mat(testImg->ImgPtr[i].ImgData, tmp);
		Mat labeltmp(1, outSize, CV_32FC1, testLabel->LabelPtr[i].LabelData);

		testdata_list.push_back(tmp.clone());
		testdata_label.push_back(labeltmp.clone());
	}
#else

	vector<Mat> traindata_list;
	vector<Mat> traindata_label;
	vector<Mat> testdata_list;
	vector<Mat> testdata_label;

	traindata_label = read_Lable_to_Mat("Minst/train-labels.idx1-ubyte");
	traindata_list = read_Img_to_Mat("Minst/train-images.idx3-ubyte");
	testdata_label = read_Lable_to_Mat("Minst/t10k-labels.idx1-ubyte");
	testdata_list = read_Img_to_Mat("Minst/t10k-images.idx3-ubyte");
	
	int train_num = traindata_list.size();
	int test_num = testdata_list.size();
	int outSize = testdata_label[0].cols;

	int row = traindata_list[0].rows;
	int col = traindata_list[0].cols;


#endif

	CNNOpts opts;
	opts.numepochs = 1;
	opts.alpha = 0.01;
	int trainNum = 60000;

	CNN cnn;
	cnnsetup(cnn, row, col, outSize);   //cnn初始化
	cnntrain(cnn, traindata_list, traindata_label, opts, train_num);

	float success = cnntest(cnn, testdata_list, testdata_label);
	printf("success=%f\n", 1 - success);
}



int main(void)
{

	//CNN_test();
	minst_cnn_test();

	return 0;
}