#ifndef _CNN_H_
#define _CNN_H_

#include <opencv2\opencv.hpp>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>

#define AvePool 0
#define MaxPool 1
#define MinPool 2

#define RELU_USE   //是否使用relu激活函数，否则使用sigmoid激活函数
#define A1 18.0
#define PI 3.1415926

#define full 0
#define same 1
#define valid 2

using namespace cv;
using namespace std;


// 卷积层
typedef struct convolutional_layer
{
	int inputWidth;   //输入图像的宽
	int inputHeight;  //输入图像的长
	int mapSize;      //特征模板的大小，模板一般都是正方形

	int inChannels;   //输入图像的数目
	int outChannels;  //输出图像的数目

	// 关于特征模板的权重分布，这里是一个四维数组
	// 其大小为inChannels*outChannels*mapSize*mapSize大小
	// 这里用四维数组，主要是为了表现全连接的形式，实际上卷积层并没有用到全连接的形式
	// 这里的例子是DeapLearningToolboox里的CNN例子，其用到就是全连接
	//float **** mapData;     //存放特征模块的数据
	//float **** dmapData;    //存放特征模块的数据的局部梯度
	vector<vector<Mat>> mapData; //存放特征模块的数据，四维float数组
	//vector<vector<Mat>> dmapData;  //存放特征模块的数据的局部梯度，四维float数组

	//float* basicData;   //偏置，偏置的大小，为outChannels
	Mat basicData; //偏置，偏置的大小，为outChannels， 一维float数组

	bool isFullConnect; //是否为全连接
	//bool *connectModel; //连接模式（默认为全连接）

	// 下面三者的大小同输出的维度相同
	//float*** v; // 进入激活函数的输入值
	//float*** y; // 激活函数后神经元的输出
	vector<Mat> v;   //进入激活函数的输入值,三维数组float型
	vector<Mat> y;   //激活函数后神经元的输出，三维数组float型

	// 输出像素的局部梯度
	//float*** d; // 网络的局部梯度,δ值 
	vector<Mat> d;     // 网络的局部梯度,δ值，三维数组float型

}CovLayer;



// 池化层pooling
typedef struct pooling_layer
{
	int inputWidth;   //输入图像的宽
	int inputHeight;  //输入图像的长
	int mapSize;      //特征模板的大小

	int inChannels;   //输入图像的数目
	int outChannels;  //输出图像的数目

	int poolType;     //Pooling的方法
	//float* basicData;   //偏置
	Mat basicData;    //偏置, 一维float数组

	//float*** y; // 采样函数后神经元的输出,无激活函数
	//float*** d; // 网络的局部梯度,δ值
	//int*** max_position; // 最大值模式下最大值的位置

	vector<Mat> y;   //采样函数后神经元的输出,无激活函数，三维数组float型
	vector<Mat> d;   //网络的局部梯度,δ值，三维数组float型
	vector<Mat> max_position;   // 最大值模式下最大值的位置，三维数组float型

}PoolLayer;



// 输出层 全连接的神经网络
typedef struct nn_layer
{
	int inputNum;   //输入数据的数目
	int outputNum;  //输出数据的数目

	//float** wData;       // 权重数据，为一个inputNum*outputNum大小
	//float* basicData;   //偏置，大小为outputNum大小
	Mat wData;            // 权重数据，为一个inputNum*outputNum大小
	Mat basicData;        //偏置，大小为outputNum大小

	// 下面三者的大小同输出的维度相同
	//float* v; // 进入激活函数的输入值
	//float* y; // 激活函数后神经元的输出
	//float* d; // 网络的局部梯度,δ值
	Mat v;     // 进入激活函数的输入值
	Mat y;     // 激活函数后神经元的输出
	Mat d;     // 网络的局部梯度,δ值

	bool isFullConnect; //是否为全连接
}OutLayer;


typedef struct cnn_network
{
	int layerNum;
	CovLayer C1;
	PoolLayer S2;
	CovLayer C3;
	PoolLayer S4;
	OutLayer O5;

	Mat e;   // 训练误差
	Mat L;   // 瞬时误差能量

	/*CovLayer* C1;
	PoolLayer* S2;
	CovLayer* C3;
	PoolLayer* S4;
	OutLayer* O5;

	float* e; // 训练误差
	float* L; // 瞬时误差能量*/
}CNN;


typedef struct train_opts
{
	int numepochs; // 训练的迭代次数
	float alpha; // 学习速率
}CNNOpts;





void CNN_test(void);

void cnnsetup(CNN &cnn, int inputSize_r, int inputSize_c, int outputSize);

void cnntrain(CNN &cnn, vector<Mat> inputData, vector<Mat> outputData, CNNOpts opts, int trainNum);

float cnntest(CNN cnn, vector<Mat> inputData, vector<Mat> outputData);


vector<Mat> read_Img_to_Mat(const char* filename);


vector<Mat> read_Lable_to_Mat(const char* filename);


#endif