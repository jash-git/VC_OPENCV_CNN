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

#define RELU_USE   //�Ƿ�ʹ��relu�����������ʹ��sigmoid�����
#define A1 18.0
#define PI 3.1415926

#define full 0
#define same 1
#define valid 2

using namespace cv;
using namespace std;


// �����
typedef struct convolutional_layer
{
	int inputWidth;   //����ͼ��Ŀ�
	int inputHeight;  //����ͼ��ĳ�
	int mapSize;      //����ģ��Ĵ�С��ģ��һ�㶼��������

	int inChannels;   //����ͼ�����Ŀ
	int outChannels;  //���ͼ�����Ŀ

	// ��������ģ���Ȩ�طֲ���������һ����ά����
	// ���СΪinChannels*outChannels*mapSize*mapSize��С
	// ��������ά���飬��Ҫ��Ϊ�˱���ȫ���ӵ���ʽ��ʵ���Ͼ���㲢û���õ�ȫ���ӵ���ʽ
	// �����������DeapLearningToolboox���CNN���ӣ����õ�����ȫ����
	//float **** mapData;     //�������ģ�������
	//float **** dmapData;    //�������ģ������ݵľֲ��ݶ�
	vector<vector<Mat>> mapData; //�������ģ������ݣ���άfloat����
	//vector<vector<Mat>> dmapData;  //�������ģ������ݵľֲ��ݶȣ���άfloat����

	//float* basicData;   //ƫ�ã�ƫ�õĴ�С��ΪoutChannels
	Mat basicData; //ƫ�ã�ƫ�õĴ�С��ΪoutChannels�� һάfloat����

	bool isFullConnect; //�Ƿ�Ϊȫ����
	//bool *connectModel; //����ģʽ��Ĭ��Ϊȫ���ӣ�

	// �������ߵĴ�Сͬ�����ά����ͬ
	//float*** v; // ���뼤���������ֵ
	//float*** y; // ���������Ԫ�����
	vector<Mat> v;   //���뼤���������ֵ,��ά����float��
	vector<Mat> y;   //���������Ԫ���������ά����float��

	// ������صľֲ��ݶ�
	//float*** d; // ����ľֲ��ݶ�,��ֵ 
	vector<Mat> d;     // ����ľֲ��ݶ�,��ֵ����ά����float��

}CovLayer;



// �ػ���pooling
typedef struct pooling_layer
{
	int inputWidth;   //����ͼ��Ŀ�
	int inputHeight;  //����ͼ��ĳ�
	int mapSize;      //����ģ��Ĵ�С

	int inChannels;   //����ͼ�����Ŀ
	int outChannels;  //���ͼ�����Ŀ

	int poolType;     //Pooling�ķ���
	//float* basicData;   //ƫ��
	Mat basicData;    //ƫ��, һάfloat����

	//float*** y; // ������������Ԫ�����,�޼����
	//float*** d; // ����ľֲ��ݶ�,��ֵ
	//int*** max_position; // ���ֵģʽ�����ֵ��λ��

	vector<Mat> y;   //������������Ԫ�����,�޼��������ά����float��
	vector<Mat> d;   //����ľֲ��ݶ�,��ֵ����ά����float��
	vector<Mat> max_position;   // ���ֵģʽ�����ֵ��λ�ã���ά����float��

}PoolLayer;



// ����� ȫ���ӵ�������
typedef struct nn_layer
{
	int inputNum;   //�������ݵ���Ŀ
	int outputNum;  //������ݵ���Ŀ

	//float** wData;       // Ȩ�����ݣ�Ϊһ��inputNum*outputNum��С
	//float* basicData;   //ƫ�ã���СΪoutputNum��С
	Mat wData;            // Ȩ�����ݣ�Ϊһ��inputNum*outputNum��С
	Mat basicData;        //ƫ�ã���СΪoutputNum��С

	// �������ߵĴ�Сͬ�����ά����ͬ
	//float* v; // ���뼤���������ֵ
	//float* y; // ���������Ԫ�����
	//float* d; // ����ľֲ��ݶ�,��ֵ
	Mat v;     // ���뼤���������ֵ
	Mat y;     // ���������Ԫ�����
	Mat d;     // ����ľֲ��ݶ�,��ֵ

	bool isFullConnect; //�Ƿ�Ϊȫ����
}OutLayer;


typedef struct cnn_network
{
	int layerNum;
	CovLayer C1;
	PoolLayer S2;
	CovLayer C3;
	PoolLayer S4;
	OutLayer O5;

	Mat e;   // ѵ�����
	Mat L;   // ˲ʱ�������

	/*CovLayer* C1;
	PoolLayer* S2;
	CovLayer* C3;
	PoolLayer* S4;
	OutLayer* O5;

	float* e; // ѵ�����
	float* L; // ˲ʱ�������*/
}CNN;


typedef struct train_opts
{
	int numepochs; // ѵ���ĵ�������
	float alpha; // ѧϰ����
}CNNOpts;





void CNN_test(void);

void cnnsetup(CNN &cnn, int inputSize_r, int inputSize_c, int outputSize);

void cnntrain(CNN &cnn, vector<Mat> inputData, vector<Mat> outputData, CNNOpts opts, int trainNum);

float cnntest(CNN cnn, vector<Mat> inputData, vector<Mat> outputData);


vector<Mat> read_Img_to_Mat(const char* filename);


vector<Mat> read_Lable_to_Mat(const char* filename);


#endif