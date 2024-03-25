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


#define randf(a, b) (((rand()%10000+rand()%10000*10000)/100000000.0)*((b)-(a))+(a))


//初始化卷积层
CovLayer initCovLayer(int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels)
{
	CovLayer covL;

	covL.inputHeight = inputHeight;
	covL.inputWidth = inputWidth;
	covL.mapSize = mapSize;

	covL.inChannels = inChannels;
	covL.outChannels = outChannels;

	covL.isFullConnect = true;   // 默认为全连接

	// 权重空间的初始化，先行再列调用，[r][c]
	srand((unsigned)time(NULL));   //设置随机数种子
	for(int i = 0; i < inChannels; i++)   //输入通道数
	{
		vector<Mat> tmp;
		for(int j = 0; j < outChannels; j++)   //输出通道数
		{
			Mat tmpmat(mapSize, mapSize, CV_32FC1);
			for(int r = 0; r < mapSize; r++)   //卷积核的高
			{
				for(int c = 0; c < mapSize; c++)  //卷积核的宽
				{
					float randnum=(((float)rand()/(float)RAND_MAX)-0.5)*2;    //生成-1~1的随机数
					tmpmat.ptr<float>(r)[c] = randnum*sqrt(6.0/(mapSize*mapSize*(inChannels+outChannels)));
					//tmpmat.ptr<float>(r)[c] = randf(-0.05, 0.05);
				}
			}
			tmp.push_back(tmpmat.clone());
		}
		covL.mapData.push_back(tmp);
	}


	/*Mat tmpmat1 = Mat::zeros(mapSize, mapSize, CV_32FC1);
	for(int i = 0; i < inChannels; i++)   //输入通道数
	{
		vector<Mat> tmp;
		for(int j = 0; j < outChannels; j++)   //输出通道数
		{
			tmp.push_back(tmpmat1.clone());
		}
		covL.dmapData.push_back(tmp);
	}*/
 
	covL.basicData = Mat::zeros(1, outChannels, CV_32FC1);   //初始化卷积层偏置的内存

	int outW = inputWidth - mapSize + 1;   //卷积层输出的宽
	int outH = inputHeight - mapSize + 1;  //卷积层输出的高

	Mat tmpmat2 = Mat::zeros(outH, outW, CV_32FC1);
	for(int i = 0; i < outChannels; i++)
	{
		covL.d.push_back(tmpmat2.clone());
		covL.v.push_back(tmpmat2.clone());
		covL.y.push_back(tmpmat2.clone());
	}

	return covL;
}



//池化层初始化
PoolLayer initPoolLayer(int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels, int poolType)
{
	PoolLayer poolL;

	poolL.inputHeight=inputHeight;    //输入高度
	poolL.inputWidth=inputWidth;      //输入宽度
	poolL.mapSize=mapSize;            //卷积核尺寸，池化层相当于做一个特殊的卷积操作
	poolL.inChannels=inChannels;      //输入通道
	poolL.outChannels=outChannels;    //输出通道
	poolL.poolType=poolType;          //最大值模式/平均值模式

	poolL.basicData = Mat::zeros(1, outChannels, CV_32FC1);    //池化层无偏置，无激活，所以这里申请内存只是预留

	int outW = inputWidth/mapSize;   //池化层的卷积核为2*2
	int outH = inputHeight/mapSize;

	Mat tmpmat = Mat::zeros(outH, outW, CV_32FC1);
	Mat tmpmat1 = Mat::zeros(outH, outW, CV_32SC1);
	for(int i = 0; i < outChannels; i++)
	{
		poolL.d.push_back(tmpmat.clone());   //局域梯度
		poolL.y.push_back(tmpmat.clone());   //采样函数后神经元输出，无激活函数
		poolL.max_position.push_back(tmpmat1.clone());   //最大值模式下最大值的位置
	}

	return poolL;
}


//输出层初始化
OutLayer initOutLayer(int inputNum, int outputNum)
{
	OutLayer outL;

	outL.inputNum = inputNum;
	outL.outputNum = outputNum;
	outL.isFullConnect = true;

	outL.basicData = Mat::zeros(1, outputNum, CV_32FC1);    //偏置,分配内存的同时初始化为0
	outL.d = Mat::zeros(1, outputNum, CV_32FC1);
	outL.v = Mat::zeros(1, outputNum, CV_32FC1);
	outL.y = Mat::zeros(1, outputNum, CV_32FC1);

	// 权重的初始化
	outL.wData = Mat::zeros(outputNum, inputNum, CV_32FC1);   // 输出行，输入列,权重为10*192矩阵
	srand((unsigned)time(NULL));
	for(int i = 0; i < outputNum; i++)
	{
		float *p = outL.wData.ptr<float>(i);
		for(int j = 0; j < inputNum; j++)
		{
			float randnum = (((float)rand()/(float)RAND_MAX)-0.5)*2; // 产生一个-1到1的随机数,rand()的取值范围为0~RAND_MAX
			p[j] = randnum*sqrt(6.0/(inputNum+outputNum));
			//p[j] = randf(-0.05, 0.05);
		}
	}

	return outL;
}


//1行n列的向量
int vecmaxIndex(Mat vec)  //返回向量最大数的序号
{
	int veclength = vec.cols;
	float maxnum = -1.0;
	int maxIndex = 0;

	float *p = vec.ptr<float>(0);
	for(int i=0; i < veclength; i++)
	{
		if(maxnum < p[i])
		{
			maxnum = p[i];
			maxIndex = i;
		}
	}
	return maxIndex;
}

// 矩阵翻转180度
Mat rotate180(Mat mat)
{
#if 0
	int outSizeW = mat.cols;   //宽
	int outSizeH = mat.rows;   //高

	Mat tmp(mat.size(), CV_32FC1);
	for(int i = 0; i < outSizeH; i++)
	{
		for(int j = 0; j < outSizeW; j++)
		{
			tmp.ptr<float>(i)[j] = mat.ptr<float>(outSizeH-i-1)[outSizeW-j-1];
		}
	}

	return tmp;
#else
	Mat tmp;
	flip(mat, tmp, -1);// 翻转模式，flipCode == 0垂直翻转（沿X轴翻转），flipCode>0水平翻转（沿Y轴翻转），flipCode<0水平垂直翻转（先沿X轴翻转，再沿Y轴翻转，等价于旋转180°）
	return tmp;
#endif
}



#if 1

// 关于卷积和相关操作的输出选项
// 这里共有三种选择：full、same、valid，分别表示
// full指完全，操作后结果的大小为inSize+(mapSize-1)
// same指同输入相同大小
// valid指完全操作后的大小，一般为inSize-(mapSize-1)大小，其不需要将输入添0扩大。

Mat correlation(Mat map, Mat inputData, int type) // 互相关
{
	// 这里的互相关是在后向传播时调用，类似于将Map反转180度再卷积
	// 为了方便计算，这里先将图像扩充一圈
	// 这里的卷积要分成两拨，偶数模板同奇数模板

	const int map_row = map.rows;
	const int map_col = map.cols;
	const int map_row_2 = map.rows/2;
	const int map_col_2 = map.cols/2;
	const int in_row = inputData.rows;
	const int in_col = inputData.cols;

	Mat exInputData;
	copyMakeBorder(inputData, exInputData, map_row_2, map_row_2, map_col_2, map_col_2, BORDER_CONSTANT, 0);
	Mat OutputData;
	filter2D(exInputData, OutputData, exInputData.depth(), map);


	//full模式
	if(type == full)
	{
		return OutputData;
	}
	else if(type == valid)
	{
		int out_row = in_row - (map_row - 1);
		int out_col = in_col - (map_col - 1);
		Mat outtmp;
		OutputData(Rect(2*map_col_2, 2*map_row_2, out_col, out_row)).copyTo(outtmp);
		return outtmp;
	}
	else
	{
		Mat outtmp;
		OutputData(Rect(map_col_2, map_row_2, in_col, in_row)).copyTo(outtmp);
		return outtmp;
	}
	
}
#else

// 给二维矩阵边缘扩大，增加addw大小的0值边
Mat matEdgeExpand(Mat mat, int addc, int addr)
{ // 向量边缘扩大
	int i, j;
	int c = mat.cols;
	int r = mat.rows;

	//float** res = (float**)malloc((r + 2 * addr) * sizeof(float*)); // 结果的初始化
	//for (i = 0; i<(r + 2 * addr); i++)
	//	res[i] = (float*)malloc((c + 2 * addc) * sizeof(float));

	Mat res(r + 2 * addr, c + 2 * addc, CV_32FC1);

	for (j = 0; j < r + 2 * addr; j++) 
	{
		float *p = res.ptr<float>(j);
		float *mat_p = mat.ptr<float>(j - addr);
		for (i = 0; i < c + 2 * addc; i++) 
		{
			if (j < addr || i < addc || j >= (r + addr) || i >= (c + addc))
				//res[j][i] = (float)0.0;
				p[i] = 0.0;
			else
				//res[j][i] = mat[j - addr][i - addc]; // 复制原向量的数据
				p[i] = mat_p[i - addc];

		}
	}
	return res;
}


// 给二维矩阵边缘缩小，擦除shrinkc大小的边
Mat matEdgeShrink(Mat mat, int shrinkc, int shrinkr)
{ // 向量的缩小，宽缩小addw，高缩小addh
	int i, j;
	int c = mat.cols;
	int r = mat.rows;
	//float** res = (float**)malloc((r - 2 * shrinkr) * sizeof(float*)); // 结果矩阵的初始化
	//for (i = 0; i<(r - 2 * shrinkr); i++)
	//	res[i] = (float*)malloc((c - 2 * shrinkc) * sizeof(float));

	Mat res(r - 2 * shrinkr, c - 2 * shrinkc, CV_32FC1);

	for (j = 0; j < r; j++) 
	{
		float *p = res.ptr<float>(j - shrinkr);
		float *mat_p = mat.ptr<float>(j);
		for (i = 0; i < c; i++) 
		{
			if (j >= shrinkr && i >= shrinkc && j < (r - shrinkr) && i < (c - shrinkc))
				//res[j - shrinkr][i - shrinkc] = mat[j][i]; // 复制原向量的数据
				p[i - shrinkc] = mat_p[i];
		}
	}
	return res;
}


Mat correlation(Mat map, Mat inputData, int type)// 互相关
{
	// 这里的互相关是在后向传播时调用，类似于将Map反转180度再卷积
	// 为了方便计算，这里先将图像扩充一圈
	// 这里的卷积要分成两拨，偶数模板同奇数模板
	int i, j, c, r;
	int halfmapsizew;
	int halfmapsizeh;
	if (map.rows% 2 == 0 && map.cols%2 == 0)
	{ // 模板大小为偶数
		halfmapsizew = map.cols / 2; // 卷积模块的半瓣大小
		halfmapsizeh = map.rows / 2;
	}
	else
	{
		halfmapsizew = (map.cols - 1) / 2; // 卷积模块的半瓣大小
		halfmapsizeh = (map.rows - 1) / 2;
	}

	// 这里先默认进行full模式的操作，full模式的输出大小为inSize+(mapSize-1)
	int outSizeW = inputData.cols + (map.cols - 1); // 这里的输出扩大一部分
	int outSizeH = inputData.rows + (map.rows - 1);
	Mat outputData = Mat::zeros(outSizeH, outSizeW, CV_32FC1);
	/*float** outputData = (float**)malloc(outSizeH * sizeof(float*)); // 互相关的结果扩大了
	for (i = 0; i<outSizeH; i++)
	{
		outputData[i] = (float*)calloc(outSizeW, sizeof(float));   //申请二维数组
	}*/

	// 为了方便计算，将inputData扩大一圈
	//float** exInputData = matEdgeExpand(inputData, inSize, mapSize.c - 1, mapSize.r - 1);
	Mat exInputData = matEdgeExpand(inputData, map.cols - 1, map.rows - 1);

	for (j = 0; j < outSizeH; j++)  //默认卷积核的滑动步长为1
	{
		float *outputData_p = outputData.ptr<float>(j);
		
		for (i = 0; i < outSizeW; i++)
		{
			for (r = 0; r < map.rows; r++)
			{
				float *exInputData_p = exInputData.ptr<float>(j + r);
				float *map_p = map.ptr<float>(r);
				for (c = 0; c < map.cols; c++)
				{
					//outputData[j][i] = outputData[j][i] + map[r][c] * exInputData[j + r][i + c];   //计算卷积，即矩阵每个位置的值相乘，结果累加
					outputData_p[i] = outputData_p[i] + map_p[c] * exInputData_p[i + c];
				}
			}
		}
	}

	//for (i = 0; i<inSize.r + 2 * (mapSize.r - 1); i++)   //释放二维数组内存
	//	free(exInputData[i]);
	//free(exInputData);

	//nSize outSize = { outSizeW,outSizeH };
	switch (type)   //根据不同的情况，返回不同的结果
	{
	case full: //完全大小的情况
		return outputData;
	case same:
	{
		//float** sameres = matEdgeShrink(outputData, outSize, halfmapsizew, halfmapsizeh);
		//for (i = 0; i<outSize.r; i++)
		//	free(outputData[i]);
		//free(outputData);
		//return sameres;
		Mat tmp = matEdgeShrink(outputData, halfmapsizew, halfmapsizeh);
		return tmp;
	}
	case valid:
	{
		//float** validres;
		Mat tmp;
		if (map.rows % 2 == 0 && map.cols % 2 == 0)
			//validres = matEdgeShrink(outputData, outSize, halfmapsizew * 2 - 1, halfmapsizeh * 2 - 1);
			tmp = matEdgeShrink(outputData, halfmapsizew * 2 - 1, halfmapsizeh * 2 - 1);
		else
			//validres = matEdgeShrink(outputData, outSize, halfmapsizew * 2, halfmapsizeh * 2);
			tmp = matEdgeShrink(outputData, halfmapsizew * 2, halfmapsizeh * 2);
		//for (i = 0; i<outSize.r; i++)
		//	free(outputData[i]);
		//free(outputData);
		//return validres;
		return tmp;
	}
	default:
		return outputData;
	}
}

#endif



// 卷积操作
Mat cov(Mat map, Mat inputData, int type) 
{
	// 卷积操作可以用旋转180度的特征模板相关来求
	//Mat flipmap = rotate180(map); //旋转180度的特征模板
	Mat flipmap;
	//cout << map << endl;
	flip(map, flipmap, -1);
	//cout << flipmap << endl;
	//while (1);
	//flipmap = rotate180(map);
	Mat res = correlation(flipmap, inputData, type);

	/*Mat X = (Mat_<uchar>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
	cout << X << endl;
	flip(X, X, -1);
	cout << endl;
	cout << X << endl;
	while (1);*/
	
	return res;
}


void avgPooling(Mat input, Mat &output, int mapSize) // 求平均值
{
	const int outputW = input.cols/mapSize;   //输出宽=输入宽/核宽
	const int outputH = input.rows/mapSize;   //输出高=输入高/核高
	float len = (float)(mapSize*mapSize);
	int i,j,m,n;
	for(i = 0;i < outputH; i++)
	{
		for(j = 0; j < outputW; j++)
		{
			float sum=0.0;
			for(m = i*mapSize; m < i*mapSize+mapSize; m++)  //取卷积核大小的窗口求和平均
			{
				for(n = j*mapSize; n < j*mapSize+mapSize; n++)
				{
					sum += input.ptr<float>(m)[n];
				}
			}

			output.ptr<float>(i)[j] = sum/len;
		}
	}
}


//最大值池化
void maxPooling(Mat input, Mat &max_position, Mat &output, int mapSize)
{
	int outputW = input.cols / mapSize;   //输出宽=输入宽/核宽
	int outputH = input.rows / mapSize;   //输出高=输入高/核高

	int i, j, m, n;
	for (i = 0; i < outputH; i++)
	{
		for (j = 0; j < outputW; j++)
		{
			float max = -999999.0;
			int max_index = 0;

			for (m = i*mapSize; m<i*mapSize + mapSize; m++)  //取卷积核大小的窗口的最大值
			{
				for (n = j*mapSize; n<j*mapSize + mapSize; n++)
				{
					if (max < input.ptr<float>(m)[n])
					{
						max = input.ptr<float>(m)[n];
						max_index = m*input.cols + n;
					}
				}
			}

			output.ptr<float>(i)[j] = max;
			max_position.ptr<int>(i)[j] = max_index;
		}
	}
}


/*
relu6:
0, x ≤ 0
y = x, 0 < x < 6
6, x ≥ 6
*/

// 激活函数 input是数据，inputNum说明数据数目，bas表明偏置
float activation_Sigma(float input, float bas) // sigma激活函数
{
#ifdef RELU_USE
	float temp = input + bas;
	return (temp > 0 ? temp: 0);
	//return (temp>0 ? (temp<6?temp:6): 0);
	//return (temp>0.0 ? (temp<6.0?temp:6.0): (temp/A1));
	/*if(temp >= 0.0)
	return temp;
	else
	return (temp/A1);*/

	/*if (temp > 0.0)
	{
		if (temp < 6.0)
			return temp;    //dy/dx = 1, 0.0<y<6.0
		else
			return 6.0;     //dy/dx = 0, y>=6.0
	}
	else
	{
		float x = temp / A1;
		if (x <= -6.0)
			return -6.0;   //dy/dx = 0, y<=-6.0
		else
			return x;      //dy/dx = 1/A1, -6.0<y≤0.0
	}*/
#else
	float temp = input + bas;
	return (float)1.0 / ((float)(1.0 + exp(-temp)));
#endif
}


// 单层全连接神经网络的前向传播
float vecMulti(Mat vec1, float *vec2)// 两向量相乘
{
	float *p1 = vec1.ptr<float>(0);
	float m = 0;
	for (int i = 0; i < vec1.cols; i++)
		m = m + p1[i] * vec2[i];
	return m;
}

void nnff(Mat input, Mat wdata, Mat &output)
{
	for (int i = 0; i < output.cols; i++)  //分别计算多个向量相乘的乘积
		output.ptr<float>(0)[i] = vecMulti(input, wdata.ptr<float>(i));   //由于输入激活函数之前就有加上偏置的操作，所以此处不再加偏置
}

void softmax(OutLayer &O)
{
	float sum = 0.0;
	float *p_y = O.y.ptr<float>(0);
	float *p_v = O.v.ptr<float>(0);
	float *p_b = O.basicData.ptr<float>(0);
	for (int i = 0; i < O.outputNum; i++)
	{
		//O.y.ptr<float>(0)[i] = activation_Sigma(O.v.ptr<float>(0)[i], O.basicData.ptr<float>(0)[i]);
		float Yi = exp(p_v[i]+ p_b[i]);
		//float Yi = exp(p_y[i]);
		sum += Yi;
		p_y[i] = Yi;
	}

	for (int i = 0; i < O.outputNum; i++)
	{
		p_y[i] = p_y[i]/sum;
	}
	//cout << O.y << endl;
}


void cov_layer_ff(vector<Mat> inputData, int cov_type, CovLayer &C)
{
	for (int i = 0; i < (C.outChannels); i++)   
	{
		for (int j = 0; j < (C.inChannels); j++)   
		{
			//Mat mapout = cov(C.mapData[j][i], inputData[j], cov_type);   //计算卷积，mapData为四维矩阵
			Mat mapout = correlation(C.mapData[j][i], inputData[j], cov_type);
			C.v[i] += mapout;           //所有输入通道的卷积结果累加
		}

		int output_r = C.y[i].rows;
		int output_c = C.y[i].cols;
		for (int r = 0; r < output_r; r++)
		{
			for (int c = 0; c < output_c; c++)
			{
				C.y[i].ptr<float>(r)[c] = activation_Sigma(C.v[i].ptr<float>(r)[c], C.basicData.ptr<float>(0)[i]);   //先加上偏置，再输入sigmoid激活函数
			}
		}
	}
}


void pool_layer_ff(vector<Mat> inputData, int pool_type, PoolLayer &S)
{
	if (pool_type == AvePool)
	{
		for (int i = 0; i < S.outChannels; i++)  
		{
			avgPooling(inputData[i], S.y[i], S.mapSize);
		}
	}
	else if(pool_type == MaxPool)
	{
		for (int i = 0; i < S.outChannels; i++)  
		{
			maxPooling(inputData[i], S.max_position[i], S.y[i], S.mapSize);
		}
	}
	else
	{
		printf("pool type erroe!\n");
	}
}


void out_layer_ff(vector<Mat> inputData, OutLayer &O)
{
	Mat OinData(1, O.inputNum, CV_32FC1);   //输入192通道
	float *OinData_p = OinData.ptr<float>(0);
	int outsize_r = inputData[0].rows;
	int outsize_c = inputData[0].cols;
	int last_output_len = inputData.size();
	for (int i = 0; i < last_output_len; i++)   //上一层S4输出12通道的4*4矩阵
	{
		for (int r = 0; r < outsize_r; r++)
		{
			for (int c = 0; c < outsize_c; c++)
			{
				OinData_p[i*outsize_r*outsize_c + r*outsize_c + c] = inputData[i].ptr<float>(r)[c];  //将12通道4*4矩阵一维展开
			}
		}
	}

	//192*10个权重
	nnff(OinData, O.wData, O.v);   //10通道输出,1个通道的输出等于192个输入分别与192个权重相乘的和：∑in[i]*w[i], 0≤i<192

	//for (int i = 0; i < O.outputNum; i++)
	//	O.y.ptr<float>(0)[i] = activation_Sigma(O.v.ptr<float>(0)[i], O.basicData.ptr<float>(0)[i]);

	softmax(O);
}


void cnnff(CNN &cnn, Mat inputData)
{
	// 第一层的传播
	//5*5卷积核
	//输入28*28矩阵
	//输出(28-25+1)*(28-25+1) = 24*24矩阵
	vector<Mat> input_tmp;
	input_tmp.push_back(inputData);
	cov_layer_ff(input_tmp, valid, cnn.C1);

	//24*24-->12*12
	pool_layer_ff(cnn.C1.y, MaxPool, cnn.S2);

	//12*12-->8*8
	cov_layer_ff(cnn.S2.y, valid, cnn.C3);

	//8*8-->4*4
	pool_layer_ff(cnn.C3.y, MaxPool, cnn.S4);

	//12*4*4-->192-->1*10
	out_layer_ff(cnn.S4.y, cnn.O5);
}



float sigma_derivation(float y)
{ // Logic激活函数的自变量微分

#ifdef RELU_USE
  return (y > 0.0 ? 1.0 : 0.0);
  //return ((y<=0.0) ? (1.0/A1) : (y<6.0?1.0:0.0));
  /*if(y > 0.0 && y < 6.0)
  return 1.0;
  else
  return 0.0;*/
  /*if(y >= 0.0)
  return 1.0;
  else
  return (1.0/A1);*/

	/*if (y>0.0 && y<6.0)
		return 1.0;
	else if (y >= 6.0 || y <= -6.0)
		return 0.0;
	else
		return (1.0 / A1);*/

#else
	return y*(1 - y); // 这里y是指经过激活函数的输出值，而不是自变量
#endif
}


/*
矩阵上采样，upc及upr是内插倍数
如果是最大值池化模式，则把局域梯度放到池化前最大值的位置，比如池化窗口2*2，池化前最大值的位置分别为左上、右上、左下、右下，则上采样后为：
5 9				5 0 0 9
     -->		0 0 0 0
3 6				0 0 0 0
                3 0 0 6
如果是均值池化模式，则把局域梯度除以池化窗口的尺寸2*2=4:
5 9				1.25 1.25 2.25 2.25
     -->		1.25 1.25 2.25 2.25
3 6				0.75 0.75 1.5  1.5
                0.75 0.75 1.5  1.5
*/
Mat UpSample(Mat mat, int upc, int upr)   //均值池化层的向上采样
{
	//int i, j, m, n;
	int c = mat.cols;
	int r = mat.rows;
	
	Mat res(r*upr, c*upc, CV_32FC1);

	float pooling_size = 1.0 / (upc*upr);

	for (int j = 0; j < r*upr; j += upr)
	{
		for (int i = 0; i < c*upc; i += upc)  // 宽的扩充
		{
			for (int m = 0; m < upc; m++)
			{

				//res[j][i + m] = mat[j / upr][i / upc] * pooling_size;
				res.ptr<float>(j)[i + m] = mat.ptr<float>(j/upr)[i/upc] * pooling_size;
			}
		}

		for (int n = 1; n < upr; n++)      //  高的扩充
		{
			for (int i = 0; i < c*upc; i++)
			{
				//res[j + n][i] = res[j][i];
				res.ptr<float>(j+n)[i] = res.ptr<float>(j)[i];

			}
		}
	}
	return res;
}


//最大值池化层的向上采样
Mat maxUpSample(Mat mat, Mat max_position, int upc, int upr)
{
	//int i, j, m, n;
	int c = mat.cols;
	int r = mat.rows;

	int outsize_r = r*upr;
	int outsize_c = c*upc;

	Mat res = Mat::zeros(outsize_r, outsize_c, CV_32FC1);

	for (int j = 0; j < r; j++)
	{
		for (int i = 0; i < c; i++)
		{
			int index_r = max_position.ptr<int>(j)[i] / outsize_c;   //计算最大值的索引
			int index_c = max_position.ptr<int>(j)[i] % outsize_c;
			res.ptr<float>(index_r)[index_c] = mat.ptr<float>(j)[i];
		}
	}
	return res;
}





//反向：输出层-->全连接层
void softmax_bp(Mat outputData, Mat &e, OutLayer &O)
{
	for (int i = 0; i < O.outputNum; i++)
		e.ptr<float>(0)[i] = O.y.ptr<float>(0)[i] - outputData.ptr<float>(0)[i];   //这里是求1/2∑(di-yi)^2对yi的偏导数,得到yi-di

	/*从后向前反向计算*/																				
	for (int i = 0; i < O.outputNum; i++)
		O.d.ptr<float>(0)[i] = e.ptr<float>(0)[i];// *sigma_derivation(O.y.ptr<float>(0)[i]);
}


//反向：全连接层-->池化层
void full2pool_bp(OutLayer O, PoolLayer &S)
{
	int outSize_r = S.inputHeight / S.mapSize;
	int outSize_c = S.inputWidth / S.mapSize;
	for (int i = 0; i < S.outChannels; i++)  //输出12张4*4图像
	{
		for (int r = 0; r < outSize_r; r++)
		{
			for (int c = 0; c < outSize_c; c++)
			{
				int wInt = i*outSize_c*outSize_r + r*outSize_c + c;  //i*outSize.c*outSize.r为图像索引，r*outSize.c+c为每张图像中的像素索引
				for (int j = 0; j < O.outputNum; j++)   //O5输出层的输出个数
				{
					S.d[i].ptr<float>(r)[c] = S.d[i].ptr<float>(r)[c] + O.d.ptr<float>(0)[j] * O.wData.ptr<float>(j)[wInt];  //d_S4 = ∑d_O5*W
				}
				//S.d[i].ptr<float>(r)[c] /= 10.0;
			}
		}
	}
}


//反向：池化层-->卷积层
void pool2cov_bp(PoolLayer S, CovLayer &C)
{

	for (int i = 0; i < C.outChannels; i++)   //12通道
	{
		Mat C3e;
		if (S.poolType == AvePool)
			C3e = UpSample(S.d[i], S.mapSize, S.mapSize);    //向上采样，把S4层的局域梯度由4*4扩充为8*8
		else if (S.poolType == MaxPool)
			C3e = maxUpSample(S.d[i], S.max_position[i], S.mapSize, S.mapSize);


		for (int r = 0; r < S.inputHeight; r++)   //8*8
		{
			for (int c = 0; c < S.inputWidth; c++)
			{
				C.d[i].ptr<float>(r)[c] = C3e.ptr<float>(r)[c] * sigma_derivation(C.y[i].ptr<float>(r)[c]);
			}
		}
	}
}


//反向：卷积层-->池化层
void cov2pool_bp(CovLayer C, int cov_type, PoolLayer &S)
{

	for (int i = 0; i < S.outChannels; i++)   //S2有6通道
	{
		for (int j = 0; j < S.inChannels; j++)  //C3有12通道
		{
			//Mat tmp;
			//flip(C.mapData[i][j], flipmap, -1);
			//Mat corr = correlation(C.mapData[i][j], C.d[j], cov_type);   //计算互相关,得到12*12矩阵：full模式下为(inSize+mapSize-1)*(inSize+mapSize-1)
		    Mat corr = cov(C.mapData[i][j], C.d[j], cov_type);
			S.d[i] = S.d[i] + corr;   //矩阵累加：cnn->S2->d[i] = cnn->S2->d[i] + corr，得到6个12*12局域梯度
		}
		//S.d[i] /= S.inChannels;
	}
}



void cnnbp(CNN &cnn, Mat outputData) // 网络的后向传播
{
	softmax_bp(outputData, cnn.e, cnn.O5);
	full2pool_bp(cnn.O5, cnn.S4);
	pool2cov_bp(cnn.S4, cnn.C3);
	cov2pool_bp(cnn.C3, full, cnn.S2);
	pool2cov_bp(cnn.S2, cnn.C1);
}




void update_cov_para(vector<Mat> inputData, CNNOpts opts, CovLayer &C)
{
	for (int i = 0; i < C.outChannels; i++)   //6通道
	{
		for (int j = 0; j < C.inChannels; j++)   //1通道
		{
			//Mat flipinputData = rotate180(inputData[j]);   //矩阵翻转180度      
			//Mat Cdk = cov(C.d[i], flipinputData, valid);   //计算σ*X，valid模式下输出(ySize-dSize+1)*(ySize-dSize+1) = 5*5矩阵
			Mat Cdk = correlation(C.d[i], inputData[j], valid);
			Cdk = Cdk*(-opts.alpha);   //矩阵乘以系数-α得到-α*σ*X
			C.mapData[j][i] = C.mapData[j][i] + Cdk;   //计算W[n+1] = W[n] - α*σ*X
		}

		float d_sum = (float)cv::sum(C.d[i])[0];   //这里有6个24*24的d，6个偏置b，一个偏置b对应一个24*24矩阵d的所有元素和
		C.basicData.ptr<float>(0)[i] = C.basicData.ptr<float>(0)[i] - opts.alpha*d_sum;
	}
}


void update_full_para(vector<Mat> inputData, CNNOpts opts, OutLayer &O)
{
	int outSize_r = inputData[0].rows;
	int outSize_c = inputData[0].cols;
	Mat OinData(1, outSize_r*outSize_c*inputData.size(), CV_32FC1);
	for (int i = 0; i < inputData.size(); i++)   //12通道
	{
		for (int r = 0; r < outSize_r; r++)  //4
		{
			for (int c = 0; c < outSize_c; c++)   //4
			{
				OinData.ptr<float>(0)[i*outSize_r*outSize_c + r*outSize_c + c] = inputData[i].ptr<float>(r)[c];
			}
		}
	}

	for (int j = 0; j < O.outputNum; j++)  //10通道
	{
		for (int i = 0; i < O.inputNum; i++)  //192通道
		{
			O.wData.ptr<float>(j)[i] = O.wData.ptr<float>(j)[i] - opts.alpha*O.d.ptr<float>(0)[j] * OinData.ptr<float>(0)[i];
		}

		O.basicData.ptr<float>(0)[j] = O.basicData.ptr<float>(0)[j] - opts.alpha*O.d.ptr<float>(0)[j];
	}
}

void cnnapplygrads(CNN &cnn, CNNOpts opts, Mat inputData) // 更新权重
{
	vector<Mat> input_tmp;
	input_tmp.push_back(inputData);

	update_cov_para(input_tmp, opts, cnn.C1);

	update_cov_para(cnn.S2.y, opts, cnn.C3);

	update_full_para(cnn.S4.y, opts, cnn.O5);
}



void clear_cov_mid_para(CovLayer &C)
{
	int row = C.d[0].rows;
	int col = C.d[0].cols;
	for (int j = 0; j < C.outChannels; j++)
	{
		for (int r = 0; r < row; r++)
		{
			for (int c = 0; c < col; c++)
			{
				C.d[j].ptr<float>(r)[c] = 0.0;
				C.v[j].ptr<float>(r)[c] = 0.0;
				C.y[j].ptr<float>(r)[c] = 0.0;
			}
		}
		//C.d[j] = Mat::zeros(row, col, CV_32FC1);
		//C.v[j] = Mat::zeros(row, col, CV_32FC1);
		//C.y[j] = Mat::zeros(row, col, CV_32FC1);
	}
}


void clear_pool_mid_para(PoolLayer &S)
{
	int row = S.d[0].rows;
	int col = S.d[0].cols;
	for (int j = 0; j < S.outChannels; j++)
	{
		for (int r = 0; r < row; r++)
		{
			for (int c = 0; c < col; c++)
			{
				S.d[j].ptr<float>(r)[c] = 0.0;
				S.y[j].ptr<float>(r)[c] = 0.0;
			}
		}
		//S.d[j] = Mat::zeros(row, col, CV_32FC1);
		//S.y[j] = Mat::zeros(row, col, CV_32FC1);
	}
}

void clear_out_mid_para(OutLayer &O)
{
	for (int j = 0; j < O.outputNum; j++)
	{
		O.d.ptr<float>(0)[j] = 0.0;
		O.v.ptr<float>(0)[j] = 0.0;
		O.y.ptr<float>(0)[j] = 0.0;
	}
	/*int row = O.d.rows;
	int col = O.d.cols;
	O.d = Mat::zeros(row, col, CV_32FC1);
	O.v = Mat::zeros(row, col, CV_32FC1);
	O.y = Mat::zeros(row, col, CV_32FC1);*/
}

void cnnclear(CNN &cnn)
{
	clear_cov_mid_para(cnn.C1);
	clear_pool_mid_para(cnn.S2);
	clear_cov_mid_para(cnn.C3);
	clear_pool_mid_para(cnn.S4);
	clear_out_mid_para(cnn.O5);
}




void cnnsetup(CNN &cnn, int inputSize_r, int inputSize_c, int outputSize)   //cnn初始化
{
	cnn.layerNum = 5;

	//nSize inSize;
	int mapSize = 5;
	int inSize_c = inputSize_c;   //28
	int inSize_r = inputSize_r;   //28
	int C1_outChannels = 6;
	cnn.C1 = initCovLayer(inSize_c, inSize_r, mapSize, 1, C1_outChannels);   //卷积层1

	inSize_c = inSize_c - cnn.C1.mapSize + 1;  //24
	inSize_r = inSize_r - cnn.C1.mapSize + 1;  //24
	mapSize = 2;
	cnn.S2 = initPoolLayer(inSize_c, inSize_r, mapSize, cnn.C1.outChannels, cnn.C1.outChannels, MaxPool);   //池化层

	inSize_c = inSize_c / cnn.S2.mapSize;   //12
	inSize_r = inSize_r / cnn.S2.mapSize;   //12
	mapSize = 5;
	int C3_outChannes = 12;
	cnn.C3 = initCovLayer(inSize_c, inSize_r, mapSize, cnn.S2.outChannels, C3_outChannes);   //卷积层

	inSize_c = inSize_c - cnn.C3.mapSize + 1;   //8
	inSize_r = inSize_r - cnn.C3.mapSize + 1;   //8
	mapSize = 2;
	cnn.S4 = initPoolLayer(inSize_c, inSize_r, mapSize, cnn.C3.outChannels, cnn.C3.outChannels, MaxPool);    //池化层

	inSize_c = inSize_c / cnn.S4.mapSize;   //4
	inSize_r = inSize_r / cnn.S4.mapSize;   //4
	cnn.O5 = initOutLayer(inSize_c*inSize_r*cnn.S4.outChannels, outputSize);    //输出层

	cnn.e = Mat::zeros(1, cnn.O5.outputNum, CV_32FC1);   //输出层的输出值与标签值之差
}

vector<float> LL;
void write_LL(void)
{
	FILE *fp = fopen("LL.bin", "wb+");

	for (int i = 0; i < LL.size(); i++)
	{
		fprintf(fp, "%f, ", LL[i]);
	}

	fclose(fp);
}

void cnntrain(CNN &cnn, vector<Mat> inputData, vector<Mat> outputData, CNNOpts opts, int trainNum)
{
	// 学习训练误差曲线
	//cnn->L = (float*)malloc(trainNum * sizeof(float));
	cnn.L = Mat(1, trainNum, CV_32FC1).clone();
	for (int e = 0;  e < opts.numepochs; e++)
	{
		for (int n = 0; n < trainNum; n++)
		{
			#ifndef RELU_USE
			opts.alpha = 0.1 - 0.075*n / (trainNum - 1);    //学习率递减2.0~0.25
			#else
			opts.alpha = 0.03 - 0.029*n / (trainNum - 1);    //学习率递减0.01~0.001
			//opts.alpha = 0.01 -0.009*n / (trainNum - 1);
			#endif

														
			cnnff(cnn, inputData[n]);  // 前向传播，这里主要计算各
			cnnbp(cnn, outputData[n]); // 后向传播，这里主要计算各神经元的误差梯度
			cnnapplygrads(cnn, opts, inputData[n]); // 更新权重

													// 计算并保存误差能量
#if 0
			float l = 0.0;
			for (int i = 0; i < cnn.O5.outputNum; i++)
				l = l + cnn.e.ptr<float>(0)[i] * cnn.e.ptr<float>(0)[i];

			cnn.L.ptr<float>(0)[n] = l / 2.0;

			LL.push_back(cnn.L.ptr<float>(0)[n]);
#else
			float l = 0.0;
			for (int i = 0; i < cnn.O5.outputNum; i++)
			{
				l = l - outputData[n].ptr<float>(0)[i] * log(cnn.O5.y.ptr<float>(0)[i]);
			}
			//while (1);

			cnn.L.ptr<float>(0)[n] = l;

			//if(n%101 == 0)
			//	LL.push_back(l);
#endif

			cnnclear(cnn);

			printf("n=%d, f=%f, α=%f\n", n, cnn.L.ptr<float>(0)[n], opts.alpha);
		}
	}

	//write_LL();
}




// 测试cnn函数
float cnntest(CNN cnn, vector<Mat> inputData, vector<Mat> outputData)
{
	int incorrectnum = 0;  //错误预测的数目
	for (int i = 0; i < inputData.size(); i++)
	{
		cnnff(cnn, inputData[i]);

		if (vecmaxIndex(cnn.O5.y) != vecmaxIndex(outputData[i]))
		{
			incorrectnum++;
			printf("i = %d, 识别失败\n", i);
		}
		else
		{
			printf("i = %d, 识别成功\n", i);
		}
		cnnclear(cnn);
	}
	printf("incorrectnum=%d\n", incorrectnum);
	printf("inputData.size()=%d\n", inputData.size());
	return (float)incorrectnum / (float)inputData.size();
}


void CNN_test(void)
{
	vector<Mat> traindata;
	vector<Mat> trainlabel;
	char ad[128] = { 0 };

#if 1
	const int trainnum = 900;
	
	//加载data_batch_1.bin的训练数据
	for (int j = 0; j < trainnum; j++)
	{
		for (int i = 0; i < 10; i++)
		{
			printf("i=%d, j=%d\n", i, j);
			sprintf_s(ad, "D:/Program Files (x86)/Microsoft Visual Studio 14.0/prj/KNN_test/KNN_test/cifar/batch1/%d/%d.tif", i, j);
			Mat srcimage = imread(ad, CV_LOAD_IMAGE_GRAYSCALE);

			resize(srcimage, srcimage, Size(28, 28), INTER_CUBIC);
			srcimage.convertTo(srcimage, CV_32F);
			srcimage /= 255.0;
			traindata.push_back(srcimage.clone());

			float label[10] = { 0.0 };
			label[i] = 1.0;
			Mat label_tmp(1, 10, CV_32FC1, label);
			trainlabel.push_back(label_tmp.clone());
		}
	}

	//加载data_batch_2.bin的训练数据
	for (int j = 0; j < trainnum; j++)
	{
		for (int i = 0; i < 10; i++)
		{
			printf("i=%d, j=%d\n", i, j);
			sprintf_s(ad, "D:/Program Files (x86)/Microsoft Visual Studio 14.0/prj/KNN_test/KNN_test/cifar/batch2/%d/%d.tif", i, j);
			Mat srcimage = imread(ad, CV_LOAD_IMAGE_GRAYSCALE);

			resize(srcimage, srcimage, Size(28, 28), INTER_CUBIC);
			srcimage.convertTo(srcimage, CV_32F);
			srcimage /= 255.0;
			traindata.push_back(srcimage.clone());

			float label[10] = { 0.0 };
			label[i] = 1.0;
			Mat label_tmp(1, 10, CV_32FC1, label);
			trainlabel.push_back(label_tmp.clone());
		}
	}

	//加载data_batch_3.bin的训练数据
	for (int j = 0; j < trainnum; j++)
	{
		for (int i = 0; i < 10; i++)
		{
			printf("i=%d, j=%d\n", i, j);
			sprintf_s(ad, "D:/Program Files (x86)/Microsoft Visual Studio 14.0/prj/KNN_test/KNN_test/cifar/batch3/%d/%d.tif", i, j);
			Mat srcimage = imread(ad, CV_LOAD_IMAGE_GRAYSCALE);

			resize(srcimage, srcimage, Size(28, 28), INTER_CUBIC);
			srcimage.convertTo(srcimage, CV_32F);
			srcimage /= 255.0;
			traindata.push_back(srcimage.clone());
			
			float label[10] = {0.0};
			label[i] = 1.0;
			Mat label_tmp(1, 10, CV_32FC1, label);
			trainlabel.push_back(label_tmp.clone());
		}
	}

	//加载data_batch_4.bin的训练数据
	for (int j = 0; j < trainnum; j++)
	{
		for (int i = 0; i < 10; i++)
		{
			printf("i=%d, j=%d\n", i, j);
			sprintf_s(ad, "D:/Program Files (x86)/Microsoft Visual Studio 14.0/prj/KNN_test/KNN_test/cifar/batch4/%d/%d.tif", i, j);
			Mat srcimage = imread(ad, CV_LOAD_IMAGE_GRAYSCALE);

			resize(srcimage, srcimage, Size(28, 28), INTER_CUBIC);
			srcimage.convertTo(srcimage, CV_32F);
			srcimage /= 255.0;
			traindata.push_back(srcimage.clone());

			float label[10] = { 0.0 };
			label[i] = 1.0;
			Mat label_tmp(1, 10, CV_32FC1, label);
			trainlabel.push_back(label_tmp.clone());
		}
	}

	//加载data_batch_5.bin的训练数据
	for (int j = 0; j < trainnum; j++)
	{
		for (int i = 0; i < 10; i++)
		{
			printf("i=%d, j=%d\n", i, j);
			sprintf_s(ad, "D:/Program Files (x86)/Microsoft Visual Studio 14.0/prj/KNN_test/KNN_test/cifar/batch5/%d/%d.tif", i, j);
			Mat srcimage = imread(ad, CV_LOAD_IMAGE_GRAYSCALE);

			resize(srcimage, srcimage, Size(28, 28), INTER_CUBIC);
			srcimage.convertTo(srcimage, CV_32F);
			srcimage /= 255.0;
			traindata.push_back(srcimage.clone());

			float label[10] = { 0.0 };
			label[i] = 1.0;
			Mat label_tmp(1, 10, CV_32FC1, label);
			trainlabel.push_back(label_tmp.clone());
		}
	}

	vector<Mat> testdata_list;
	vector<Mat> testdata_label;
	//对test_batch.bin中的前800张图像进行预测分类
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 800; j++)
		{
			sprintf_s(ad, "D:/Program Files (x86)/Microsoft Visual Studio 14.0/prj/KNN_test/KNN_test/cifar/test_batch/%d/%d.tif", i, j);
			Mat testdata = imread(ad, CV_LOAD_IMAGE_GRAYSCALE);

			resize(testdata, testdata, Size(28, 28), INTER_CUBIC);
			testdata.convertTo(testdata, CV_32F);
			testdata /= 255.0;
			testdata_list.push_back(testdata.clone());

			float label[10] = { 0.0 };
			label[i] = 1.0;
			Mat label_tmp(1, 10, CV_32FC1, label);
			testdata_label.push_back(label_tmp.clone());
		}
	}

#else

	
		

	for (int j = 0; j < 400; j++)
	{
		for (int i = 0; i < 10; i++)
		{
			float label[10] = { 0.0 };
			label[i] = 1.0;
			Mat label_tmp(1, 10, CV_32FC1, label);
				
			printf("i=%d, j=%d\n", i, j);
			sprintf_s(ad, "D:/Program Files (x86)/Microsoft Visual Studio 14.0/prj/KNN_test/KNN_test/%d/%d.jpg", i, j);
			Mat srcimage = imread(ad, CV_LOAD_IMAGE_GRAYSCALE);
			resize(srcimage, srcimage, Size(28, 28), INTER_CUBIC);

			//imshow("srcimage", srcimage);
			//waitKey();

			srcimage.convertTo(srcimage, CV_32F);
			srcimage /= 255.0;
			traindata.push_back(srcimage.clone());   //将向量输入到训练矩阵中
			trainlabel.push_back(label_tmp.clone());
		}
			
	}
	

	vector<Mat> testdata_list;
	vector<Mat> testdata_label;
	for (int i = 0; i < 10; i++)
	{
		float label[10] = { 0.0 };
		label[i] = 1.0;
		Mat label_tmp(1, 10, CV_32FC1, label);
		
		for (int j = 400; j < 500; j++)
		{
			sprintf_s(ad, "D:/Program Files (x86)/Microsoft Visual Studio 14.0/prj/KNN_test/KNN_test/%d/%d.jpg", i, j);
			Mat testdata = imread(ad, CV_LOAD_IMAGE_GRAYSCALE);
			resize(testdata, testdata, Size(28, 28), INTER_CUBIC);
			testdata.convertTo(testdata, CV_32F);
			testdata /= 255.0;
			testdata_list.push_back(testdata.clone());   //将向量输入到训练矩阵中
			testdata_label.push_back(label_tmp.clone());
		}
	}

#endif

	CNNOpts opts;
	opts.numepochs = 1;
	opts.alpha = 1.5;
	int trainNum = 60000;

	CNN cnn;
	printf("***********************cnnsetup\n");
	cnnsetup(cnn, 28, 28, 10);   //cnn初始化
	printf("***********************ccnntrain\n");
	cnntrain(cnn, traindata, trainlabel, opts, traindata.size());
	printf("***********************finish\n");

	float success = cnntest(cnn, testdata_list, testdata_label);
	printf("success=%f\n", 1-success);
}




vector<Mat> read_Img_to_Mat(const char* filename) // 读入图像
{
	FILE  *fp = NULL;
	fp = fopen(filename, "rb");
	if (fp == NULL)
		printf("open file failed\n");
	assert(fp);

	int magic_number = 0;
	int number_of_images = 0;
	int n_rows = 0;
	int n_cols = 0;
	
	fread(&magic_number, sizeof(int), 1, fp);   //从文件中读取sizeof(int) 个字符到 &magic_number  
	magic_number = ReverseInt(magic_number);

	//printf("magic_number=%d\n", magic_number);
	//while (1);
	
	fread(&number_of_images, sizeof(int), 1, fp);   //获取训练或测试image的个数number_of_images 
	number_of_images = ReverseInt(number_of_images);
	
	fread(&n_rows, sizeof(int), 1, fp);   //获取训练或测试图像的高度Heigh  
	n_rows = ReverseInt(n_rows);
	
	fread(&n_cols, sizeof(int), 1, fp);   //获取训练或测试图像的宽度Width  
	n_cols = ReverseInt(n_cols);

	//printf("magic_number=%d, number_of_images=%d, n_rows=%d, n_cols=%d\n", magic_number, number_of_images, n_rows, n_cols);


	//获取第i幅图像，保存到vec中 
	int i, r, c;

	// 图像数组的初始化
	//ImgArr imgarr = (ImgArr)malloc(sizeof(MinstImgArr));
	//imgarr->ImgNum = number_of_images;
	//imgarr->ImgPtr = (MinstImg*)malloc(number_of_images * sizeof(MinstImg));
	
	int img_size = n_rows*n_cols;
	vector<Mat> img_list;
	for (i = 0; i < number_of_images; ++i)
	{
		
		/*for (r = 0; r < n_rows; ++r)
		{
			float *p = tmp.ptr<float>(r);
			for (c = 0; c < n_cols; ++c)
			{
				unsigned char temp = 0;
				fread((char*)&temp, sizeof(temp), 1, fp);
				//imgarr->ImgPtr[i].ImgData[r][c] = (float)temp / 255.0;
				p[c] = (float)temp / 255.0;
			}
		}*/

		Mat tmp(n_rows, n_cols, CV_8UC1);
		fread(tmp.data, sizeof(uchar), img_size, fp);
		tmp.convertTo(tmp, CV_32F);
		tmp = tmp / 255.0;
		img_list.push_back(tmp.clone());
	}

	fclose(fp);
	return img_list;
}




vector<Mat> read_Lable_to_Mat(const char* filename)// 读入图像
{
	FILE  *fp = NULL;
	fp = fopen(filename, "rb");
	if (fp == NULL)
		printf("open file failed\n");
	assert(fp);

	int magic_number = 0;
	int number_of_labels = 0;
	int label_long = 10;

	
	fread(&magic_number, sizeof(int), 1, fp);   //从文件中读取sizeof(magic_number) 个字符到 &magic_number  
	magic_number = ReverseInt(magic_number);

	//printf("magic_number=%d\n", magic_number);
	//while (1);
	
	fread(&number_of_labels, sizeof(int), 1, fp);   //获取训练或测试image的个数number_of_images 
	number_of_labels = ReverseInt(number_of_labels);

	int i, l;

	// 图像标记数组的初始化
	//LabelArr labarr = (LabelArr)malloc(sizeof(MinstLabelArr));
	//labarr->LabelNum = number_of_labels;
	//labarr->LabelPtr = (MinstLabel*)malloc(number_of_labels * sizeof(MinstLabel));
	vector<Mat> label_list;
	
	for (i = 0; i < number_of_labels; ++i)
	{
		//labarr->LabelPtr[i].l = 10;
		//labarr->LabelPtr[i].LabelData = (float*)calloc(label_long, sizeof(float));

		Mat tmp = Mat::zeros(1, label_long, CV_32FC1);

		unsigned char temp = 0;
		fread(&temp, sizeof(unsigned char), 1, fp);
		tmp.ptr<float>(0)[(int)temp] = 1.0;

		label_list.push_back(tmp.clone());
	}

	fclose(fp);
	return label_list;
}
