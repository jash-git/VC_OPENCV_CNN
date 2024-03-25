#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <random>
#include <ctime>
#include <windows.h>    //΢�뼶��ʱ��غ���
#include "CNN.h"
#include "minst.h"

using namespace cv;
using namespace std;


#define randf(a, b) (((rand()%10000+rand()%10000*10000)/100000000.0)*((b)-(a))+(a))


//��ʼ�������
CovLayer initCovLayer(int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels)
{
	CovLayer covL;

	covL.inputHeight = inputHeight;
	covL.inputWidth = inputWidth;
	covL.mapSize = mapSize;

	covL.inChannels = inChannels;
	covL.outChannels = outChannels;

	covL.isFullConnect = true;   // Ĭ��Ϊȫ����

	// Ȩ�ؿռ�ĳ�ʼ�����������е��ã�[r][c]
	srand((unsigned)time(NULL));   //�������������
	for(int i = 0; i < inChannels; i++)   //����ͨ����
	{
		vector<Mat> tmp;
		for(int j = 0; j < outChannels; j++)   //���ͨ����
		{
			Mat tmpmat(mapSize, mapSize, CV_32FC1);
			for(int r = 0; r < mapSize; r++)   //����˵ĸ�
			{
				for(int c = 0; c < mapSize; c++)  //����˵Ŀ�
				{
					float randnum=(((float)rand()/(float)RAND_MAX)-0.5)*2;    //����-1~1�������
					tmpmat.ptr<float>(r)[c] = randnum*sqrt(6.0/(mapSize*mapSize*(inChannels+outChannels)));
					//tmpmat.ptr<float>(r)[c] = randf(-0.05, 0.05);
				}
			}
			tmp.push_back(tmpmat.clone());
		}
		covL.mapData.push_back(tmp);
	}


	/*Mat tmpmat1 = Mat::zeros(mapSize, mapSize, CV_32FC1);
	for(int i = 0; i < inChannels; i++)   //����ͨ����
	{
		vector<Mat> tmp;
		for(int j = 0; j < outChannels; j++)   //���ͨ����
		{
			tmp.push_back(tmpmat1.clone());
		}
		covL.dmapData.push_back(tmp);
	}*/
 
	covL.basicData = Mat::zeros(1, outChannels, CV_32FC1);   //��ʼ�������ƫ�õ��ڴ�

	int outW = inputWidth - mapSize + 1;   //���������Ŀ�
	int outH = inputHeight - mapSize + 1;  //���������ĸ�

	Mat tmpmat2 = Mat::zeros(outH, outW, CV_32FC1);
	for(int i = 0; i < outChannels; i++)
	{
		covL.d.push_back(tmpmat2.clone());
		covL.v.push_back(tmpmat2.clone());
		covL.y.push_back(tmpmat2.clone());
	}

	return covL;
}



//�ػ����ʼ��
PoolLayer initPoolLayer(int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels, int poolType)
{
	PoolLayer poolL;

	poolL.inputHeight=inputHeight;    //����߶�
	poolL.inputWidth=inputWidth;      //������
	poolL.mapSize=mapSize;            //����˳ߴ磬�ػ����൱����һ������ľ������
	poolL.inChannels=inChannels;      //����ͨ��
	poolL.outChannels=outChannels;    //���ͨ��
	poolL.poolType=poolType;          //���ֵģʽ/ƽ��ֵģʽ

	poolL.basicData = Mat::zeros(1, outChannels, CV_32FC1);    //�ػ�����ƫ�ã��޼���������������ڴ�ֻ��Ԥ��

	int outW = inputWidth/mapSize;   //�ػ���ľ����Ϊ2*2
	int outH = inputHeight/mapSize;

	Mat tmpmat = Mat::zeros(outH, outW, CV_32FC1);
	Mat tmpmat1 = Mat::zeros(outH, outW, CV_32SC1);
	for(int i = 0; i < outChannels; i++)
	{
		poolL.d.push_back(tmpmat.clone());   //�����ݶ�
		poolL.y.push_back(tmpmat.clone());   //������������Ԫ������޼����
		poolL.max_position.push_back(tmpmat1.clone());   //���ֵģʽ�����ֵ��λ��
	}

	return poolL;
}


//������ʼ��
OutLayer initOutLayer(int inputNum, int outputNum)
{
	OutLayer outL;

	outL.inputNum = inputNum;
	outL.outputNum = outputNum;
	outL.isFullConnect = true;

	outL.basicData = Mat::zeros(1, outputNum, CV_32FC1);    //ƫ��,�����ڴ��ͬʱ��ʼ��Ϊ0
	outL.d = Mat::zeros(1, outputNum, CV_32FC1);
	outL.v = Mat::zeros(1, outputNum, CV_32FC1);
	outL.y = Mat::zeros(1, outputNum, CV_32FC1);

	// Ȩ�صĳ�ʼ��
	outL.wData = Mat::zeros(outputNum, inputNum, CV_32FC1);   // ����У�������,Ȩ��Ϊ10*192����
	srand((unsigned)time(NULL));
	for(int i = 0; i < outputNum; i++)
	{
		float *p = outL.wData.ptr<float>(i);
		for(int j = 0; j < inputNum; j++)
		{
			float randnum = (((float)rand()/(float)RAND_MAX)-0.5)*2; // ����һ��-1��1�������,rand()��ȡֵ��ΧΪ0~RAND_MAX
			p[j] = randnum*sqrt(6.0/(inputNum+outputNum));
			//p[j] = randf(-0.05, 0.05);
		}
	}

	return outL;
}


//1��n�е�����
int vecmaxIndex(Mat vec)  //������������������
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

// ����ת180��
Mat rotate180(Mat mat)
{
#if 0
	int outSizeW = mat.cols;   //��
	int outSizeH = mat.rows;   //��

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
	flip(mat, tmp, -1);// ��תģʽ��flipCode == 0��ֱ��ת����X�ᷭת����flipCode>0ˮƽ��ת����Y�ᷭת����flipCode<0ˮƽ��ֱ��ת������X�ᷭת������Y�ᷭת���ȼ�����ת180�㣩
	return tmp;
#endif
}



#if 1

// ���ھ������ز��������ѡ��
// ���ﹲ������ѡ��full��same��valid���ֱ��ʾ
// fullָ��ȫ�����������Ĵ�СΪinSize+(mapSize-1)
// sameָͬ������ͬ��С
// validָ��ȫ������Ĵ�С��һ��ΪinSize-(mapSize-1)��С���䲻��Ҫ��������0����

Mat correlation(Mat map, Mat inputData, int type) // �����
{
	// ����Ļ�������ں��򴫲�ʱ���ã������ڽ�Map��ת180���پ��
	// Ϊ�˷�����㣬�����Ƚ�ͼ������һȦ
	// ����ľ��Ҫ�ֳ�������ż��ģ��ͬ����ģ��

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


	//fullģʽ
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

// ����ά�����Ե��������addw��С��0ֵ��
Mat matEdgeExpand(Mat mat, int addc, int addr)
{ // ������Ե����
	int i, j;
	int c = mat.cols;
	int r = mat.rows;

	//float** res = (float**)malloc((r + 2 * addr) * sizeof(float*)); // ����ĳ�ʼ��
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
				//res[j][i] = mat[j - addr][i - addc]; // ����ԭ����������
				p[i] = mat_p[i - addc];

		}
	}
	return res;
}


// ����ά�����Ե��С������shrinkc��С�ı�
Mat matEdgeShrink(Mat mat, int shrinkc, int shrinkr)
{ // ��������С������Сaddw������Сaddh
	int i, j;
	int c = mat.cols;
	int r = mat.rows;
	//float** res = (float**)malloc((r - 2 * shrinkr) * sizeof(float*)); // �������ĳ�ʼ��
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
				//res[j - shrinkr][i - shrinkc] = mat[j][i]; // ����ԭ����������
				p[i - shrinkc] = mat_p[i];
		}
	}
	return res;
}


Mat correlation(Mat map, Mat inputData, int type)// �����
{
	// ����Ļ�������ں��򴫲�ʱ���ã������ڽ�Map��ת180���پ��
	// Ϊ�˷�����㣬�����Ƚ�ͼ������һȦ
	// ����ľ��Ҫ�ֳ�������ż��ģ��ͬ����ģ��
	int i, j, c, r;
	int halfmapsizew;
	int halfmapsizeh;
	if (map.rows% 2 == 0 && map.cols%2 == 0)
	{ // ģ���СΪż��
		halfmapsizew = map.cols / 2; // ���ģ��İ���С
		halfmapsizeh = map.rows / 2;
	}
	else
	{
		halfmapsizew = (map.cols - 1) / 2; // ���ģ��İ���С
		halfmapsizeh = (map.rows - 1) / 2;
	}

	// ������Ĭ�Ͻ���fullģʽ�Ĳ�����fullģʽ�������СΪinSize+(mapSize-1)
	int outSizeW = inputData.cols + (map.cols - 1); // ������������һ����
	int outSizeH = inputData.rows + (map.rows - 1);
	Mat outputData = Mat::zeros(outSizeH, outSizeW, CV_32FC1);
	/*float** outputData = (float**)malloc(outSizeH * sizeof(float*)); // ����صĽ��������
	for (i = 0; i<outSizeH; i++)
	{
		outputData[i] = (float*)calloc(outSizeW, sizeof(float));   //�����ά����
	}*/

	// Ϊ�˷�����㣬��inputData����һȦ
	//float** exInputData = matEdgeExpand(inputData, inSize, mapSize.c - 1, mapSize.r - 1);
	Mat exInputData = matEdgeExpand(inputData, map.cols - 1, map.rows - 1);

	for (j = 0; j < outSizeH; j++)  //Ĭ�Ͼ���˵Ļ�������Ϊ1
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
					//outputData[j][i] = outputData[j][i] + map[r][c] * exInputData[j + r][i + c];   //��������������ÿ��λ�õ�ֵ��ˣ�����ۼ�
					outputData_p[i] = outputData_p[i] + map_p[c] * exInputData_p[i + c];
				}
			}
		}
	}

	//for (i = 0; i<inSize.r + 2 * (mapSize.r - 1); i++)   //�ͷŶ�ά�����ڴ�
	//	free(exInputData[i]);
	//free(exInputData);

	//nSize outSize = { outSizeW,outSizeH };
	switch (type)   //���ݲ�ͬ����������ز�ͬ�Ľ��
	{
	case full: //��ȫ��С�����
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



// �������
Mat cov(Mat map, Mat inputData, int type) 
{
	// ���������������ת180�ȵ�����ģ���������
	//Mat flipmap = rotate180(map); //��ת180�ȵ�����ģ��
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


void avgPooling(Mat input, Mat &output, int mapSize) // ��ƽ��ֵ
{
	const int outputW = input.cols/mapSize;   //�����=�����/�˿�
	const int outputH = input.rows/mapSize;   //�����=�����/�˸�
	float len = (float)(mapSize*mapSize);
	int i,j,m,n;
	for(i = 0;i < outputH; i++)
	{
		for(j = 0; j < outputW; j++)
		{
			float sum=0.0;
			for(m = i*mapSize; m < i*mapSize+mapSize; m++)  //ȡ����˴�С�Ĵ������ƽ��
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


//���ֵ�ػ�
void maxPooling(Mat input, Mat &max_position, Mat &output, int mapSize)
{
	int outputW = input.cols / mapSize;   //�����=�����/�˿�
	int outputH = input.rows / mapSize;   //�����=�����/�˸�

	int i, j, m, n;
	for (i = 0; i < outputH; i++)
	{
		for (j = 0; j < outputW; j++)
		{
			float max = -999999.0;
			int max_index = 0;

			for (m = i*mapSize; m<i*mapSize + mapSize; m++)  //ȡ����˴�С�Ĵ��ڵ����ֵ
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
0, x �� 0
y = x, 0 < x < 6
6, x �� 6
*/

// ����� input�����ݣ�inputNum˵��������Ŀ��bas����ƫ��
float activation_Sigma(float input, float bas) // sigma�����
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
			return x;      //dy/dx = 1/A1, -6.0<y��0.0
	}*/
#else
	float temp = input + bas;
	return (float)1.0 / ((float)(1.0 + exp(-temp)));
#endif
}


// ����ȫ�����������ǰ�򴫲�
float vecMulti(Mat vec1, float *vec2)// ���������
{
	float *p1 = vec1.ptr<float>(0);
	float m = 0;
	for (int i = 0; i < vec1.cols; i++)
		m = m + p1[i] * vec2[i];
	return m;
}

void nnff(Mat input, Mat wdata, Mat &output)
{
	for (int i = 0; i < output.cols; i++)  //�ֱ������������˵ĳ˻�
		output.ptr<float>(0)[i] = vecMulti(input, wdata.ptr<float>(i));   //�������뼤���֮ǰ���м���ƫ�õĲ��������Դ˴����ټ�ƫ��
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
			//Mat mapout = cov(C.mapData[j][i], inputData[j], cov_type);   //��������mapDataΪ��ά����
			Mat mapout = correlation(C.mapData[j][i], inputData[j], cov_type);
			C.v[i] += mapout;           //��������ͨ���ľ������ۼ�
		}

		int output_r = C.y[i].rows;
		int output_c = C.y[i].cols;
		for (int r = 0; r < output_r; r++)
		{
			for (int c = 0; c < output_c; c++)
			{
				C.y[i].ptr<float>(r)[c] = activation_Sigma(C.v[i].ptr<float>(r)[c], C.basicData.ptr<float>(0)[i]);   //�ȼ���ƫ�ã�������sigmoid�����
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
	Mat OinData(1, O.inputNum, CV_32FC1);   //����192ͨ��
	float *OinData_p = OinData.ptr<float>(0);
	int outsize_r = inputData[0].rows;
	int outsize_c = inputData[0].cols;
	int last_output_len = inputData.size();
	for (int i = 0; i < last_output_len; i++)   //��һ��S4���12ͨ����4*4����
	{
		for (int r = 0; r < outsize_r; r++)
		{
			for (int c = 0; c < outsize_c; c++)
			{
				OinData_p[i*outsize_r*outsize_c + r*outsize_c + c] = inputData[i].ptr<float>(r)[c];  //��12ͨ��4*4����һάչ��
			}
		}
	}

	//192*10��Ȩ��
	nnff(OinData, O.wData, O.v);   //10ͨ�����,1��ͨ�����������192������ֱ���192��Ȩ����˵ĺͣ���in[i]*w[i], 0��i<192

	//for (int i = 0; i < O.outputNum; i++)
	//	O.y.ptr<float>(0)[i] = activation_Sigma(O.v.ptr<float>(0)[i], O.basicData.ptr<float>(0)[i]);

	softmax(O);
}


void cnnff(CNN &cnn, Mat inputData)
{
	// ��һ��Ĵ���
	//5*5�����
	//����28*28����
	//���(28-25+1)*(28-25+1) = 24*24����
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
{ // Logic��������Ա���΢��

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
	return y*(1 - y); // ����y��ָ��������������ֵ���������Ա���
#endif
}


/*
�����ϲ�����upc��upr���ڲ屶��
��������ֵ�ػ�ģʽ����Ѿ����ݶȷŵ��ػ�ǰ���ֵ��λ�ã�����ػ�����2*2���ػ�ǰ���ֵ��λ�÷ֱ�Ϊ���ϡ����ϡ����¡����£����ϲ�����Ϊ��
5 9				5 0 0 9
     -->		0 0 0 0
3 6				0 0 0 0
                3 0 0 6
����Ǿ�ֵ�ػ�ģʽ����Ѿ����ݶȳ��Գػ����ڵĳߴ�2*2=4:
5 9				1.25 1.25 2.25 2.25
     -->		1.25 1.25 2.25 2.25
3 6				0.75 0.75 1.5  1.5
                0.75 0.75 1.5  1.5
*/
Mat UpSample(Mat mat, int upc, int upr)   //��ֵ�ػ�������ϲ���
{
	//int i, j, m, n;
	int c = mat.cols;
	int r = mat.rows;
	
	Mat res(r*upr, c*upc, CV_32FC1);

	float pooling_size = 1.0 / (upc*upr);

	for (int j = 0; j < r*upr; j += upr)
	{
		for (int i = 0; i < c*upc; i += upc)  // �������
		{
			for (int m = 0; m < upc; m++)
			{

				//res[j][i + m] = mat[j / upr][i / upc] * pooling_size;
				res.ptr<float>(j)[i + m] = mat.ptr<float>(j/upr)[i/upc] * pooling_size;
			}
		}

		for (int n = 1; n < upr; n++)      //  �ߵ�����
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


//���ֵ�ػ�������ϲ���
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
			int index_r = max_position.ptr<int>(j)[i] / outsize_c;   //�������ֵ������
			int index_c = max_position.ptr<int>(j)[i] % outsize_c;
			res.ptr<float>(index_r)[index_c] = mat.ptr<float>(j)[i];
		}
	}
	return res;
}





//���������-->ȫ���Ӳ�
void softmax_bp(Mat outputData, Mat &e, OutLayer &O)
{
	for (int i = 0; i < O.outputNum; i++)
		e.ptr<float>(0)[i] = O.y.ptr<float>(0)[i] - outputData.ptr<float>(0)[i];   //��������1/2��(di-yi)^2��yi��ƫ����,�õ�yi-di

	/*�Ӻ���ǰ�������*/																				
	for (int i = 0; i < O.outputNum; i++)
		O.d.ptr<float>(0)[i] = e.ptr<float>(0)[i];// *sigma_derivation(O.y.ptr<float>(0)[i]);
}


//����ȫ���Ӳ�-->�ػ���
void full2pool_bp(OutLayer O, PoolLayer &S)
{
	int outSize_r = S.inputHeight / S.mapSize;
	int outSize_c = S.inputWidth / S.mapSize;
	for (int i = 0; i < S.outChannels; i++)  //���12��4*4ͼ��
	{
		for (int r = 0; r < outSize_r; r++)
		{
			for (int c = 0; c < outSize_c; c++)
			{
				int wInt = i*outSize_c*outSize_r + r*outSize_c + c;  //i*outSize.c*outSize.rΪͼ��������r*outSize.c+cΪÿ��ͼ���е���������
				for (int j = 0; j < O.outputNum; j++)   //O5�������������
				{
					S.d[i].ptr<float>(r)[c] = S.d[i].ptr<float>(r)[c] + O.d.ptr<float>(0)[j] * O.wData.ptr<float>(j)[wInt];  //d_S4 = ��d_O5*W
				}
				//S.d[i].ptr<float>(r)[c] /= 10.0;
			}
		}
	}
}


//���򣺳ػ���-->�����
void pool2cov_bp(PoolLayer S, CovLayer &C)
{

	for (int i = 0; i < C.outChannels; i++)   //12ͨ��
	{
		Mat C3e;
		if (S.poolType == AvePool)
			C3e = UpSample(S.d[i], S.mapSize, S.mapSize);    //���ϲ�������S4��ľ����ݶ���4*4����Ϊ8*8
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


//���򣺾����-->�ػ���
void cov2pool_bp(CovLayer C, int cov_type, PoolLayer &S)
{

	for (int i = 0; i < S.outChannels; i++)   //S2��6ͨ��
	{
		for (int j = 0; j < S.inChannels; j++)  //C3��12ͨ��
		{
			//Mat tmp;
			//flip(C.mapData[i][j], flipmap, -1);
			//Mat corr = correlation(C.mapData[i][j], C.d[j], cov_type);   //���㻥���,�õ�12*12����fullģʽ��Ϊ(inSize+mapSize-1)*(inSize+mapSize-1)
		    Mat corr = cov(C.mapData[i][j], C.d[j], cov_type);
			S.d[i] = S.d[i] + corr;   //�����ۼӣ�cnn->S2->d[i] = cnn->S2->d[i] + corr���õ�6��12*12�����ݶ�
		}
		//S.d[i] /= S.inChannels;
	}
}



void cnnbp(CNN &cnn, Mat outputData) // ����ĺ��򴫲�
{
	softmax_bp(outputData, cnn.e, cnn.O5);
	full2pool_bp(cnn.O5, cnn.S4);
	pool2cov_bp(cnn.S4, cnn.C3);
	cov2pool_bp(cnn.C3, full, cnn.S2);
	pool2cov_bp(cnn.S2, cnn.C1);
}




void update_cov_para(vector<Mat> inputData, CNNOpts opts, CovLayer &C)
{
	for (int i = 0; i < C.outChannels; i++)   //6ͨ��
	{
		for (int j = 0; j < C.inChannels; j++)   //1ͨ��
		{
			//Mat flipinputData = rotate180(inputData[j]);   //����ת180��      
			//Mat Cdk = cov(C.d[i], flipinputData, valid);   //�����*X��validģʽ�����(ySize-dSize+1)*(ySize-dSize+1) = 5*5����
			Mat Cdk = correlation(C.d[i], inputData[j], valid);
			Cdk = Cdk*(-opts.alpha);   //�������ϵ��-���õ�-��*��*X
			C.mapData[j][i] = C.mapData[j][i] + Cdk;   //����W[n+1] = W[n] - ��*��*X
		}

		float d_sum = (float)cv::sum(C.d[i])[0];   //������6��24*24��d��6��ƫ��b��һ��ƫ��b��Ӧһ��24*24����d������Ԫ�غ�
		C.basicData.ptr<float>(0)[i] = C.basicData.ptr<float>(0)[i] - opts.alpha*d_sum;
	}
}


void update_full_para(vector<Mat> inputData, CNNOpts opts, OutLayer &O)
{
	int outSize_r = inputData[0].rows;
	int outSize_c = inputData[0].cols;
	Mat OinData(1, outSize_r*outSize_c*inputData.size(), CV_32FC1);
	for (int i = 0; i < inputData.size(); i++)   //12ͨ��
	{
		for (int r = 0; r < outSize_r; r++)  //4
		{
			for (int c = 0; c < outSize_c; c++)   //4
			{
				OinData.ptr<float>(0)[i*outSize_r*outSize_c + r*outSize_c + c] = inputData[i].ptr<float>(r)[c];
			}
		}
	}

	for (int j = 0; j < O.outputNum; j++)  //10ͨ��
	{
		for (int i = 0; i < O.inputNum; i++)  //192ͨ��
		{
			O.wData.ptr<float>(j)[i] = O.wData.ptr<float>(j)[i] - opts.alpha*O.d.ptr<float>(0)[j] * OinData.ptr<float>(0)[i];
		}

		O.basicData.ptr<float>(0)[j] = O.basicData.ptr<float>(0)[j] - opts.alpha*O.d.ptr<float>(0)[j];
	}
}

void cnnapplygrads(CNN &cnn, CNNOpts opts, Mat inputData) // ����Ȩ��
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




void cnnsetup(CNN &cnn, int inputSize_r, int inputSize_c, int outputSize)   //cnn��ʼ��
{
	cnn.layerNum = 5;

	//nSize inSize;
	int mapSize = 5;
	int inSize_c = inputSize_c;   //28
	int inSize_r = inputSize_r;   //28
	int C1_outChannels = 6;
	cnn.C1 = initCovLayer(inSize_c, inSize_r, mapSize, 1, C1_outChannels);   //�����1

	inSize_c = inSize_c - cnn.C1.mapSize + 1;  //24
	inSize_r = inSize_r - cnn.C1.mapSize + 1;  //24
	mapSize = 2;
	cnn.S2 = initPoolLayer(inSize_c, inSize_r, mapSize, cnn.C1.outChannels, cnn.C1.outChannels, MaxPool);   //�ػ���

	inSize_c = inSize_c / cnn.S2.mapSize;   //12
	inSize_r = inSize_r / cnn.S2.mapSize;   //12
	mapSize = 5;
	int C3_outChannes = 12;
	cnn.C3 = initCovLayer(inSize_c, inSize_r, mapSize, cnn.S2.outChannels, C3_outChannes);   //�����

	inSize_c = inSize_c - cnn.C3.mapSize + 1;   //8
	inSize_r = inSize_r - cnn.C3.mapSize + 1;   //8
	mapSize = 2;
	cnn.S4 = initPoolLayer(inSize_c, inSize_r, mapSize, cnn.C3.outChannels, cnn.C3.outChannels, MaxPool);    //�ػ���

	inSize_c = inSize_c / cnn.S4.mapSize;   //4
	inSize_r = inSize_r / cnn.S4.mapSize;   //4
	cnn.O5 = initOutLayer(inSize_c*inSize_r*cnn.S4.outChannels, outputSize);    //�����

	cnn.e = Mat::zeros(1, cnn.O5.outputNum, CV_32FC1);   //���������ֵ���ǩֵ֮��
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
	// ѧϰѵ���������
	//cnn->L = (float*)malloc(trainNum * sizeof(float));
	cnn.L = Mat(1, trainNum, CV_32FC1).clone();
	for (int e = 0;  e < opts.numepochs; e++)
	{
		for (int n = 0; n < trainNum; n++)
		{
			#ifndef RELU_USE
			opts.alpha = 0.1 - 0.075*n / (trainNum - 1);    //ѧϰ�ʵݼ�2.0~0.25
			#else
			opts.alpha = 0.03 - 0.029*n / (trainNum - 1);    //ѧϰ�ʵݼ�0.01~0.001
			//opts.alpha = 0.01 -0.009*n / (trainNum - 1);
			#endif

														
			cnnff(cnn, inputData[n]);  // ǰ�򴫲���������Ҫ�����
			cnnbp(cnn, outputData[n]); // ���򴫲���������Ҫ�������Ԫ������ݶ�
			cnnapplygrads(cnn, opts, inputData[n]); // ����Ȩ��

													// ���㲢�����������
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

			printf("n=%d, f=%f, ��=%f\n", n, cnn.L.ptr<float>(0)[n], opts.alpha);
		}
	}

	//write_LL();
}




// ����cnn����
float cnntest(CNN cnn, vector<Mat> inputData, vector<Mat> outputData)
{
	int incorrectnum = 0;  //����Ԥ�����Ŀ
	for (int i = 0; i < inputData.size(); i++)
	{
		cnnff(cnn, inputData[i]);

		if (vecmaxIndex(cnn.O5.y) != vecmaxIndex(outputData[i]))
		{
			incorrectnum++;
			printf("i = %d, ʶ��ʧ��\n", i);
		}
		else
		{
			printf("i = %d, ʶ��ɹ�\n", i);
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
	
	//����data_batch_1.bin��ѵ������
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

	//����data_batch_2.bin��ѵ������
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

	//����data_batch_3.bin��ѵ������
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

	//����data_batch_4.bin��ѵ������
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

	//����data_batch_5.bin��ѵ������
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
	//��test_batch.bin�е�ǰ800��ͼ�����Ԥ�����
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
			traindata.push_back(srcimage.clone());   //���������뵽ѵ��������
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
			testdata_list.push_back(testdata.clone());   //���������뵽ѵ��������
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
	cnnsetup(cnn, 28, 28, 10);   //cnn��ʼ��
	printf("***********************ccnntrain\n");
	cnntrain(cnn, traindata, trainlabel, opts, traindata.size());
	printf("***********************finish\n");

	float success = cnntest(cnn, testdata_list, testdata_label);
	printf("success=%f\n", 1-success);
}




vector<Mat> read_Img_to_Mat(const char* filename) // ����ͼ��
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
	
	fread(&magic_number, sizeof(int), 1, fp);   //���ļ��ж�ȡsizeof(int) ���ַ��� &magic_number  
	magic_number = ReverseInt(magic_number);

	//printf("magic_number=%d\n", magic_number);
	//while (1);
	
	fread(&number_of_images, sizeof(int), 1, fp);   //��ȡѵ�������image�ĸ���number_of_images 
	number_of_images = ReverseInt(number_of_images);
	
	fread(&n_rows, sizeof(int), 1, fp);   //��ȡѵ�������ͼ��ĸ߶�Heigh  
	n_rows = ReverseInt(n_rows);
	
	fread(&n_cols, sizeof(int), 1, fp);   //��ȡѵ�������ͼ��Ŀ��Width  
	n_cols = ReverseInt(n_cols);

	//printf("magic_number=%d, number_of_images=%d, n_rows=%d, n_cols=%d\n", magic_number, number_of_images, n_rows, n_cols);


	//��ȡ��i��ͼ�񣬱��浽vec�� 
	int i, r, c;

	// ͼ������ĳ�ʼ��
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




vector<Mat> read_Lable_to_Mat(const char* filename)// ����ͼ��
{
	FILE  *fp = NULL;
	fp = fopen(filename, "rb");
	if (fp == NULL)
		printf("open file failed\n");
	assert(fp);

	int magic_number = 0;
	int number_of_labels = 0;
	int label_long = 10;

	
	fread(&magic_number, sizeof(int), 1, fp);   //���ļ��ж�ȡsizeof(magic_number) ���ַ��� &magic_number  
	magic_number = ReverseInt(magic_number);

	//printf("magic_number=%d\n", magic_number);
	//while (1);
	
	fread(&number_of_labels, sizeof(int), 1, fp);   //��ȡѵ�������image�ĸ���number_of_images 
	number_of_labels = ReverseInt(number_of_labels);

	int i, l;

	// ͼ��������ĳ�ʼ��
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
