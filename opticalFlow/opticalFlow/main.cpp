#include<iostream>
#include<vector>
#include<numeric>
#include<algorithm>
#include<iostream>
#include<vector>
#include<opencv2/opencv.hpp>
#include<opencv2/video/tracking.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<time.h>
using namespace cv;
using namespace std;
using namespace cv::cuda;


//��ȡ��Ƶ·��
string path = "D://FFOutput//ԭ��Ƶ//20190919_082450//VID_20190919_082450��������ָ.mp4";
//����ͼ��֡·��
string imagePath = "D://FFOutput//frames//3//";
//�������֡·��
//string flowPath = "E://���ݼ�//pdflow//2//";
//�������֡x·��
string xPath = "D://FFOutput//flow//3//";
//�������֡y·��
string yPath = "D://FFOutput//flow//3//";


void convertFlowToImage(const Mat &flow, Mat &img_x, Mat &img_y, double lowerBound, double higherBound)
{
	//cvRound()���������������뺯��
	//int x = (v) < (L) ? 0 : cvRound(255*((v) - (L))
	//���v����L��v-L����255���������루ȡ����
	//int y = (v) > (H) ? 255 : x
	//���vС��H����ȡx,����ȡ255
	//����Ϊv,L,H�Ĺ�ϵ
#define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))

	for (int i = 0; i < img_x.rows; ++i) {
		for (int j = 0; j < img_x.cols; ++j) {
			img_x.at<uchar>(i, j) = CAST(flow.at<Point2f>(i, j).x, lowerBound, higherBound);
			img_y.at<uchar>(i, j) = CAST(flow.at<Point2f>(i, j).x, lowerBound, higherBound);
		}
	}

#undef CAST
}

//��Ƶ��Ԥ�������ͷֱ��ʣ��ü���
void resize(Mat& src)
{
	//resize(src, src, cv::Size(720, 1280), INTER_AREA);
	Rect rect(600, 180, 640, 480);   //��������Ͻǵ������Ϊ��10,10��							   //�����Ϊ150����Ϊ100
	src = src(rect);
	return;
}


int main(int argc, char * argv[]) {
	vector<Mat> flow;
	Mat prev, curr, frame;
	cv::Ptr<cv::DualTVL1OpticalFlow> tvl1 = cv::DualTVL1OpticalFlow::create();

	VideoCapture  capture(path);
	if (!capture.isOpened()) {
		cout << "Read video failed" << endl;
		return -1;
	}

	capture.read(frame);
	resize(frame);
	cvtColor(frame, prev, CV_BGR2GRAY);//ת��ֵͼ
	cv::imwrite(imagePath + "img_00001.jpg", frame);//
	int h = frame.rows;
	int w = frame.cols;

	clock_t begin, end;
	begin = clock();//��ʱ
	int frameNum = 2;//ȡ��֡�����������
					 //������Ƶ���Ĳ�������һ����������������Ƶ�ĵڼ���������
					 //CAP_PROP_POS_FRAMES�����ò����֡�Ļ���0����������Ϊ2�����ӵڶ�֡��ʼ��ȡ
	capture.set(CAP_PROP_POS_FRAMES, frameNum);
	//��ȡ��Ƶ֡��
	int count = capture.get(CAP_PROP_FRAME_COUNT);
	int fps = capture.get(5);	//CV_CAP_PROP_FPS ֡����
	string str, strflow;
	int num = 2, flownum = 1;
	while (num < count) {
		Mat d_flow;
		Mat out;

		capture >> frame;
		resize(frame);
		cvtColor(frame, curr, CV_BGR2GRAY);
		imshow("sur", frame);
		stringstream ss;
		ss << setw(5) << setfill('0') << num;
		str = ss.str();
		cv::imwrite(imagePath + "img_" + str + ".jpg", frame);//����֡ͼ��
															  //�������
		tvl1->calc(prev, curr, d_flow);
		Mat img_x(h, w, CV_8UC1);
		Mat img_y(h, w, CV_8UC1);


		stringstream sflow;
		sflow << setw(5) << setfill('0') << flownum;
		strflow = sflow.str();
		convertFlowToImage(d_flow, img_x, img_y, -15, 15);
		cv::imwrite(yPath + "flow_y_" + strflow + ".jpg", img_y);
		cv::imwrite(xPath + "flow_x_" + strflow + ".jpg", img_x);

		prev = curr.clone();
		num++;
		flownum++;
		waitKey(30);
	}

	end = clock();
	std::cout << "total frames: " << num << endl;
	std::cout << "time used: " << (double)(end - begin) / CLOCKS_PER_SEC << endl;
	return 0;
}



