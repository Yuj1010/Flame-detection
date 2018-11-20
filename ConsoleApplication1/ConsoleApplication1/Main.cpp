#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

int redThre = 49;//红色分量阈值
int saturationTh = 7;//饱和度阈值
#define threshold_diff1 25 //帧差法阈值

using namespace cv;
using namespace std;


//二帧差分法 动态检测
Mat frameDiff(Mat prevFrame, Mat nextFrame){
	Mat diffFrame1, diffFrame2, output;
	output.create(prevFrame.size(), CV_8UC1);
	absdiff(nextFrame, prevFrame, diffFrame1);
	for (int i = 0; i < diffFrame1.rows; i++){
		for (int j = 0; j < diffFrame1.cols; j++){
			if (abs(diffFrame1.at<uchar>(i, j)) >= threshold_diff1)
				diffFrame1.at<uchar>(i, j) = 255;
			else
				diffFrame1.at<uchar>(i, j) = 0;
		}
	}
	bitwise_and(diffFrame1, diffFrame1, output);
	medianBlur(output, output, 5);//中值滤波
	//dilate(output, output, Mat(5, 5, CV_8UC1));//形态学处理8
	//erode(output, output, Mat(5, 5, CV_8UC1));
	return output;
}


//灰度处理
Mat setGray(Mat frame){
	cvtColor(frame, frame, CV_BGR2GRAY);
	Sobel(frame, frame, CV_8U, 1, 0, 3, 0.4, 128);//边缘检测算子
	return frame;
}


//获取视频一帧图像
Mat getFrame(VideoCapture video){
	Mat frame;
	video >> frame;
	return frame;
}


//RGB & HSI颜色检测
Mat CheckColor(Mat inImg, Mat grayFrame){
	Mat fireImg;
	fireImg.create(inImg.size(), CV_8UC1);
	Mat multiRGB[3];
	split(inImg, multiRGB);//将原图像拆分成R、G、B三个通道
	for (int i = 0; i < inImg.rows; i++){
		for (int j = 0; j < inImg.cols; j++){
			float B, G, R;
			B = multiRGB[0].at<uchar>(i, j);
			G = multiRGB[1].at<uchar>(i, j);
			R = multiRGB[2].at<uchar>(i, j);

			float maxValue = max(max(B, G), R);
			float minValue = min(min(B, G), R);
			//计算HSI中的S
			double S = (1 - 3.0*minValue / (R + G + B));
			//颜色检测判据
			if (!(R > redThre&&R >= G&&G >= B&&S > ((255 - R)*saturationTh / redThre)) && (grayFrame.at<uchar>(i, j) == 255)){
				grayFrame.at<uchar>(i, j) = 0;
			}
		}
	}
	//dilate(grayFrame, grayFrame, Mat(5, 5, CV_8UC1));
	imshow("fire", grayFrame);
	return grayFrame;
}


//主函数
int main(int argc, unsigned char* argv[])
{
	VideoCapture capture1("test1.avi");//读取视频资源
	VideoCapture capture2("test1.avi");
	Mat prevFrame, nextFrame;
	prevFrame = getFrame(capture1);//获取第一帧图像
	prevFrame = setGray(prevFrame);
	nextFrame = getFrame(capture1);//获取第二帧图像
	nextFrame = setGray(nextFrame);
	while (1){
		Mat frame;
		capture2 >> frame;
		if (frame.empty())
			break;
		imshow("org", frame);
		CheckColor(frame, frameDiff(prevFrame, nextFrame));
		//imshow("dyn", frameDiff(prevFrame,nextFrame));

		prevFrame = nextFrame;
		nextFrame = getFrame(capture1);
		if (nextFrame.empty())
			break;
		nextFrame = setGray(nextFrame);
		waitKey(1);
	}
	return 0;
}
