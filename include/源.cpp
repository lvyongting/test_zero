
#include<opencv2\opencv.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
using namespace std;
using namespace cv;
/*#pragma comment(lib,"opencv_core2410d.lib")                
#pragma comment(lib,"opencv_highgui2410d.lib")                
#pragma comment(lib,"opencv_objdetect2410d.lib")   
#pragma comment(lib,"opencv_imgproc2410d.lib")  
/* Function Headers */
void detectAndDisplay(Mat frame);
/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
Mat dst10;

String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);

const int FRAME_WIDTH = 1280;
const int FRAME_HEIGHT = 240;



int main(void)
{
	
	if (!face_cascade.load(face_cascade_name)) 	{ printf("--(!)Error loading\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name))  { printf("--(!)Error loading\n"); return -1; };

	


	VideoCapture cap(0); //打开默认的摄像头号
	if (!cap.isOpened())  //检测是否打开成功
		return -1;

	Mat edges;
	//namedWindow("edges",1);
	for (;;)
	{
		Mat frame;
		cap >> frame; // 从摄像头中获取新的一帧
		resize(frame, dst10, Size(), 0.5, 0.5, INTER_LINEAR);
		detectAndDisplay(dst10);
		//imshow("edges", frame);
		if (waitKey(30) >= 0) break;
	}
	//摄像头会在VideoCapture的析构函数中释放
	waitKey(0);

	return 0;
}


void mapToMat(const cv::Mat &srcAlpha, cv::Mat &dest, int x, int y)
{
	int nc = 3;
	int alpha = 0;

	for (int j = 0; j < srcAlpha.rows; j++)
	{
		for (int i = 0; i < srcAlpha.cols * 3; i += 3)
		{
			alpha = srcAlpha.ptr<uchar>(j)[i / 3 * 4 + 3];
			//alpha = 255-alpha;
			if (alpha != 0) //4通道图像的alpha判断
			{
				for (int k = 0; k < 3; k++)
				{
					// if (src1.ptr<uchar>(j)[i / nc*nc + k] != 0)
					if ((j + y < dest.rows) && (j + y >= 0) &&
						((i + x * 3) / 3 * 3 + k < dest.cols * 3) && ((i + x * 3) / 3 * 3 + k >= 0) &&
						(i / nc * 4 + k < srcAlpha.cols * 4) && (i / nc * 4 + k >= 0))
					{
						dest.ptr<uchar>(j + y)[(i + x*nc) / nc*nc + k] = srcAlpha.ptr<uchar>(j)[(i) / nc * 4 + k];
					}
				}
			}
		}
	}
}

/**
* @function detectAndDisplay
*/
void detectAndDisplay(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;
	Mat hatAlpha;

	hatAlpha = imread("2.png", -1);//圣诞帽的图片

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 , Size(30, 30));

	for (size_t i = 0; i < faces.size(); i++)
	{

		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		// ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar( 255, 0, 255 ), 2, 8, 0 );

		// line(frame,Point(faces[i].x,faces[i].y),center,Scalar(255,0,0),5);

		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;

		//-- In each face, detect eyes
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30));

		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			// circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 3, 8, 0 );
		}

		// if(eyes.size())
		
			resize(hatAlpha, hatAlpha, Size(faces[i].width, faces[i].height), 0, 0, INTER_LANCZOS4);
			// mapToMat(hatAlpha,frame,center.x+2.5*faces[i].width,center.y-1.3*faces[i].height);
			mapToMat(hatAlpha, frame, faces[i].x, faces[i].y - 0.8*faces[i].height);
		
	}
	//-- Show what you got
	imshow(window_name, frame);
	imwrite("merry christmas.jpg", frame);
}

