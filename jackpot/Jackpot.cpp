#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <highgui.h>
#include "include/thinning.cpp"
using namespace cv;


/// Global Variables
Mat src;
int threshold_slider;
int threshold_slider_dist;

//récupère le foreground ( filtre +gaussian  + fermé morpho )
//threshold sur la composante verte
void extractForeground(Mat& img,Mat& dst, int threshold)
{	

	//Le threshold qui va bien = 107
	//on extrait la composante verte: 
	Mat ch1, ch2, ch3;
	vector<Mat> channels(3);
	split(img, channels);
	ch1 = channels[0];
	ch2 = channels[1];
	ch3 = channels[2];

//application du seuil sur la composante verte: 
	// cv::GaussianBlur(ch2, ch2, Size(3,3), 0, 0, 0);
	cv::threshold(ch2, dst, threshold, 255, cv::THRESH_BINARY); 
	cv::bitwise_not(dst,dst);
//on smooth tout ça ( pour la segmentation):
	cv::GaussianBlur(dst, dst, Size(3,3), 0, 0, 0);

//on ferme les trous ( pour la segmentation toujours)
	Mat eltstruct1=  getStructuringElement(MORPH_CROSS, Size(5,5),Point(-1,-1));
	Mat eltstruct2=  getStructuringElement(MORPH_ELLIPSE, Size(5,5),Point(-1,-1));
	morphologyEx(dst, dst,MORPH_CLOSE, eltstruct1, Point(-1,-1), 2, BORDER_CONSTANT);	
	morphologyEx(dst, dst,MORPH_CLOSE, eltstruct2, Point(-1,-1), 2, BORDER_CONSTANT);	


}
void on_trackbar_dist( int, void* )
{
	//on récupère le foreground:
	Mat foreground;
	extractForeground(src,foreground, threshold_slider);
	imshow( "foreground", foreground );
	
	//on squeletise: 
	Mat zuhang;
	thinning(foreground, zuhang);
	imshow( "zuhang", zuhang );
	std::cout << "zuhang size: "<< zuhang.size() << "zuhang type: "<< zuhang.type() << std::endl;
	Mat temp = src.clone();
	//on affiche le squelette sur l'image en couleurs:
	for (int i = 0; i < src.cols; ++i)
	{
		for (int j = 0; j < src.rows; ++j)
		{
			if(zuhang.at<uchar>(Point(i,j))>0)
			{	
				temp.at<cv::Vec3b>(Point(i,j))[0]=255;//sign_x.at<float>(Point(i,j));
				temp.at<cv::Vec3b>(Point(i,j))[1]=255;//sign_x.at<float>(Point(i,j));
				temp.at<cv::Vec3b>(Point(i,j))[2]=255;//sign_x.at<float>(Point(i,j));
			}

		}

	}
	imshow( "resultat", temp );
}

int main(int argc, char** argv )
{
	if ( argc != 2 )
	{
		printf("usage: DisplayImage.out <Image_Path>\n");
		return -1;
	}
	src = imread( argv[1], 1 );

	if ( !src.data )
	{
		printf("No image data \n");
		return -1;
	}
	namedWindow("foreground", WINDOW_NORMAL);

/******************************************************
	interface
*******************************************************/

/// Create Trackbars
	char TrackbarName[50];
	sprintf( TrackbarName, "seuil squeletisation %d", 255 );
	createTrackbar( TrackbarName, "foreground", &threshold_slider, 255, on_trackbar_dist );
	setTrackbarPos(TrackbarName, "foreground",150);

	waitKey(0);

	return 0;
}