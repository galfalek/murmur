#include <cv.h>
#include <highgui.h>

using namespace cv;

/// Global Variables
const int alpha_slider_max = 100;
int alpha_slider;
double alpha;
double beta;

/// Matrices to store images
Mat src1;
Mat src2;
Mat dst;

/**
 * @function on_trackbar
 * @brief Callback for trackbar
 */
 void on_trackbar( int, void* )
 {
 	alpha = (double) alpha_slider/alpha_slider_max ;
 	beta = ( 1.0 - alpha );

 	addWeighted( src1, alpha, src2, beta, 0.0, dst);

 	imshow( "Linear Blend", dst );
 }
 void watershed(Mat& src, Mat& dst)
 {
 	

    // Perform the distance transform algorithm
 	Mat dist;
    //distanceTransform(src, dist, CV_DIST_L2, 3);
 	distanceTransform(src, dist, CV_DIST_L2, 3);
    // Normalize the distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
 	normalize(dist, dist, 0, 1., NORM_MINMAX);
 	imshow("Distance Transform Image", dist);
    // Threshold to obtain the peaks
    // This will be the markers for the foreground objects
 	threshold(dist, dist, .4, 1., CV_THRESH_BINARY);
    // Dilate a bit the dist image
 	Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
 	dilate(dist, dist, kernel1);
 	imshow("Peaks", dist);
    // Create the CV_8U version of the distance image
    // It is needed for findContours()
 	Mat dist_8u;
 	dist.convertTo(dist_8u, CV_8U);
    // Find total markers
 	vector<vector<Point> > contours;
 	findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    // Create the marker image for the watershed algorithm
 	Mat markers = Mat::zeros(dist.size(), CV_32SC1);
    // Draw the foreground markers
 	for (size_t i = 0; i < contours.size(); i++)
 		drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);
    // Draw the background marker
 	circle(markers, Point(5,5), 3, CV_RGB(255,255,255), -1);
 	imshow("Markers", markers*10000);
    // Perform the watershed algorithm
 	watershed(src, markers);
 	Mat mark = Mat::zeros(markers.size(), CV_8UC1);
 	markers.convertTo(mark, CV_8UC1);
 	bitwise_not(mark, mark);
    //imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
                                  // image looks like at that point
   // Generate random colors
 	vector<Vec3b> colors;
 	for (size_t i = 0; i < contours.size(); i++)
 	{
 		int b = theRNG().uniform(0, 255);
 		int g = theRNG().uniform(0, 255);
 		int r = theRNG().uniform(0, 255);
 		colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
 	}
    // Create the result image
 	dst = Mat::zeros(markers.size(), CV_8UC3);
    // Fill labeled objects with random colors
 	for (int i = 0; i < markers.rows; i++)
 	{
 		for (int j = 0; j < markers.cols; j++)
 		{
 			int index = markers.at<int>(i,j);
 			if (index > 0 && index <= static_cast<int>(contours.size()))
 				dst.at<Vec3b>(i,j) = colors[index-1];
 			else
 				dst.at<Vec3b>(i,j) = Vec3b(0,0,0);
 		}
 	}
    // Visualize the final image
 	imshow("Final Result", dst);

 }
 int main( int argc, char** argv )
 {
 /// Read image ( same size, same type )
 	Mat tempsrc1 = imread("greenScreen01.jpg");
 	Mat tempsrc2 = imread("greenScreen01.jpg");
 	if( !tempsrc1.data ) { printf("Error loading src1 \n"); return -1; }
 	if( !tempsrc2.data ) { printf("Error loading src2 \n"); return -1; }
 	Mat src1,src2;

 	cvtColor(tempsrc1, src1, CV_BGR2GRAY);
 	cvtColor(tempsrc2, src2, CV_BGR2GRAY);

 /// Initialize values
 	alpha_slider = 0;

 /// Create Windows
 	namedWindow("Linear Blend", 1);

 /// Create Trackbars
 	char TrackbarName[50];
 	sprintf( TrackbarName, "Alpha x %d", alpha_slider_max );

 	createTrackbar( TrackbarName, "Linear Blend", &alpha_slider, alpha_slider_max, on_trackbar );

 /// Show some stuff
 	on_trackbar( alpha_slider, 0 );

 /// Wait until user press some key
 	waitKey(0);
 	return 0;
 }