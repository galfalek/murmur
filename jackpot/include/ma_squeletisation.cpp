
void squeletisation(Mat& img,Mat & dst)
{
	//squeletisation par lantuejoul
	cv::Mat skel(img.size(), CV_8UC1, cv::Scalar(0));
	cv::Mat temp;
	cv::Mat eroded;

	cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
	bool done;		
	do
	{
		cv::erode(img, eroded, element);
		cv::dilate(eroded, temp, element); // temp = open(img)
		cv::subtract(img, temp, temp);
		cv::bitwise_or(skel, temp, skel);
		eroded.copyTo(img);
		done = (cv::countNonZero(img) == 0);
	} while (!done);
	dst=skel;
}
void signe_derivee(Mat& img, Mat& dst, int dir)
{	//renvoie une matrice de bits : positif => même signe de la dérivée de chaque coté du pixel
	//négatif => changement de signe=> crête

	Mat kernel= Mat::zeros(2,2,CV_32F );
	Mat kernel_i= Mat::zeros(2,2,CV_32F );
	Mat filtered,filtered_i;
	switch(dir)
	{
		case 6: 
		//dérivée en x
		kernel.at<float>(Point(0,0))=-1;
		kernel.at<float>(Point(0,1))=1;
		kernel_i.at<float>(Point(0,0))=1;
		kernel_i.at<float>(Point(0,1))=-1;
		filter2D(img, filtered,CV_32F, kernel, Point(0,0), 0, BORDER_DEFAULT );
		filter2D(img, filtered_i,CV_32F, kernel_i, Point(1,0), 0, BORDER_DEFAULT );

		break;
		case 8: 
		//dérivée en y
		kernel.at<float>(Point(0,0))=-1;
		kernel.at<float>(Point(1,0))=1;		
		kernel_i.at<float>(Point(0,0))=1;
		kernel_i.at<float>(Point(1,0))=-1;
		filter2D(img, filtered,CV_32F, kernel, Point(1,0), 0, BORDER_DEFAULT );
		filter2D(img, filtered_i,CV_32F, kernel_i, Point(0,0), 0, BORDER_DEFAULT );

		break;
		case 7: 
		//dérivée en diagonale 7
		kernel.at<float>(Point(0,0))=-1;
		kernel.at<float>(Point(1,1))=1;		
		kernel_i.at<float>(Point(0,0))=1;
		kernel_i.at<float>(Point(1,1))=-1;
		filter2D(img, filtered,CV_32F, kernel, Point(0,0), 0, BORDER_DEFAULT );
		filter2D(img, filtered_i,CV_32F, kernel_i, Point(1,1), 0, BORDER_DEFAULT );

		break;
		case 9: 
		//dérivée en diagonale 9
		kernel.at<float>(Point(0,1))=-1;
		kernel.at<float>(Point(1,0))=1;		
		kernel_i.at<float>(Point(0,1))=1;
		kernel_i.at<float>(Point(1,0))=-1;
		filter2D(img, filtered,CV_32F, kernel, Point(0,1), 0, BORDER_DEFAULT );
		filter2D(img, filtered_i,CV_32F, kernel_i, Point(1,0), 0, BORDER_DEFAULT );
		break;

	}
	dst= filtered.mul(filtered_i);
	//affichage: 
	Mat normed;
	normalize(dst, normed, 0, 1., NORM_MINMAX);
	imshow( "ma diderction"+ dir-6, normed );
}
void squeletisation_dist(Mat& img, Mat& dst, int lowThreshold)
{
	imshow( "distance map", img );
	Mat img_copy= img.clone();
	Mat signe_deriv;
	dst= Mat::zeros(img.size(),CV_32F);
	//pour chaque direction de la dérivée:
	for(int dir= 6;dir<10;dir++)
	{
		//on trouve les points de crête
		signe_derivee(img_copy,signe_deriv,dir);
	//on garde les valeurs négatives des deux dérivées => points de crète
		for (int i = 0; i < src.cols; ++i)
		{
			for (int j = 0; j < src.rows; ++j)
			{
				if(signe_deriv.at<float>(Point(i,j))>0)
				{	 dst.at<float>(Point(i,j))=1;//sign_x.at<float>(Point(i,j));
				}
				
			}

		}

	}
}
void on_trackbar_dist( int, void* )
{
	Mat eltstruct1=  getStructuringElement(MORPH_CROSS, Size(5,5),Point(-1,-1));
	Mat eltstruct2=  getStructuringElement(MORPH_ELLIPSE, Size(5,5),Point(-1,-1));
	Mat foreground, skeleton,dist;
	Mat temp= src.clone();
	//on récupère les thresholds de la GUI	
	int threshold_foreground =  threshold_slider;
	double threshold_squel= (double)threshold_slider_dist/(double) 255;

	//on trouve le foreground:
	extractForeground(temp,foreground, threshold_foreground);



	imshow( "Threshold", foreground );
	resizeWindow("Threshold", 1024, 1024);
	//on le ferme, pour la sgementation:
	morphologyEx(foreground, foreground,MORPH_CLOSE, eltstruct1, Point(-1,-1), 2, BORDER_CONSTANT);	
	morphologyEx(foreground, foreground,MORPH_CLOSE, eltstruct2, Point(-1,-1), 2, BORDER_CONSTANT);	
	imshow( "Opened Threshold", foreground );

		// Skining Zhuang: 
	Mat zuhang;
	thinning(foreground, zuhang);
	imshow( "zuhang", zuhang );

	//on fait la distance transform, qu'on normalise
	distanceTransform(foreground, dist, CV_DIST_L2, 3);
	normalize(dist, dist, 0, 1., NORM_MINMAX);
	imshow( "distance", dist );

	//on squeletise
	squeletisation_dist(dist, dist, threshold_squel);
	imshow( "distanceSkeleton", dist );
	std::cout << "Temp type: "<< temp.type() << " dist type: "<< dist.type()<< std::endl;
	std::cout << "Temp size: "<< temp.size() << " dist size: "<< dist.size()<< std::endl;
	// cvtColor(temp,temp,CV_BGR2GRAY);
	//on affiche par rapport à l'image originale: 
	for (int i = 0; i < src.cols; ++i)
	{
		for (int j = 0; j < src.rows; ++j)
		{
			if(dist.at<float>(Point(i,j))>0)
			{	
				temp.at<cv::Vec3b>(Point(i,j))[0]=255;//sign_x.at<float>(Point(i,j));
				temp.at<cv::Vec3b>(Point(i,j))[1]=255;//sign_x.at<float>(Point(i,j));
				temp.at<cv::Vec3b>(Point(i,j))[2]=255;//sign_x.at<float>(Point(i,j));
			}

		}

	}
	namedWindow("Squelete", WINDOW_NORMAL );
	resizeWindow("Squelete", 1024, 1024);
	imshow( "Squelete", temp );
}