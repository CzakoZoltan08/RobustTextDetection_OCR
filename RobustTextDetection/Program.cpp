#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv/highgui.h"

#include <stdlib.h>
#include <stdio.h>
#include "Utils.h"

using namespace cv;


int main()
{
	string temp_output_path = "../images/TestText.png";

	// STEP.[1] - Display default image
	Mat src = imread(temp_output_path);
	char* window_name_image = "Original Image";
	namedWindow(window_name_image, CV_WINDOW_AUTOSIZE);

	if (!src.data)
	{
		std::cout << "Error loading image!" << std::endl;
		return -1;
	}
	imshow(window_name_image, src);

	// STEP.[2] - Convert default image to Grayscale
	Mat grayImg = ConvertToGrayscale(src);
	char* window_name_gray = "Grayscale Image";
	namedWindow(window_name_gray, CV_WINDOW_AUTOSIZE);
	imshow(window_name_gray, grayImg);

	// STEP.[3] - Extract Edge-Enhanced MSER Regions
	Mat mserRegionsImg = ExtractMSERRegions(grayImg);
	char* window_name_mser = "MSER Regions Image";
	namedWindow(window_name_mser, CV_WINDOW_AUTOSIZE);
	imshow(window_name_mser, mserRegionsImg);

	// STEP.[4] - Extract Canny-Edges
	Mat cannyEdgesImg;
	Canny(grayImg, cannyEdgesImg, 20, 100);
	char* window_name_cannyedge = "Canny Edges Image";
	namedWindow(window_name_cannyedge, CV_WINDOW_AUTOSIZE);
	imshow(window_name_cannyedge, cannyEdgesImg);

	// STEP.[5] - Intersect Canny Edges with MSER Regions
	Mat cannyMSERIntersectionImg = cannyEdgesImg & mserRegionsImg;
	char* window_name_intersectMSER = "Intersect MSER & Canny Edges";
	namedWindow(window_name_intersectMSER, CV_WINDOW_AUTOSIZE);
	imshow(window_name_intersectMSER, cannyMSERIntersectionImg);

	// STEP.[6] - Compute Gradient direction
	Mat gradientDirectionImg = DoGradient(grayImg);
	char* window_name_Gradient= "Gradient Image";
	namedWindow(window_name_Gradient, CV_WINDOW_AUTOSIZE);
	imshow(window_name_Gradient, gradientDirectionImg);

	Mat newMSERRegionsImg = ~gradientDirectionImg & mserRegionsImg;
	char* window_name_EdgeEnhanced= "EdgeEnhanced Image";
	namedWindow(window_name_EdgeEnhanced, CV_WINDOW_AUTOSIZE);
	imshow(window_name_EdgeEnhanced, newMSERRegionsImg);

	// STEP.[7] - Geometric Filtering
	Mat* geometricFilteredImgs = DoGeometricFiltering(newMSERRegionsImg, 0.3, 2.8);
	char* window_name_geometricFiltered = "Geometric Filtering Image";
	namedWindow(window_name_geometricFiltered, CV_WINDOW_AUTOSIZE);
	imshow(window_name_geometricFiltered, geometricFilteredImgs[0]);

	// STEP.[8] - Find STROKE Width
	Mat strokedWidthImg = FindStrokeWidth(geometricFilteredImgs[1]);
	char* window_name_stroked_width = "Stroked Width Image";
	namedWindow(window_name_stroked_width, CV_WINDOW_AUTOSIZE);
	imshow(window_name_stroked_width, strokedWidthImg);

	// STEP.[9] - Find Connected Components
	Mat output = Mat::zeros(strokedWidthImg.size(), CV_8UC3);
	Mat binary;
	vector <vector<Point2i > > blobs;	

	threshold(strokedWidthImg, binary, 0.0, 1.0, THRESH_BINARY);
	FindBlobs(binary, blobs);

	// Randomy color the blobs
	for (size_t i = 0; i < blobs.size(); i++) {
		unsigned char r = 255 * (rand() / (1.0 + RAND_MAX));
		unsigned char g = 255 * (rand() / (1.0 + RAND_MAX));
		unsigned char b = 255 * (rand() / (1.0 + RAND_MAX));

		for (size_t j = 0; j < blobs[i].size(); j++) {
			int x = blobs[i][j].x;
			int y = blobs[i][j].y;

			output.at<Vec3b>(y, x)[0] = b;
			output.at<Vec3b>(y, x)[1] = g;
			output.at<Vec3b>(y, x)[2] = r;
		}
	}

	Mat core = Mat(output.size(), CV_8UC3, cvScalar(0, 0, 0));
	for (int i = 0; i < blobs.size(); i++)
	{
		auto area = contourArea(blobs[i]);
		vector<Point2i> hull;
		convexHull(blobs[i], hull);

		auto hull_area = contourArea(hull);
		auto solidity = float(area) / hull_area;

		Moments moment = cv::moments(blobs[i]);
		double left_comp = (moment.nu20 + moment.nu02) / 2.0;
		double right_comp = sqrt((4 * moment.nu11 * moment.nu11) + (moment.nu20 - moment.nu02)*(moment.nu20 - moment.nu02)) / 2.0;

		double eig_val_1 = left_comp + right_comp;
		double eig_val_2 = left_comp - right_comp;

		auto eccentrcity =  sqrtf(1.0 - (eig_val_2 / eig_val_1));
		//auto areaX = countNonZero(blobs[i]);

		if (solidity >= 0.01 && hull_area > 400){
			drawContours(core, blobs, i, Scalar(255, 255, 255), 2, 8, noArray(), 0, Point());
		}else
		{
			drawContours(core, blobs, i, Scalar(0, 0, 0), 2, 8, noArray(), 0, Point());
		}
	}

	
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	Mat inputSrc;

	cvtColor(core, inputSrc, CV_BGR2GRAY);
	findContours(inputSrc, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	vector<Point2i> vals;
	for (int i = 0; i < contours.size(); ++i)
	{
		vals.insert(vals.end(), contours[i].begin(), contours[i].end());
	}

	auto rect = boundingRect(vals);
	rectangle(src, rect, Scalar(0, 255, 0), 2);

	char* window_name_connnected_components = "Connected Components Image";
	namedWindow(window_name_connnected_components, CV_WINDOW_AUTOSIZE);
	imshow(window_name_connnected_components, core);


	char* finalImg = "Detected text";
	namedWindow(finalImg, CV_WINDOW_AUTOSIZE);
	imshow(finalImg, src);

	waitKey(0);
	return 0;
}