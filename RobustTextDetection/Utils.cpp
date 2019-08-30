#include "Utils.h"
#include <bitset>
#include "Mat3d.h"

Mat ConvertToGrayscale(Mat sourceImg) {

	Mat grayImg;

	cv::cvtColor(sourceImg, grayImg, CV_BGR2GRAY);
	return grayImg;
}

Mat ExtractMSERRegions(Mat sourceImg) {

	// See: http://docs.opencv.org/2.4/modules/features2d/doc/feature_detection_and_description.html

	MSER mser(8, 100, 500, 0.25, 0.2, 200, 1.01, 0.003, 5);
	vector<vector<Point>> mserContours;

	mser(sourceImg, mserContours);

	// Initialize dummy image (black)
	Mat mserRegion = Mat::zeros(sourceImg.size(), CV_8UC1);

	for (int i = 0; i < mserContours.size(); ++i) {

		for (int j = 0; j < mserContours.at(i).size(); ++j)
		{
			auto& p = mserContours.at(i).at(j);
			mserRegion.at<uchar>(p) = 255;
		}
	}

	return mserRegion;
}

Mat DoGradient(Mat sourceImg)
{
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	Mat grad;
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	// Gradient on X
	Sobel(sourceImg, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	/// Gradient on Y
	Sobel(sourceImg, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	/// Total Gradient (approximate)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	return grad;
}

bool isValidCC(vector<Point> points, float min_treshold, float max_treshold)
{
	auto boundingBox = boundingRect(points);

	auto aspectRatio = float(boundingBox.width) / boundingBox.height;

	if (aspectRatio < min_treshold || aspectRatio > max_treshold)
		return false;
	return true;
}

Mat* DoGeometricFiltering(Mat sourceImg, float min_treshold, float max_treshold)
{
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(sourceImg, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	vector<vector<Point>> validContours;
	for (auto i = 0; i < contours.size(); ++i)
	{
		if (isValidCC(contours.at(i), min_treshold, max_treshold))
			validContours.push_back(contours.at(i));
	}

	Mat* mats = new Mat[2];

	Mat display = Mat(sourceImg.size(), CV_8UC3, cvScalar(0, 0, 0));
	Mat core = Mat(sourceImg.size(), CV_8UC3, cvScalar(0, 0, 0));
	for (auto i = 0; i < contours.size(); ++i)
	{
		auto white = Scalar(255, 255, 255);
		auto red = Scalar(0, 0, 255);

		if (isValidCC(contours.at(i), min_treshold, max_treshold)) {
			drawContours(display, contours, i, white, 2, 8, hierarchy, 0, Point());
			drawContours(core, contours, i, white, 2, 8, hierarchy, 0, Point());
		}
		else
			drawContours(display, contours, i, red, 2, 8, hierarchy, 0, Point());
	}

	mats[0] = display;
	mats[1] = core;

	return mats;
}

std::string GetMatType(const cv::Mat& mat)
{
	const int mtype = mat.type();

	switch (mtype)
	{
	case CV_8UC1:  return "CV_8UC1";
	case CV_8UC2:  return "CV_8UC2";
	case CV_8UC3:  return "CV_8UC3";
	case CV_8UC4:  return "CV_8UC4";

	case CV_8SC1:  return "CV_8SC1";
	case CV_8SC2:  return "CV_8SC2";
	case CV_8SC3:  return "CV_8SC3";
	case CV_8SC4:  return "CV_8SC4";

	case CV_16UC1: return "CV_16UC1";
	case CV_16UC2: return "CV_16UC2";
	case CV_16UC3: return "CV_16UC3";
	case CV_16UC4: return "CV_16UC4";

	case CV_16SC1: return "CV_16SC1";
	case CV_16SC2: return "CV_16SC2";
	case CV_16SC3: return "CV_16SC3";
	case CV_16SC4: return "CV_16SC4";

	case CV_32SC1: return "CV_32SC1";
	case CV_32SC2: return "CV_32SC2";
	case CV_32SC3: return "CV_32SC3";
	case CV_32SC4: return "CV_32SC4";

	case CV_32FC1: return "CV_32FC1";
	case CV_32FC2: return "CV_32FC2";
	case CV_32FC3: return "CV_32FC3";
	case CV_32FC4: return "CV_32FC4";

	case CV_64FC1: return "CV_64FC1";
	case CV_64FC2: return "CV_64FC2";
	case CV_64FC3: return "CV_64FC3";
	case CV_64FC4: return "CV_64FC4";

	default:
		return "Invalid type of matrix!";
	}
}

bool isLowerThan(int leftOperand, int rightOperand)
{
	return leftOperand < rightOperand && leftOperand != 0;
}

uchar ComputeNeighbourValues(int* previousRow, int* currentRow, int* nextRow, int column)
{
	bitset<8> neighbours;

	// TOP LEFT
	if (isLowerThan(previousRow[column - 1], currentRow[column]))
		neighbours[0] = 1;

	// TOP MIDDLE
	if (isLowerThan(previousRow[column], currentRow[column]))
		neighbours[1] = 1;

	// TOP RIGHT
	if (isLowerThan(previousRow[column + 1], currentRow[column]))
		neighbours[2] = 1;

	// RIGHT
	if (isLowerThan(currentRow[column + 1], currentRow[column]))
		neighbours[3] = 1;

	// BOTTOM RIGHT
	if (isLowerThan(nextRow[column + 1], currentRow[column]))
		neighbours[4] = 1;

	// BOTTOM MIDDLE
	if (isLowerThan(nextRow[column], currentRow[column]))
		neighbours[5] = 1;

	// BOTTOM LEFT
	if (isLowerThan(nextRow[column - 1], currentRow[column]))
		neighbours[6] = 1;

	// LEFT
	if (isLowerThan(currentRow[column - 1], currentRow[column]))
		neighbours[7] = 1;


	return static_cast<uchar>(neighbours.to_ullong());
}

bitset<8> ToBits(unsigned char byte)
{
	return bitset<8>(byte);
}

void PopulateNeighboursLocations(vector<Point>* neighboursLocation, uchar* lookUpContent, int x, int y)
{
	bitset<8> neighbours = ToBits(*lookUpContent);

	// TOP LEFT
	if (neighbours[0] == 1)
		neighboursLocation->push_back(Point(x - 1, y - 1));

	// TOP MIDDLE
	if (neighbours[1] == 1)
		neighboursLocation->push_back(Point(x - 1, y));

	// TOP RIGHT
	if (neighbours[2] == 1)
		neighboursLocation->push_back(Point(x - 1, y + 1));

	// MIDDLE RIGHT
	if (neighbours[3] == 1)
		neighboursLocation->push_back(Point(x, y + 1));

	// BOTTOM RIGHT
	if (neighbours[4] == 1)
		neighboursLocation->push_back(Point(x + 1, y + 1));

	// BOTTOM MIDDLE
	if (neighbours[5] == 1)
		neighboursLocation->push_back(Point(x + 1, y));

	// BOTTOM LEFT
	if (neighbours[6] == 1)
		neighboursLocation->push_back(Point(x + 1, y - 1));

	// MIDDLE LEFT
	if (neighbours[7] == 1)
		neighboursLocation->push_back(Point(x, y - 1));
}

Mat FindStrokeWidth(Mat& sourceImg)
{
	// Distance Transform 

	Mat inputSrc;
	cvtColor(sourceImg, inputSrc, CV_BGR2GRAY);
	string dx = GetMatType(inputSrc);

	Mat D(sourceImg.size(), CV_32FC1, Scalar(255));
	cv::distanceTransform(inputSrc, D, CV_DIST_L2, 3);

	normalize(D, D, 0.0, 1.0, NORM_MINMAX);

	// Collecting neighbours
	int* previousRow;
	int* currentRow;
	int* nextRow;

	Mat lookUpMatrix(D.size(), CV_8UC1, Scalar(0));

	for (auto i = 1; i < D.rows - 2; ++i)
	{
		previousRow = D.ptr<int>(i - 1);
		currentRow = D.ptr<int>(i);
		nextRow = D.ptr<int>(i + 1);

		auto pixel = lookUpMatrix.ptr<uchar>(i);
		for (auto j = 1; j < D.cols - 2; ++j)
		{
			if (currentRow[j] != 0)
			{
				pixel[j] = ComputeNeighbourValues(previousRow, currentRow, nextRow, j);
			}
		}
	}

	// Find max stroke from D
	double min, max;

	minMaxIdx(D, &min, &max);

	// Compute Neighbour Index
	for (auto stroke = max; stroke >= 1; --stroke)
	{
		vector<Point> strokeLocations;
		vector<Point> neighboursLocations;

		// Find pixels location with stroke range from max to > 0
		findNonZero(D == stroke, strokeLocations);

		// Find neighbours locations of each pixel location
		for (int i = 0; i < strokeLocations.size(); ++i)
		{
			auto contentLookup = lookUpMatrix.at<uchar>(strokeLocations.at(i));

			PopulateNeighboursLocations(&neighboursLocations, &contentLookup, strokeLocations.at(i).x, strokeLocations.at(i).y);

			while (neighboursLocations.size() != 0)
			{
				for (int j = 0; j < neighboursLocations.size(); ++j)
				{
					D.at<int>(neighboursLocations.at(j)) = stroke;
				}

				neighboursLocations.clear();
				for (int j = 0; j < neighboursLocations.size(); ++j)
				{
					PopulateNeighboursLocations(&neighboursLocations, &contentLookup, neighboursLocations.at(j).x, neighboursLocations.at(j).y);
				}
			}
		}
	}

	return D;
}

void FindBlobs(const cv::Mat &binary, std::vector < std::vector<cv::Point2i> > &blobs)
{
	blobs.clear();

	// Fill the label_image with the blobs
	// 0  - background
	// 1  - unlabelled foreground
	// 2+ - labelled foreground

	cv::Mat label_image;
	binary.convertTo(label_image, CV_32SC1);

	int label_count = 2; // starts at 2 because 0,1 are used already

	for (int y = 0; y < label_image.rows; y++) {
		int *row = (int*)label_image.ptr(y);
		for (int x = 0; x < label_image.cols; x++) {
			if (row[x] != 1) {
				continue;
			}

			cv::Rect rect;
			cv::floodFill(label_image, cv::Point(x, y), label_count, &rect, 0, 0, 4);

			std::vector <cv::Point2i> blob;

			for (int i = rect.y; i < (rect.y + rect.height); i++) {
				int *row2 = (int*)label_image.ptr(i);
				for (int j = rect.x; j < (rect.x + rect.width); j++) {
					if (row2[j] != label_count) {
						continue;
					}

					blob.push_back(cv::Point2i(j, i));
				}
			}

			blobs.push_back(blob);

			label_count++;
		}
	}
}

int ToOctave(const float angle, const int neighbors) {
	const float divisor = 180.0 / neighbors;
	return static_cast<int>((floor(angle / divisor) / 2) + 1) % neighbors + 1;
}

Mat GrowEdges(Mat& image, Mat& edges) {

	Mat grad_x, grad_y;
	Sobel(image, grad_x, CV_32FC1, 1, 0);
	Sobel(image, grad_y, CV_32FC1, 0, 1);

	Mat grad_mag, grad_dir;
	cartToPolar(grad_x, grad_y, grad_mag, grad_dir, true);


	/* Convert the angle into predefined 3x3 neighbor locations
	| 2 | 3 | 4 |
	| 1 | 0 | 5 |
	| 8 | 7 | 6 |
	*/
	for (int y = 0; y < grad_dir.rows; y++) {
		float * grad_ptr = grad_dir.ptr<float>(y);

		for (int x = 0; x < grad_dir.cols; x++) {
			if (grad_ptr[x] != 0)
				grad_ptr[x] = ToOctave(grad_ptr[x], 8);
		}
	}
	grad_dir.convertTo(grad_dir, CV_8UC1);


	/* Perform region growing based on the gradient direction */
	Mat result = edges.clone();

	uchar * prev_ptr = result.ptr<uchar>(0);
	uchar * curr_ptr = result.ptr<uchar>(1);

	for (int y = 1; y < edges.rows - 1; y++) {
		uchar * edge_ptr = edges.ptr<uchar>(y);
		uchar * grad_ptr = grad_dir.ptr<uchar>(y);
		uchar * next_ptr = result.ptr<uchar>(y + 1);

		for (int x = 1; x < edges.cols - 1; x++) {
			/* Only consider the contours */
			if (edge_ptr[x] != 0) {

				switch (grad_ptr[x]) {
				case 1: curr_ptr[x - 1] = 255; break;
				case 2: prev_ptr[x - 1] = 255; break;
				case 3: prev_ptr[x] = 255; break;
				case 4: prev_ptr[x + 1] = 255; break;
				case 5: curr_ptr[x] = 255; break;
				case 6: next_ptr[x + 1] = 255; break;
				case 7: next_ptr[x] = 255; break;
				case 8: next_ptr[x - 1] = 255; break;
				default: break;
				}
			}
		}

		prev_ptr = curr_ptr;
		curr_ptr = next_ptr;
	}

	return result;
}