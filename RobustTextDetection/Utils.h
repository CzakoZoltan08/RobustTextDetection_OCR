#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat ConvertToGrayscale(Mat image);
Mat ExtractMSERRegions(Mat grey);
Mat DoGradient(Mat sourceImg);
Mat* DoGeometricFiltering(Mat sourceImg, float min_treshold, float max_treshold);

Mat FindStrokeWidth(Mat& sourceImg);
void FindBlobs(const Mat &binary, vector <vector<Point2i> > &blobs);

Mat GrowEdges(Mat& image, Mat& edges);
int ToOctave(const float angle, const int neighbors);

#endif