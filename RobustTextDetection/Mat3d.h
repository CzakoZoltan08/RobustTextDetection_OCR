#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <bitset>

class Mat3
{

public:
	Mat3(cv::Mat sourceMatrix);
	cv::Mat matrix;
	std::bitset<8> bit_depth;
};
