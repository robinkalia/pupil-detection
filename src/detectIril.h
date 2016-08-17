#include <opencv2/opencv.hpp>

cv::Mat computeMatXGradient(const cv::Mat &mat);
/// + -> 1
/// 0 -> 0
/// - -> -1
cv::Mat sign(const cv::Mat &mat);
/// If the value of pixel is 0, assign it to the last pixel(left or up)
/// Input: Mat_<int>  -1 0 1
cv::Mat reSign(const cv::Mat mat, int XorY); 
/// Using gauss scale sapce locate Iril
std::vector<cv::Point> detectIril(const cv::Mat& image, cv::Size gauss_size, int gauss_sigma, int thresh);
/// Sort the Iril point that closer to the center of image is in the front.
bool comparePoint(cv::Point p1, cv::Point p2);
void sortIril(const cv::Mat& image, std::vector<cv::Point>& points);
/// Get best Iril point
cv::Point getIril(const cv::Mat& image, cv::Size gauss_size, int gauss_sigma, int thresh);
