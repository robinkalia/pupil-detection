#include <opencv2/opencv.hpp>
#include <iostream>
#include "CCA_Labeling.h"
using namespace cv;
using namespace std;
cv::Point p1, p2;
cv::Rect rect;
cv::Point point;
cv::Mat  roi;

int threshold_value1 = 22;
int threshold_value2 = 1000;
int threshold_value3 = 3000;
int threshold_type = 1;
int const max_value1 = 255;
int const max_value2 = 3000;
int const max_value3 = 3000;
int const max_type = 4;
int const max_BINARY_value = 255;

Mat src, src_gray, dst;
string window_name = "Threshold Demo";

string trackbar_type = "Type";
string trackbar_value1 = "Value";
string trackbar_value2 = "min Area";
string trackbar_value3 = "max Area";

/// Function headers
void update(const cv::Mat&);

int main( int argc, char** argv )
{
  /// Load vedio
  cv::VideoCapture cap(argv[1]);

  namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  /// For binary threshold
  createTrackbar( trackbar_value1,
                  window_name, &threshold_value1,
                  max_value1, NULL );
  /// For max area
  createTrackbar( trackbar_value2,
                  window_name, &threshold_value2,
                  max_value2, NULL );
  /// For min area
  createTrackbar( trackbar_value3,
                  window_name, &threshold_value3,
                  max_value3, NULL );

  /// update
  while(true)
  {
    cv::Mat frame;
    cap >> frame;
    if (frame.data == NULL) {
      std::cout << "Finished!" << std::endl;
      return 0;
    }
    cvtColor( frame, frame, CV_BGR2GRAY);
    update(frame);
    int c;
    cv::imshow("frame", frame);
    c = waitKey( 20 );
    if( (char)c == 27 )
      { break; }
   }

}


/// Image Process
void update(const cv::Mat& frame) {

  threshold(frame, dst, threshold_value1, max_BINARY_value,threshold_type );
  cv::imshow("binary", dst);

  cv::Mat cca_out = icvprCcaBySeedFill(dst / 255, threshold_value2, threshold_value3);
  int elementSize = 3;
  cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                              cv::Size(2 * elementSize + 1, 2 * elementSize + 1),
                                              cv::Point(elementSize, elementSize));

  cv::Mat dilate_out, close_out;
  cv::dilate(cca_out, dilate_out, element);
  cv::erode(dilate_out, close_out, element);
  cv::Mat inv_cca_out = icvprCcaBySeedFill((255 - close_out) / 255, 000, 20000);
  cv::Mat add_out = close_out + inv_cca_out;

  imshow( window_name, add_out);

  cv::Mat dstImage = cv::Mat::zeros(add_out.rows, add_out.cols, CV_8UC3);
  vector<vector<cv::Point> >contours;
  vector<cv::Vec4i> hierarchy;
  int index = 0;
  cv::findContours(add_out, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
  if (contours.size() != 0) {
    for (; index >= 0; index = hierarchy[index][0]) {
      cv::Scalar color( rand()&255, rand()&255, rand()&255);
      cv::drawContours(dstImage, contours, index, color, 1, 8, hierarchy);
    }
  }
  imshow( "dst", dstImage);
}
