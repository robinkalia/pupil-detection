#include<opencv2/opencv.hpp>
#include "detectIril.h"
#include <iostream>
#include <algorithm>
#include <vector>
using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
  // cv::Size gauss_size = cv::Size(91, 91);
  int gauss_size = 45;
  int gauss_sigma = 16;
  int thresh = 20;

  /// Load vedio
  cv::VideoCapture cap;
  cv::Mat image;
  if (argv[1][0] == 'v') // input video
    cap = cv::VideoCapture(argv[2]);
  else if (argv[1][0] == 'i') // input image
    image = cv::imread(argv[2], 3);
  else if (argv[1][0] == 'c') // input camera
    cap = cv::VideoCapture(argv[2][0] - '0');

  cv::namedWindow("frame");
  cv::createTrackbar("gauss size", "frame", &gauss_size, 111);
  cv::createTrackbar("gauss_sigma", "frame", &gauss_sigma, 111);
  /// update
  while(true)
    {
      cv::Mat frame;
      if (argv[1][0] == 'v') // input video 
        {
          cap >> frame;
          if (frame.data == NULL) {
            std::cout << "Finished!" << std::endl;
            return 0;
          }

        }
      else if (argv[1][0] == 'i') // input image
        frame = image.clone();
      else if (argv[1][0] == 'c') // input camera
        {
          cap >> frame;
          cap >> frame;
          if (frame.data == NULL) {
            std::cout << "Finished!" << std::endl;
            return 0;
          }
        }
      else {
        return -1;
      }

      cv::Point iril = getIril(frame, cv::Size(gauss_size * 2 + 1, gauss_size * 2 + 1), gauss_sigma, thresh);
      cv::circle(frame, iril, 2, cv::Scalar(0, 0 ,255), 3, 8, 0);
      cv::imshow("frame", frame);
      char key = cv::waitKey(1);
      if (key == 'q') break;
    }
}
