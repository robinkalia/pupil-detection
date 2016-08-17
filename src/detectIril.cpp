#include "detectIril.h"
#include <algorithm>
#include <iostream>
using namespace std;
using namespace cv;

cv::Mat computeMatXGradient(const cv::Mat &mat) {
  cv::Mat out(mat.rows,mat.cols,CV_64F);
  for (int y = 0; y < mat.rows; ++y) {
    const double *Mr = mat.ptr<double>(y);
    double *Or = out.ptr<double>(y);
    Or[0] = Mr[1] - Mr[0];
    for (int x = 1; x < mat.cols - 1; ++x) {
      Or[x] = Mr[x] - Mr[x - 1];
    }
  }
  return out;
}
cv::Mat sign(const cv::Mat &mat) {
  int cols = mat.cols, rows = mat.rows;
  cv::Mat result = cv::Mat::zeros(rows, cols, CV_32S);
  int* result_val = (int *) result.ptr();
  double* mat_val = (double *) mat.ptr();
  for (int i = 0; i < rows * cols; ++i) {
    if (mat_val[i] > 0.0000001) result_val[i] = 1;
    else if (mat_val[i] < 0.0000001 && mat_val[i] > -0.0000001) result_val[i] = 0;
    else result_val[i] = -1;
  }
  return result;
}
cv::Mat reSign(const cv::Mat mat, int XorY) {
  cv::Mat result = cv::Mat::zeros(mat.rows, mat.cols, CV_32S);
  int cols = mat.cols;
  int rows = mat.rows;
  // Y
  if (XorY == 1) {
    for (int x = 1; x < cols; ++x) {
      std::pair<int, int> yPair;
      for (int y = 1; y < rows - 1; ++y) {
        if (mat.at<int>(y, x) != 0) {
          result.at<int>(y, x) = mat.at<int>(y, x) - mat.at<int>(y - 1, x);
        }
        else {
          if (mat.at<int>(y - 1, x) != 0) {
            yPair.first = y;
          }
          if (mat.at<int>(y + 1, x) != 0) {
            yPair.second = y;
            for (int i = yPair.first; i <= yPair.second; ++i) {
              result.at<int>(i, x) = mat.at<int>(yPair.second + 1, x) - mat.at<int>(yPair.first - 1, x);
            }
          }
        }
      }
    }
  }
  // X
  else {
    for (int y = 1;y < rows; ++y) {
      std::pair<int, int> xPair;
      for (int x = 1; x < cols - 1; ++x) {
        if (mat.at<int>(y, x) != 0) {
          result.at<int>(y, x) = mat.at<int>(y, x) - mat.at<int>(y, x - 1);
        }
        else {
          if (mat.at<int>(y, x - 1) != 0) {
            xPair.first = x - 1;
          }
          if (mat.at<int>(y, x + 1) != 0) {
            xPair.second = x + 1;
            for (int i = xPair.first; i <= xPair.second; ++i) {
              result.at<int>(y, i) = mat.at<int>(y, xPair.second) - mat.at<int>(y, xPair.first);
            }
          }
        }
      }
    }
  }
  return result;
}
std::vector<cv::Point> detectIril(const cv::Mat& frame, cv::Size gauss_size, int gauss_sigma, int thresh){
  std::vector<cv::Point> result;
  cv::Mat gray;
  cvtColor(frame, gray, CV_BGR2GRAY);
  cv::Mat doubleMat;
  gray.convertTo(doubleMat, CV_64F);
  cv::Mat gauss;
  cv::GaussianBlur(doubleMat, gauss, gauss_size, gauss_sigma);

  // cv::medianBlur(doubleMat, gauss, 111);
  int cols = frame.cols;
  int rows = frame.rows;

  cv::Mat gradientX = computeMatXGradient(gauss);
  cv::Mat gradientY = computeMatXGradient(gauss.t()).t();
  cv::Mat signX = sign(gradientX);
  cv::Mat signY = sign(gradientY);
  cv::Mat gradientReSignX = reSign(signX, 0);
  cv::Mat gradientReSignY = reSign(signY, 1);
  int * dataX = (int*)gradientReSignX.ptr();
  int * dataY = (int*)gradientReSignY.ptr();
  for (int x = 0; x  < cols; ++x) {
    for (int y = 0; y < rows; ++y) {
      if (dataX[y * cols + x] == 2 && dataY[y * cols + x] == 2  && gray.at<unsigned char>(y, x) < thresh 
          )
        {
          result.push_back(cv::Point(x, y));
        }
    }
  }
  return result;
}

bool comparePoint(cv::Point p1, cv::Point p2) { // for sortIril
  return abs(p1.x) + abs(p1.y) < abs(p2.x) + abs(p2.y);
}
void sortIril(const cv::Mat& image, std::vector<cv::Point>& points) {
  int rows = image.rows;
  int cols = image.cols;
  for (std::vector<cv::Point>::iterator it = points.begin(); it != points.end(); ++it) {
    *it -=cv::Point(cols, rows);
  }
  sort(points.begin(), points.end(), comparePoint);
  for (std::vector<cv::Point>::iterator it = points.begin(); it != points.end(); ++it) {
    *it +=cv::Point(cols, rows);
  }
}
/// Get best Iril point
cv::Point getIril(const cv::Mat& image, cv::Size gauss_size, int gauss_sigma, int thresh) {
  std::vector<cv::Point> points = detectIril(image, gauss_size, gauss_sigma, thresh);
  sortIril(image, points);
  if (points.size() != 0)
    return points.at(0);
  else
    return cv::Point(0, 0);
}

