#include"CCA_Labeling.h"	
using namespace std;
using std::pair;
using std::vector;
typedef  pair<CvPoint, CvPoint> Box;
cv::Mat drawBox(const cv::Mat& _labelImg, cv::Mat& _BoxImg)
{
	if (_labelImg.empty())
	{
		std::cout << "_labelImg is empty" << std::endl;
		return _BoxImg;
	}
		if(_labelImg.type() != CV_32SC1)
	{
			std::cout << "_labelImg type "<<_labelImg.type() << std::endl;
		_BoxImg.release();
		return _BoxImg;
	}
	int rows = _labelImg.rows;
	int cols = _labelImg.cols;

	_BoxImg.release();
	_BoxImg.create(rows, cols, CV_8UC3);
	_BoxImg = cv::Scalar::all(0);
	double maxLabel;
	cv::minMaxLoc(_labelImg, NULL, &maxLabel);
	std::vector<Box>Boxs;
	for (int k = 2; k <= maxLabel; k++)
	{
		int left = cols;
		int right = 0;
		int up = rows;
		int down = 0;
		for (int i = 0; i < rows; i++)
		{
			const int* data_src = (int*)_labelImg.ptr<int>(i);
			for (int j = 0; j < cols; j++)
			{
				if (data_src[j] == k)
				{
					left = left < j ? left : j;
					right = right >j ? right : j;
					up = up < i ? up : i;
					down = down>i ? down : i;
				}
			}
		}
		CvPoint p1 = cvPoint(left, up);
		CvPoint p2 = cvPoint(right, down);
		if ((down-up)>10&&(right-left)>10)
		{
			Boxs.push_back(Box(p1, p2));
		}
	}
	int boxSize = Boxs.size();
	for (int i = 0; i < boxSize; i++)
	{
		CvPoint p1 = Boxs[i].first;
		CvPoint p2 = Boxs[i].second;
		cv::rectangle(_BoxImg, p1, p2, cvScalar(255, 0, 255), 2, 8, 0);
	}
	return _BoxImg;
}

int icvprCcaByTwoPass(const cv::Mat& _binImg, cv::Mat& _labelImg)
{
	// connected component analysis (4-component)
	// use two-pass algorithm
	// 1. first pass: label each foreground pixel with a label
	// 2. second pass: visit each labeled pixel and merge neighbor labels
	// 
	// foreground pixel: _binImg(x,y) = 1
	// background pixel: _binImg(x,y) = 0


	if (_binImg.empty() ||
		_binImg.type() != CV_8UC1)
	{
		std::cout << "_binImg is empty or _binImg's type is not CV_8UC1" << std::endl;
		return -1;
	}

	// 1. first pass

	_labelImg.release();
	_binImg.convertTo(_labelImg, CV_32SC1);

	int label = 1;  // start by 2
	std::vector<int> labelSet;
	labelSet.push_back(0);   // background: 0
	labelSet.push_back(1);   // foreground: 1

	int rows = _binImg.rows - 1;
	int cols = _binImg.cols - 1;
	for (int i = 1; i < rows; i++)
	{
		int* data_preRow = _labelImg.ptr<int>(i - 1);
		int* data_curRow = _labelImg.ptr<int>(i);
		for (int j = 1; j < cols; j++)
		{
			if (data_curRow[j] == 1)
			{
				std::vector<int> neighborLabels;
				neighborLabels.reserve(2);
				int leftPixel = data_curRow[j - 1];
				int upPixel = data_preRow[j];
				if (leftPixel > 1)
				{
					neighborLabels.push_back(leftPixel);
				}
				if (upPixel > 1)
				{
					neighborLabels.push_back(upPixel);
				}

				if (neighborLabels.empty())
				{
					labelSet.push_back(++label);  // assign to a new label
					data_curRow[j] = label;
					labelSet[label] = label;
				}
				else
				{
					std::sort(neighborLabels.begin(), neighborLabels.end());
					int smallestLabel = neighborLabels[0];
					data_curRow[j] = smallestLabel;

					// save equivalence
					for (size_t k = 1; k < neighborLabels.size(); k++)
					{
						int tempLabel = neighborLabels[k];
						int& oldSmallestLabel = labelSet[tempLabel];
						if (oldSmallestLabel > smallestLabel)
						{
							labelSet[oldSmallestLabel] = smallestLabel;
							oldSmallestLabel = smallestLabel;
						}
						else if (oldSmallestLabel < smallestLabel)
						{
							labelSet[smallestLabel] = oldSmallestLabel;
						}
					}
				}
			}
		}
	}

	// update equivalent labels
	// assigned with the smallest label in each equivalent label set
	for (size_t i = 2; i < labelSet.size(); i++)
	{
		int curLabel = labelSet[i];
		int preLabel = labelSet[curLabel];
		while (preLabel != curLabel)
		{
			curLabel = preLabel;
			preLabel = labelSet[preLabel];
		}
		labelSet[i] = curLabel;
	}
	// 2. second pass
	int maxLabelIndex = -1;
	for (int i = 0; i < rows; i++)
	{
		int* data = _labelImg.ptr<int>(i);
		for (int j = 0; j < cols; j++)
		{
			int& pixelLabel = data[j];
			pixelLabel = labelSet[pixelLabel];
			if (pixelLabel > maxLabelIndex)
			{
				maxLabelIndex = pixelLabel;
			}
		}
	}
	return maxLabelIndex;
}

cv::Scalar icvprGetRandomColor()
{
	uchar r = 255 * (rand() / (1.0 + RAND_MAX));
	uchar g = 255 * (rand() / (1.0 + RAND_MAX));
	uchar b = 255 * (rand() / (1.0 + RAND_MAX));
	return cv::Scalar(b, g, r);
}

void icvprLabelColor(const cv::Mat& _labelImg, cv::Mat& _colorLabelImg)
{
	if (_labelImg.empty() ||
		_labelImg.type() != CV_32SC1)
	{
		return;	
	}

	std::map<int, cv::Scalar> colors;
	int rows = _labelImg.rows;
	int cols = _labelImg.cols;

	_colorLabelImg.release();
	_colorLabelImg.create(rows, cols, CV_8UC3);
	_colorLabelImg = cv::Scalar::all(0);
	for (int i = 0; i < rows; i++)
	{
		const int* data_src = (int*)_labelImg.ptr<int>(i);
		uchar* data_dst = _colorLabelImg.ptr<uchar>(i);
		for (int j = 0; j < cols; j++)
		{
			int pixelValue = data_src[j];
			if (pixelValue > 1)
			{
				if (colors.count(pixelValue) <= 0)
				{
					colors[pixelValue] = icvprGetRandomColor();
				}
				cv::Scalar color = colors[pixelValue];
				*data_dst++ = color[0];
				*data_dst++ = color[1];
				*data_dst++ = color[2];
			}
			else
			{
				data_dst++;
				data_dst++;
				data_dst++;
			}
		}
	}
}

cv::Mat  icvprCcaBySeedFill(const cv::Mat& _binImg, int minArea ,int maxArea)
{
	// connected component analysis (4-component)
	// use seed filling algorithm
	// 1. begin with a foreground pixel and push its foreground neighbors into a stack;
	// 2. pop the top pixel on the stack and label it with the same label until the stack is empty
	// 
	// foreground pixel: _binImg(x,y) = 1
	// background pixel: _binImg(x,y) = 0


	pair<double, unsigned int>labelArea;
	vector<pair<double, unsigned int> >labelAreas;

	if (_binImg.empty() ||
		_binImg.type() != CV_8UC1)
	{
		return cv::Mat();
	}
  cv::Mat _labelImg;
	_binImg.convertTo(_labelImg, CV_32SC1);

	int label = 1;  // start by 2

	int rows = _binImg.rows - 1;
	int cols = _binImg.cols - 1;
	
	for (int i = 1; i < rows - 1; i++)
	{
		int* data = _labelImg.ptr<int>(i);
		for (int j = 1; j < cols - 1; j++)
		{
			if (data[j] == 1)
			{
				unsigned int area = 0;
				
				std::stack<std::pair<int, int> > neighborPixels;
				neighborPixels.push(std::pair<int, int>(i, j));     // pixel position: <i,j>
				++label;  // begin with a new label
				while (!neighborPixels.empty())
				{
					area++;
					// get the top pixel on the stack and label it with the same label
					std::pair<int, int> curPixel = neighborPixels.top();
					//int curX = curPixel.first;
					//int curY = curPixel.second;
					int curX = curPixel.first;
					int curY = curPixel.second;
					_labelImg.at<int>(curX, curY) = label;

					// pop the top pixel
					neighborPixels.pop();

					// push the 4-neighbors (foreground pixels)
					if (curY!=0)
					if (_labelImg.at<int>(curX, curY - 1) == 1)
					{// left pixel
						neighborPixels.push(std::pair<int, int>(curX, curY - 1));
					}
					if (curY<cols-1)
					if (_labelImg.at<int>(curX, curY + 1) == 1)
					{// right pixel
						neighborPixels.push(std::pair<int, int>(curX, curY + 1));
					}
					//std::cout << curX - 1 << "   " << curY << std::endl;
					if (curX!=0)
					if (_labelImg.at<int>(curX - 1, curY) == 1)
					{// up pixel
						neighborPixels.push(std::pair<int, int>(curX - 1, curY));
					}
					if (curX<rows-1)
					if (_labelImg.at<int>(curX + 1, curY) == 1)
					{// down pixel
						neighborPixels.push(std::pair<int, int>(curX + 1, curY));
					}
				}
				labelArea.first = area;
				labelArea.second = label;
				labelAreas.push_back(labelArea);
			}
			
		}
	}
	std::sort(labelAreas.begin(), labelAreas.end());
  // filter the ares < minArea's label
  vector<unsigned int> filterLabel;
  for (vector<pair<double, unsigned int> >::iterator it = labelAreas.begin(); it != labelAreas.end(); ++it) {
    if (it->first >= minArea && it->first <= maxArea) {
      filterLabel.push_back(it->second);
    }
  }
	cv::Mat temp = cv::Mat::zeros(_binImg.size(), CV_8UC1);
  if (filterLabel.size() <= 0) return temp;
	for (int i = 0; i < _labelImg.rows; ++i)
	{
		for (int j = 0; j < _labelImg.cols; ++j)
		{
      for (vector<unsigned int>::iterator it = filterLabel.begin(); it != filterLabel.end(); ++it) {
        if (_labelImg.at<unsigned int>(i, j) == *it)
          temp.at<uchar>(i, j) = 255;
      }
		}
	}
	return temp;
}
