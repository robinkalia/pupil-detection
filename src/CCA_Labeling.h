//  Connected Component Analysis/Labeling -- Test code
//  Author:  www.icvpr.com  
//  Blog  :  http://blog.csdn.net/icvpr 

#include<iostream>
#include<string>
#include<list>
#include<vector>
#include<map>
#include<stack>
#include <opencv2/opencv.hpp>


//参数：
//_binImp：输入二值图(只含0，1)
//_lableImg：输出标记的连通域
//返回连通域数目
int icvprCcaByTwoPass(const cv::Mat& _binImp, cv::Mat& _lableImg);

cv::Mat  icvprCcaBySeedFill(const cv::Mat& _binImg, int minArea, int maxArea);

//_lableImg：输入标记的连通域
cv::Mat drawBox(const cv::Mat& _labelImg, cv::Mat& _BoxImg);

cv::Scalar icvprGetRandomColor();

void icvprLabelColor(const cv::Mat&, cv::Mat&);
