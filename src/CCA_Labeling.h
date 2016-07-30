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


//������
//_binImp�������ֵͼ(ֻ��0��1)
//_lableImg�������ǵ���ͨ��
//������ͨ����Ŀ
int icvprCcaByTwoPass(const cv::Mat& _binImp, cv::Mat& _lableImg);

cv::Mat  icvprCcaBySeedFill(const cv::Mat& _binImg, int minArea, int maxArea);

//_lableImg�������ǵ���ͨ��
cv::Mat drawBox(const cv::Mat& _labelImg, cv::Mat& _BoxImg);

cv::Scalar icvprGetRandomColor();

void icvprLabelColor(const cv::Mat&, cv::Mat&);
