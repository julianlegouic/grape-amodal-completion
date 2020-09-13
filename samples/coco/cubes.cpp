#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cubes.h"
#include <pybind11/stl.h>

#include <iostream>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core.hpp>
#include <string>
#include <opencv2/opencv.hpp>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

#include "opencv/cv.h"
#include "opencv/highgui.h"


using namespace std;
namespace py = pybind11;

vector<int> cubemap_dest0(vector<int> &img){
	
// 	cv::Mat img = cv::Mat(rows,cols,CV_8UC3,data);
	int sW,sH,cW,cH,dW,dH,R,dloc,sloc;
	double phi,theta,u,v;
// 	sW = img.cols;
// 	sH = img.rows;
	sW = 1920;
	sH = 960;
	
	cW = sW;
	cH = floor(cW/2);
	dW = cW;
	dH=cH;
	
  	R = floor(dW/8);

//   	cv::Mat dest0(cv::Size(2*R, 2*R), CV_8UC3, CV_RGB(0,0,0));
	vector<int> dest0(2*240*2*240*3);
//   	cv::Mat src = cv::Mat::ones(cH, cW, CV_8U);
//   	resize(img, src, src.size(), cv::INTER_CUBIC);

  	for (int y = 0; y < R; y+=1) {
    	              for (int x = 0; x < R; x+=1) {
        	              phi = atan((double)x / R);
            	          theta = atan(sqrt((double)x * (double)x + R * R) / (R - (double)y));
                	      u = floor(dW * phi / M_PI / 2);
                    	  v = floor(dH * theta / M_PI);

	                      dloc = (R - x - 1)     + y * 2 * R;
	                      sloc = (4 * R - u - 1) + v * 8 * R;
// 	                      dest0.data[3 * dloc]     = src.data[3 * sloc];
// 	                      dest0.data[3 * dloc + 1] = src.data[3 * sloc + 1];
// 	                      dest0.data[3 * dloc + 2] = src.data[3 * sloc + 2];
						  dest0[3*dloc] = img[3*sloc];
						  dest0[3*dloc + 1] = img[3*sloc + 1];
						  dest0[3*dloc + 2] = img[3*sloc + 2];
	
	                      dloc = (R - x - 1)     + (2 * R - y - 1) * 2 * R;
	                      sloc = (4 * R - u - 1) + (4 * R - v - 1) * 8 * R;
// 	                      dest0.data[3 * dloc]     = src.data[3 * sloc];
// 	                      dest0.data[3 * dloc + 1] = src.data[3 * sloc + 1];
// 	                      dest0.data[3 * dloc + 2] = src.data[3 * sloc + 2];
						  dest0[3*dloc] = img[3*sloc];
						  dest0[3*dloc + 1] = img[3*sloc + 1];
						  dest0[3*dloc + 2] = img[3*sloc + 2];
	
	                      dloc = (R + x)     + y * 2 * R;
	                      sloc = (4 * R + u) + v * 8 * R;
// 	                      dest0.data[3 * dloc]     = src.data[3 * sloc];
// 	                      dest0.data[3 * dloc + 1] = src.data[3 * sloc + 1];
// 	                      dest0.data[3 * dloc + 2] = src.data[3 * sloc + 2];
						  dest0[3*dloc] = img[3*sloc];
						  dest0[3*dloc + 1] = img[3*sloc + 1];
						  dest0[3*dloc + 2] = img[3*sloc + 2];
	
	                      dloc = (R + x)     + (2 * R - y - 1) * 2 * R;
	                      sloc = (4 * R + u) + (4 * R - v - 1) * 8 * R;
// 	                      dest0.data[3 * dloc]     = src.data[3 * sloc];
// 	                      dest0.data[3 * dloc + 1] = src.data[3 * sloc + 1];
// 	                      dest0.data[3 * dloc + 2] = src.data[3 * sloc + 2];
						  dest0[3*dloc] = img[3*sloc];
						  dest0[3*dloc + 1] = img[3*sloc + 1];
						  dest0[3*dloc + 2] = img[3*sloc + 2];
                    }
                  }

   
return dest0;
}

PYBIND11_PLUGIN(cubes){
	py::module m("cubes","cubes made by pybind11");
	m.def("cubemap_dest0",&cubemap_dest0);
	
	return m.ptr();
	}