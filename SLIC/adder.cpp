#include <time.h>
#include <stdio.h>

#include "gSLICr_Lib/gSLICr.h"
#include "NVTimer.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include <string>

using namespace std;
using namespace cv;

// #include <stdio.h>
// #include <iostream>
// using namespace std;
// int add_integer(int a , int b);
// float add_float(float a, float b);
// Mat getPicture();


// int add_integer(int a , int b){
//   return a + b;
// }
// float add_float(float a, float b){
//   return a + b;
// }
// extern "C"{
//   int add_integer_plus(int a , int b){
//     return add_integer(a, b);
//   }
//   float add_float_plus(float a, float b){
//     return add_float(a ,b);
//   }

//   Mat getPicture_plus()
//   {
//     return getPicture();
//   }
// }


// cv::BackgroundSubtractorMOG2 *mog = cv::createBackgroundSubtractorMOG2 (500, 16, false);

// extern "C" void getfg(int rows, int cols, unsigned char* imgData,
//         unsigned char *fgD) {
//     cv::Mat img(rows, cols, CV_8UC3, (void *) imgData);
//     cv::Mat fg(rows, cols, CV_8UC1, fgD);
//     mog->apply(img, fg);
// }



// Mat getPicture()
// {
//   Mat oldFrame, frame;
//   VideoCapture cap;
//   cap.open("../fast.mp4"); 
//   // while(1)
//   // {
//   //   cap.read(oldFrame);
//   //   imshow("boundry_draw_frame", oldFrame);
//   //   waitKey(1);
//   // }
//   cout << "xxxxxxxxxxxxx" << endl;

//   return oldFrame;

// }

 
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>                   
#include   <stdlib.h>   
#define DLLEXPORT extern "C" 

using namespace cv;


cv::Mat transformImageFromPy(int height, int width, uchar* data)
{
  cv::Mat src(height, width, CV_8UC3, data);
  return src;
}

cv::Mat transformImageFromPyOneChannel(int height, int width, uchar* data)
{
  cv::Mat src(height, width, CV_8UC1, data);
  return src;
}

uchar* transformImageFromCplus(cv::Mat dst)
{
	uchar* buffer = (uchar*)malloc(sizeof(uchar)*dst.rows*dst.cols*3);
	memcpy(buffer, dst.data, dst.rows*dst.cols*3);
  return buffer;
}


uchar* transformImageFromCplusOneChannel(cv::Mat dst)
{
	uchar* buffer = (uchar*)malloc(sizeof(uchar)*dst.rows*dst.cols);
	memcpy(buffer, dst.data, dst.rows*dst.cols);
  return buffer;
}

void load_image(const Mat& inimg, gSLICr::UChar4Image* outimg)
{
	gSLICr::Vector4u* outimg_ptr = outimg->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < outimg->noDims.y;y++)
		for (int x = 0; x < outimg->noDims.x; x++)
		{
			int idx = x + y * outimg->noDims.x;
			outimg_ptr[idx].b = inimg.at<Vec3b>(y, x)[0];
			outimg_ptr[idx].g = inimg.at<Vec3b>(y, x)[1];
			outimg_ptr[idx].r = inimg.at<Vec3b>(y, x)[2];
		}
}

void load_image(const gSLICr::UChar4Image* inimg, Mat& outimg)
{
	const gSLICr::Vector4u* inimg_ptr = inimg->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < inimg->noDims.y; y++)
		for (int x = 0; x < inimg->noDims.x; x++)
		{
			int idx = x + y * inimg->noDims.x;
			outimg.at<Vec3b>(y, x)[0] = inimg_ptr[idx].b;
			outimg.at<Vec3b>(y, x)[1] = inimg_ptr[idx].g;
			outimg.at<Vec3b>(y, x)[2] = inimg_ptr[idx].r;
		}
}


//CV_8UC3????????????
DLLEXPORT  uchar* getSuperPixelMap(int height, int width, uchar* data) {
	gSLICr::objects::settings my_settings;
	my_settings.img_size.x = width;
	my_settings.img_size.y = height;
	my_settings.no_segs = 2000;
	my_settings.spixel_size = 100;
	my_settings.coh_weight = 2;
	my_settings.no_iters = 5;
	my_settings.color_space = gSLICr::XYZ; // gSLICr::CIELAB for Lab, or gSLICr::RGB for RGB
	my_settings.seg_method = gSLICr::GIVEN_SIZE; // or gSLICr::GIVEN_NUM for given number
	my_settings.do_enforce_connectivity = true; // whether or not run the enforce connectivity step

  // instantiate a core_engine
	gSLICr::engines::core_engine* gSLICr_engine = new gSLICr::engines::core_engine(my_settings);
  gSLICr::UChar4Image* in_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);
  gSLICr::UChar4Image* out_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);
  Mat frame;
  Size s(my_settings.img_size.x, my_settings.img_size.y);

	cv::Mat rawImage = transformImageFromPy(height,width,data);

  resize(rawImage, frame, s);
  load_image(frame, in_img);
  gSLICr_engine->Process_Frame(in_img);

  gSLICr_engine->Draw_Segmentation_Result(out_img);
	
  // ??????????????????????????????
  Mat boundry_draw_frame; boundry_draw_frame.create(s, CV_8UC3);
  load_image(out_img, boundry_draw_frame);

  Mat semanticMap = gSLICr_engine->getSuperPixelsMap();

  Mat semanticMap8U;
  semanticMap.convertTo(semanticMap8U, CV_8U);
  // cout << semanticMap.channels() << endl;

  // imshow("boundry_draw_frame", rawImage);
  // waitKey(2000);
  uchar* cc = transformImageFromCplus(semanticMap8U);
  // cout << cc << endl;
  // print(cc);
	return cc;

}


vector<cv::Mat> patches;
Mat semanticMap8U;
DLLEXPORT  void processImageBySLC(int height, int width, uchar* data) {
	gSLICr::objects::settings my_settings;
	my_settings.img_size.x = width;
	my_settings.img_size.y = height;
	my_settings.no_segs = 2000;
	my_settings.spixel_size = 100;
	my_settings.coh_weight = 2;
	my_settings.no_iters = 5;
	my_settings.color_space = gSLICr::XYZ; // gSLICr::CIELAB for Lab, or gSLICr::RGB for RGB
	my_settings.seg_method = gSLICr::GIVEN_SIZE; // or gSLICr::GIVEN_NUM for given number
	my_settings.do_enforce_connectivity = true; // whether or not run the enforce connectivity step
    int SUPERPIXELS_NUM = (my_settings.img_size.x/my_settings.spixel_size) * (my_settings.img_size.y/my_settings.spixel_size);
	
  // instantiate a core_engine
  static gSLICr::engines::core_engine* gSLICr_engine = new gSLICr::engines::core_engine(my_settings);
  static gSLICr::UChar4Image* in_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);
  static gSLICr::UChar4Image* out_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);
  Mat frame;
  Size s(my_settings.img_size.x, my_settings.img_size.y);

  cv::Mat rawImage = transformImageFromPy(height,width,data);

  resize(rawImage, frame, s);
  load_image(frame, in_img);
  gSLICr_engine->Process_Frame(in_img);

  gSLICr_engine->Draw_Segmentation_Result(out_img);
	
  // ??????????????????????????????
  Mat boundry_draw_frame; boundry_draw_frame.create(s, CV_8UC3);
  load_image(out_img, boundry_draw_frame);

  Mat semanticMap = gSLICr_engine->getSuperPixelsMap();

  
  semanticMap.convertTo(semanticMap8U, CV_8U);


// ????????????????????????????????????
		int minX_[SUPERPIXELS_NUM] = {0};
		int minY_[SUPERPIXELS_NUM] = {0};
		int maxX_[SUPERPIXELS_NUM] = {0};
		int maxY_[SUPERPIXELS_NUM] = {0};
		for (int i = 0; i < SUPERPIXELS_NUM; i++)
		{
			minX_[i] = 2000;
			minY_[i] = 2000;
		}
		
		for(int i=0;i<semanticMap.cols;i++)  // i???x
			for (int j = 0; j < semanticMap.rows; j++)
			{

				int label_ = (int)semanticMap.at<float>(j,i);
				if(i < minX_[label_]) minX_[label_] = i;
				if(j < minY_[label_]) minY_[label_] = j;
				if(i > maxX_[label_]) maxX_[label_] = i;
				if(j > maxY_[label_]) maxY_[label_] = j;

			}


		// ???????????? ////////////////////////////////////////////////////////
		for(int jj = 0; jj<SUPERPIXELS_NUM; jj++)
		{
			// jj = 182;
			Mat image_partSeg = semanticMap(cv::Rect(minX_[jj],minY_[jj],maxX_[jj]-minX_[jj],maxY_[jj]-minY_[jj]));
			Mat image_part = frame(cv::Rect(minX_[jj],minY_[jj],maxX_[jj]-minX_[jj],maxY_[jj]-minY_[jj])).clone();
			for(int i=0;i<image_partSeg.cols;i++)  // i???x
				for (int j = 0; j < image_partSeg.rows; j++)
				{
					int label_ = (int)image_partSeg.at<float>(j,i);
					if(label_ != jj)
					{
						// cout<< image_part.type() <<endl;
						// Vec3i bgr = image_part.at<Vec3b>(j,i);
						// cout << bgr[0] << endl;
						image_part.at<Vec3b>(j,i)[0] = NAN;
						image_part.at<Vec3b>(j,i)[1] = NAN;
						image_part.at<Vec3b>(j,i)[2] = NAN;
						// getchar();
					}

				}
			//?????????????????????????????????
			Size ResImgSiz = Size(96, 96);
			Mat ResImg = Mat(ResImgSiz, image_part.type());
			resize(image_part, ResImg, ResImgSiz, CV_INTER_CUBIC);
      		patches.push_back(ResImg);


		}

	// delete gSLICr_engine;
	// gSLICr_engine = 0;
	// delete in_img;
	// in_img = 0;
	// delete out_img;
	// out_img = 0;

  	// uchar* cc = transformImageFromCplus(semanticMap8U);
	// return cc;

}



//??????????????????????????????????????????????????????
Mat color_bar = Mat(1, 13 * 3 * 3, CV_8UC3);//??????
unsigned char *pBar = color_bar.data;
void Create_ColorBar() {//?????????????????????
						//?????????????????????H: [0, 180]  S : [0, 255] V : [0, 255],???????????????13*3*3=117?????????
	int H[13] = { 180,120,60,160,100,40,150,90,30,140,80,20,10 };
	int S[3] = { 255,100,30 };
	int V[3] = { 255,180,90 };
 
 
	Mat color = Mat(1, 13 * 3 * 3, CV_8UC3);//???????????????????????????
	unsigned char *pColor = color.data;
 
 
	int h = 0, s = 0, v = 0;
	for (int ba = 0; ba < 13 * 3 * 3; v++, s++, h++, ba++) {
		if (h == 13) h = 0;
		if (s == 3 * 13) s = 0;
		if (v == 3 * 13 * 3) v = 0;
		pColor[ba * 3 + 0] = H[h];
		pColor[ba * 3 + 1] = S[s / 13];
		pColor[ba * 3 + 2] = V[v / 13 / 3];
	}
	cvtColor(color, color_bar, CV_HSV2BGR);
}



DLLEXPORT  uchar* processImageFromLabels(int height, int width, uchar* data) {
	// Create_ColorBar();
  	cv::Mat labelV = transformImageFromPyOneChannel(height,width,data);
	vector<Vec3b> colorList;
	colorList.push_back(Vec3b(255,0,0)); 	//???1
	colorList.push_back(Vec3b(255,165,0));  //???2
	colorList.push_back(Vec3b(255,255,0));  //???3
	colorList.push_back(Vec3b(0,255,0));	//???4
	colorList.push_back(Vec3b(0,127,255));	//???5
	colorList.push_back(Vec3b(0,0,255));	//???6
	colorList.push_back(Vec3b(139,0,255));	//???7
	colorList.push_back(Vec3b(255,255,255));//???8
	colorList.push_back(Vec3b(30,105,210)); //?????????9
	colorList.push_back(Vec3b(147,20,255)); //?????????10
	colorList.push_back(Vec3b(238,130,238));//?????????11
	colorList.push_back(Vec3b(230,216,173));//?????????12


	Mat resizeLabel = Mat(semanticMap8U.rows/100, semanticMap8U.cols/100, CV_8UC1);

	cout << semanticMap8U.rows/100 << ", " << semanticMap8U.cols/100 << endl;

	
	for (int j = 0; j < resizeLabel.rows; j++) //rows???, cols???
		for(int i=0;i<resizeLabel.cols;i++)  // i???x
	{
		resizeLabel.at<uchar>(j,i) = labelV.at<uchar>(i+j*resizeLabel.cols,0);
	}

	for(int times=0;times<5;times++)//????????????
		for (int j = 0; j < resizeLabel.rows; j++) //rows???, cols???
			for(int i=0;i<resizeLabel.cols;i++)  // i???x
		{
			int count = 0;
			int specital = 0;
			if(j>0)
			{
				if(resizeLabel.at<uchar>(j,i) == resizeLabel.at<uchar>(j-1,i)) count++; else;
			}
			else
			{
				count++;
				specital++;
			}
				

			if(j<resizeLabel.rows-1)
			{
				if(resizeLabel.at<uchar>(j,i) == resizeLabel.at<uchar>(j+1,i)) count++; else;
			}
			else
			{
				count++;
				specital++;
			}

			if(i>0)
			{
				if(resizeLabel.at<uchar>(j,i) == resizeLabel.at<uchar>(j,i-1)) count++; else;
			}
			else
			{
				count++;
				specital++;
			}

			if(i<resizeLabel.cols-1)
			{
				if(resizeLabel.at<uchar>(j,i) == resizeLabel.at<uchar>(j,i+1)) count++; else;
			}
			else
			{
				count++;
				specital++;
			}

			if(specital == 0)
			{
				if(resizeLabel.at<uchar>(j,i) == resizeLabel.at<uchar>(j+1,i+1)) count++; else;
				if(resizeLabel.at<uchar>(j,i) == resizeLabel.at<uchar>(j-1,i+1)) count++; else;
				if(resizeLabel.at<uchar>(j,i) == resizeLabel.at<uchar>(j+1,i-1)) count++; else;
				if(resizeLabel.at<uchar>(j,i) == resizeLabel.at<uchar>(j-1,i-1)) count++; else;
			}

			



			if(count <= 1)
			{
				if(i == resizeLabel.cols-1)
				{
					resizeLabel.at<uchar>(j,i) = resizeLabel.at<uchar>(j,i-1);
				}
				else
				{
					resizeLabel.at<uchar>(j,i) = resizeLabel.at<uchar>(j,i+1);
				}
				
				
			}

			if(specital == 2)
			{
				if(count <= 2)
				{
					if(i == resizeLabel.cols-1)
					{
						resizeLabel.at<uchar>(j,i) = resizeLabel.at<uchar>(j,i-1);
					}
					else
					{
						resizeLabel.at<uchar>(j,i) = resizeLabel.at<uchar>(j,i+1);
					}
					
					
				}
			}
				

			// resizeLabel.at<uchar>(j,i) = labelV.at<uchar>(i+j*resizeLabel.cols,0);
		}	


	Mat labelV2 = Mat(labelV.rows, 	1, CV_8UC1);
	for (int j = 0; j < resizeLabel.rows; j++) //rows???, cols???
		for(int i=0;i<resizeLabel.cols;i++)  // i???x
	{
		labelV2.at<uchar>(i+j*resizeLabel.cols,0) = resizeLabel.at<uchar>(j,i);
	}

	// cout << resizeLabel << endl;
	//semanticMap8U.rows 1080
	//semanticMap8U.cols 1920
	Mat resultMap = Mat(semanticMap8U.rows, semanticMap8U.cols, CV_8UC3);


	//???????????????label??????
	for(int i=0;i<semanticMap8U.cols;i++)  // i???x
		for (int j = 0; j < semanticMap8U.rows; j++) //rows???, cols???
		{
			int lab_ = labelV2.at<uchar>(semanticMap8U.at<uchar>(j,i),0);
			
			resultMap.at<Vec3b>(j,i) = colorList[lab_];
		}


	colorList.clear();
  	uchar* cc = transformImageFromCplus(resultMap);
	return cc;

}



DLLEXPORT  uchar* processImageFromLabels2(int height, int width, uchar* data) {
	// Create_ColorBar();
  	cv::Mat labelV = transformImageFromPyOneChannel(height,width,data);

	Mat resultMap = Mat(semanticMap8U.rows, semanticMap8U.cols, CV_8UC1);
	//???????????????label??????
	for(int i=0;i<semanticMap8U.cols;i++)  // i???x
		for (int j = 0; j < semanticMap8U.rows; j++) //rows???, cols???
		{
			int lab_ = labelV.at<uchar>(semanticMap8U.at<uchar>(j,i),0);
			
			resultMap.at<uchar>(j,i) = lab_;
		}


  	uchar* cc = transformImageFromCplus(resultMap);
	return cc;

}



// CV_8UC3????????????
DLLEXPORT  uchar* getPatches(int index) {
  Mat combine = patches[index];
//   for (int i = 1; i < patches.size(); i++)
//   {
// 	  hconcat(combine,patches[i],combine);
//   }
  uchar* cc = transformImageFromCplus(combine);
  return cc;
}



DLLEXPORT void release(uchar* data) {
	free(data);
}


DLLEXPORT void clearVector() {
	patches.clear();
}
