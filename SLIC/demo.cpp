// Copyright 2014-2015 Isis Innovation Limited and the authors of gSLICr

#include <time.h>
#include <stdio.h>

#include "gSLICr_Lib/gSLICr.h"
#include "NVTimer.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"



using namespace std;
using namespace cv;



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




int main()
{

    VideoCapture cap;
    cap.open("../fast.mp4"); 
	// VideoCapture cap(0);

	// if (!cap.isOpened()) 
	// {
	// 	cerr << "unable to open camera!\n";
	// 	return -1;
	// }
	

	// gSLICr settings
	gSLICr::objects::settings my_settings;
	my_settings.img_size.x = 1920;
	my_settings.img_size.y = 1080;
	my_settings.no_segs = 2000;
	my_settings.spixel_size = 100;
	my_settings.coh_weight = 2;
	my_settings.no_iters = 5;
	my_settings.color_space = gSLICr::XYZ; // gSLICr::CIELAB for Lab, or gSLICr::RGB for RGB
	my_settings.seg_method = gSLICr::GIVEN_SIZE; // or gSLICr::GIVEN_NUM for given number
	my_settings.do_enforce_connectivity = true; // whether or not run the enforce connectivity step

	// instantiate a core_engine
	gSLICr::engines::core_engine* gSLICr_engine = new gSLICr::engines::core_engine(my_settings);

	// gSLICr takes gSLICr::UChar4Image as input and out put
	gSLICr::UChar4Image* in_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);
	gSLICr::UChar4Image* out_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);

	Size s(my_settings.img_size.x, my_settings.img_size.y);
	Mat oldFrame, frame;
	Mat boundry_draw_frame; boundry_draw_frame.create(s, CV_8UC3);

    StopWatchInterface *my_timer; sdkCreateTimer(&my_timer);

    int SUPERPIXELS_NUM = (my_settings.img_size.x/my_settings.spixel_size) * (my_settings.img_size.y/my_settings.spixel_size);
	
	int key; int save_count = 0;
	int frameCount =0;
	while (cap.read(oldFrame))
	{
		frameCount++;
		// if(frameCount < 143) continue;

		resize(oldFrame, frame, s);
		
		load_image(frame, in_img);
        
        sdkResetTimer(&my_timer); sdkStartTimer(&my_timer);
		gSLICr_engine->Process_Frame(in_img);
        sdkStopTimer(&my_timer); 
        cout<<"\rsegmentation in:["<<sdkGetTimerValue(&my_timer)<<"]ms"<<flush;
        
		gSLICr_engine->Draw_Segmentation_Result(out_img);
		
		load_image(out_img, boundry_draw_frame);
		// imshow("segmentation", boundry_draw_frame);

		// 获取分割的图像，把图像分为312类
		Mat semanticMap = gSLICr_engine->getSuperPixelsMap();
		// int minX = 2000, minY = 2000;
		// int maxX = -1, maxY = -1;
		// for(int i=0;i<semanticMap.cols;i++)  // i是x
		// 	for (int j = 0; j < semanticMap.rows; j++)
		// 	{

		// 		if(int(semanticMap.at<float>(j,i)) == 99 )
		// 		{
		// 			semanticMap.at<float>(j,i) = 312;
		// 			if(i < minX) minX = i;
		// 			if(j < minY) minY = j;
		// 			if(i > maxX) maxX = i;
		// 			if(j > maxY) maxY = j;
		// 		}
		// 		else
		// 		{
		// 			semanticMap.at<float>(j,i) = 0;
		// 		}
		// 	}


        sdkResetTimer(&my_timer); sdkStartTimer(&my_timer);


		// 提取超像素的最大矩形边缘
		int minX_[SUPERPIXELS_NUM] = {0};
		int minY_[SUPERPIXELS_NUM] = {0};
		int maxX_[SUPERPIXELS_NUM] = {0};
		int maxY_[SUPERPIXELS_NUM] = {0};
		for (int i = 0; i < SUPERPIXELS_NUM; i++)
		{
			minX_[i] = 2000;
			minY_[i] = 2000;
		}
		
		for(int i=0;i<semanticMap.cols;i++)  // i是x
			for (int j = 0; j < semanticMap.rows; j++)
			{

				int label_ = (int)semanticMap.at<float>(j,i);
				if(i < minX_[label_]) minX_[label_] = i;
				if(j < minY_[label_]) minY_[label_] = j;
				if(i > maxX_[label_]) maxX_[label_] = i;
				if(j > maxY_[label_]) maxY_[label_] = j;

			}

			

		
		// 可视化 ///////////////////////////////////////////////////////////////////////////////
		// Mat B;
		// semanticMap.convertTo(B, CV_8U, 255.0f/float(SUPERPIXELS_NUM));//映射到0-255
		
		// printf("\n%d,%d,%d \n",B.at<uchar>(0,0),B.at<uchar>(200,1500),B.at<uchar>(1000,1910));


		// 绘制矩形框
		// for(int i=0;i<SUPERPIXELS_NUM;i++)
		// {
		// 	rectangle(boundry_draw_frame,Point(minX_[i],minY_[i]),Point(maxX_[i],maxY_[i]),Scalar(155,155,155),3,8,0);
		// }
		
		// imshow("segmentation", boundry_draw_frame);

		// 图像裁剪 ////////////////////////////////////////////////////////
		for(int jj = 0; jj<SUPERPIXELS_NUM; jj++)
		{
			// jj = 182;
			Mat image_partSeg = semanticMap(cv::Rect(minX_[jj],minY_[jj],maxX_[jj]-minX_[jj],maxY_[jj]-minY_[jj]));
			Mat image_part = frame(cv::Rect(minX_[jj],minY_[jj],maxX_[jj]-minX_[jj],maxY_[jj]-minY_[jj])).clone();
			for(int i=0;i<image_partSeg.cols;i++)  // i是x
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
			//设置缩放后的图片的尺寸
			Size ResImgSiz = Size(96, 96);
			Mat ResImg = Mat(ResImgSiz, image_part.type());
			resize(image_part, ResImg, ResImgSiz, CV_INTER_CUBIC);

			// 保存patch到本地
			// char out_name[100];
			// sprintf(out_name, "photo3/img_%06i_%03i.jpg", save_count,jj);
			// imwrite(out_name, ResImg);
			save_count++;
			// break;
		}
        sdkStopTimer(&my_timer); 
        cout << endl <<"xxxxxxxxx in:["<<sdkGetTimerValue(&my_timer)<<"]ms"<<flush;
		// int tempIndex_ = 182;
		// rectangle(boundry_draw_frame,Point(minX_[tempIndex_],minY_[tempIndex_]),Point(maxX_[tempIndex_],maxY_[tempIndex_]),Scalar(155,155,155),3,8,0);
		imshow("boundry_draw_frame", boundry_draw_frame);


		getchar();
		
		cout << frameCount << endl;
		key = waitKey(1);



		// if (key == 27) break;
		// // else if (key == 's')
		// // {
		// 	char out_name[100];
		// 	sprintf(out_name, "photo/seg_%04i.pgm", save_count);
		// 	cout << out_name << endl;
		// 	gSLICr_engine->Write_Seg_Res_To_PGM(out_name);
		// 	sprintf(out_name, "photo/edge_%04i.png", save_count);
		// 	imwrite(out_name, boundry_draw_frame);
		// 	// sprintf(out_name, "img_%04i.png", save_count);
		// 	// imwrite(out_name, frame);
		// 	// printf("\nsaved segmentation %04i\n", save_count);
		// 	save_count++;
		// // }
	}

	destroyAllWindows();
    return 0;
}
