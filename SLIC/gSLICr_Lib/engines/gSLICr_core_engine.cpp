// Copyright 2014-2015 Isis Innovation Limited and the authors of gSLICr

#pragma once
#include "gSLICr_core_engine.h"
#include <fstream>





using namespace cv;

using namespace gSLICr;
using namespace std;

gSLICr::engines::core_engine::core_engine(const objects::settings& in_settings)
{
	slic_seg_engine = new seg_engine_GPU(in_settings);
}

gSLICr::engines::core_engine::~core_engine()
{
		delete slic_seg_engine;
}

void gSLICr::engines::core_engine::Process_Frame(UChar4Image* in_img)
{
	slic_seg_engine->Perform_Segmentation(in_img);
}

const IntImage * gSLICr::engines::core_engine::Get_Seg_Res()
{
	return slic_seg_engine->Get_Seg_Mask();
}

void gSLICr::engines::core_engine::Draw_Segmentation_Result(UChar4Image* out_img)
{
	slic_seg_engine->Draw_Segmentation_Result(out_img);
}

void gSLICr::engines::core_engine::Write_Seg_Res_To_PGM(const char* fileName)
{
	const IntImage* idx_img = slic_seg_engine->Get_Seg_Mask();
	int width = idx_img->noDims.x;
	int height = idx_img->noDims.y;
	const int* data_ptr = idx_img->GetData(MEMORYDEVICE_CPU);

	ofstream f(fileName, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
	f << "P5\n" << width << " " << height << "\n65535\n";
	for (int i = 0; i < height * width; ++i)
	{
		ushort lable = (ushort)data_ptr[i];
		ushort lable_buffer = (lable << 8 | lable >> 8);
		// printf("%d,%d,%d",lable,height,width);
		// getchar();
		f.write((const char*)&lable_buffer, sizeof(ushort));
	}
	f.close();
}

Mat gSLICr::engines::core_engine::getSuperPixelsMap()
{
	const IntImage* idx_img = slic_seg_engine->Get_Seg_Mask();
	int width = idx_img->noDims.x;
	int height = idx_img->noDims.y;
	const int* data_ptr = idx_img->GetData(MEMORYDEVICE_CPU);

	Mat img1(height,width, CV_32F);
	// ofstream f(fileName, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
	// f << "P5\n" << width << " " << height << "\n65535\n";
	for (int i = 0; i < height * width; ++i)
	{
		img1.at<float>(i/width,i%width) = (ushort)data_ptr[i];
	}

	// double minv = 0.0, maxv = 0.0;
	// double* minp = &minv;
	// double* maxp = &maxv;
	// minMaxIdx(img1,minp,maxp);
	// printf("\n%f,%f,%f\n",minp,maxp,maxp);

	// 可视化 ///////////////////////////////////////////////////////////////////////////////
	// Mat B;
	// img1.convertTo(B, CV_8U, 255.0f/312.0f);//映射到0-255
	// imshow("segmentation", B);
	// printf("\n%d,%d,%d \n",B.at<uchar>(0,0),B.at<uchar>(200,1500),B.at<uchar>(1000,1910));

	return img1;
}

