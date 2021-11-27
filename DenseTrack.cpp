#include <iostream>
#include <cmath>

#include "DenseTrack.h"
#include "Initialize.h"
#include "Descriptors.h"
#include "OpticalFlow.h"



#define OPENCV
#define GPU

#include "yolo_v2_class.hpp" //引用动态链接库中的头文件
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
 

int show_track = 1; // set show_track = 1, if you want to visualize the trajectories
char frame_num_string[4];

//一个判断点是否在框内的函数，目前没有用到
bool in_bbox(int point_x, int point_y,std::vector<bbox_t> result_vec )
{
	for (auto &i : result_vec) {
	        if(inbox(point_x,point_y,i.x,i.y,i.w,i.h))
            {
				return true;
				break;
			}
            else
				return false;

	}
}

void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names,
    int current_det_fps = -1, int current_cap_fps = -1)
{
    int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };
 
    for (auto &i : result_vec) {
        cv::Scalar color = obj_id_to_color(i.obj_id);
        //cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
        if (obj_names.size() > i.obj_id) {
            std::string obj_name = obj_names[i.obj_id];
            if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
            cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
            int const max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
			std::string person_string ("person");
        
			
			if(obj_name.compare (person_string)==0){
				cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);//框出检测物体
				
            //cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 30, 0)),
              //  cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
               // color, CV_FILLED, 8, 0);//在物体上方画出包含物体名称的框
			//打印物体名称
            //putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
			
			}
			
        }
    }
    if (current_det_fps >= 0 && current_cap_fps >= 0) {
        std::string fps_str = "FPS detection: " + std::to_string(current_det_fps) + "   FPS capture: " + std::to_string(current_cap_fps);
        putText(mat_img, fps_str, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
    }
}
 
std::vector<std::string> objects_names_from_file(std::string const filename) {
    std::ifstream file(filename);
    std::vector<std::string> file_lines;
    if (!file.is_open()) return file_lines;
    for (std::string line; getline(file, line);) file_lines.push_back(line);
    std::cout << "object names loaded \n";
    return file_lines;
}

//判断是否为抛投轨迹的评分函数
double score(double * input) {
    double var0;
    if ((input[2]) >= (0.45)) {
        if ((input[0]) >= (1.62)) {
            var0 = 0.5;
        } else {
            var0 = 1.7777778;
        }
    } else {
        var0 = -1.755102;
    }
    double var1;
    if ((input[2]) >= (0.45)) {
        if ((input[2]) >= (0.565)) {
            var1 = 0.9896138;
        } else {
            var1 = 0.18545206;
        }
    } else {
        if ((input[2]) >= (0.13499999)) {
            var1 = -0.96529984;
        } else {
            var1 = -0.08937744;
        }
    }
    double var2;
    var2 = (1.0) / ((1.0) + (exp((0.0) - ((var0) + (var1)))));
    //memcpy(output, (double[]){(1.0) - (var2), var2}, 2 * sizeof(double));
    double prob=1.0-var2;
    return prob;
}

int main(int argc, char** argv)
{
	 int video_fps = 25;
	 std::string out_videofile = "output_files/result_video/result_test_clas.avi";
    bool const save_output_videofile = true;   // true - for saving history
	
	//yolo 检测器文件配置
	std::string names_file = "yolo_files/coco.names";
    std::string cfg_file = "yolo_files/yolov4.cfg";
    std::string weights_file = "yolo_files/yolov4.weights";
	
	//初始化检测器
    Detector detector(cfg_file, weights_file, 0);

	 //读入分类对象文件
	auto obj_names = objects_names_from_file(names_file);


	VideoCapture capture;
	char* video = argv[1];
	int flag = arg_parse(argc, argv);
	capture.open(video);

	if(!capture.isOpened()) {
		fprintf(stderr, "Could not initialize capturing..\n");
		return -1;
	}

	int frame_num = 0;
	TrackInfo trackInfo;
	DescInfo hogInfo, hofInfo, mbhInfo;

	InitTrackInfo(&trackInfo, track_length, init_gap);
	InitDescInfo(&hogInfo, 8, false, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&hofInfo, 9, true, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&mbhInfo, 8, false, patch_size, nxy_cell, nt_cell);

	SeqInfo seqInfo;
	InitSeqInfo(&seqInfo, video);



	if(flag)
		seqInfo.length = end_frame - start_frame + 1;



	if(show_track == 1)
		namedWindow("DenseTrack", 0);

	Mat image, prev_grey, grey;

	std::vector<float> fscales(0);
	std::vector<Size> sizes(0);

	std::vector<Mat> prev_grey_pyr(0), grey_pyr(0), flow_pyr(0);
	std::vector<Mat> prev_poly_pyr(0), poly_pyr(0); // for optical flow

	std::vector<std::list<Track> > xyScaleTracks;
	int init_counter = 0; // indicate when to detect new feature points
	
	//检测视频保存
	cv::VideoWriter output_video;
	int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);   //宽和高保持不变
    int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);

	#ifdef CV_VERSION_EPOCH // OpenCV 2.x
                video_fps = capture.get(CV_CAP_PROP_FPS);
		#else
                video_fps = capture.get(cv::CAP_PROP_FPS);
		#endif

	 if (save_output_videofile)
		#ifdef CV_VERSION_EPOCH // OpenCV 2.x
                    output_video.open(out_videofile, CV_FOURCC('D', 'I', 'V', 'X'), std::max(30, video_fps), Size(frame_width,frame_height), true);
		#else
                    output_video.open(out_videofile, cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), std::max(30, video_fps), Size(frame_width,frame_height), true);
		#endif
	
    //printf("(1,1) \t \n");
	while(true) {
		Mat frame;
		int i, j, c;

		// get a new frame
		capture >> frame;

        //yolo检测
		std::vector<bbox_t> result_vec = detector.detect(frame);

        draw_boxes(frame, result_vec, obj_names);//只对人进行框出

		if(frame.empty())
			break;

		if(frame_num < start_frame || frame_num > end_frame) {
			frame_num++;
			continue;
		}

		 /*-----------------------对第一帧做处理-------------------------*/
        //由于光流需要两帧进行计算，故第一帧不计算光流
		if(frame_num == start_frame) {
			image.create(frame.size(), CV_8UC3);
			grey.create(frame.size(), CV_8UC1);
			prev_grey.create(frame.size(), CV_8UC1);

			InitPry(frame, fscales, sizes);

			BuildPry(sizes, CV_8UC1, prev_grey_pyr);
			BuildPry(sizes, CV_8UC1, grey_pyr);

			BuildPry(sizes, CV_32FC2, flow_pyr);
			BuildPry(sizes, CV_32FC(5), prev_poly_pyr);
			BuildPry(sizes, CV_32FC(5), poly_pyr);

			xyScaleTracks.resize(scale_num);

			frame.copyTo(image);
			cvtColor(image, prev_grey, CV_BGR2GRAY);

			//对于每个图像尺度分别密集采样特征点
			for(int iScale = 0; iScale < scale_num; iScale++) {
			

				if(iScale == 0)
					prev_grey.copyTo(prev_grey_pyr[0]);
				else
					resize(prev_grey_pyr[iScale-1], prev_grey_pyr[iScale], prev_grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);

				// dense sampling feature points
				std::vector<Point2f> points(0);
				DenseSample(prev_grey_pyr[iScale], points, quality, min_distance);
				
				

				// save the feature points
				std::list<Track>& tracks = xyScaleTracks[iScale];
				for(i = 0; i < points.size(); i++)
					tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
			}

			// compute polynomial expansion计算多项式展开
			my::FarnebackPolyExpPyr(prev_grey, prev_poly_pyr, fscales, 7, 1.5);

			frame_num++;
			continue;
		}

		init_counter++;
		frame.copyTo(image);
		cvtColor(image, grey, CV_BGR2GRAY);

		// compute optical flow for all scales once
		my::FarnebackPolyExpPyr(grey, poly_pyr, fscales, 7, 1.5);
		my::calcOpticalFlowFarneback(prev_poly_pyr, poly_pyr, flow_pyr, 10, 2);

        //在每个尺度分别计算特征
		for(int iScale = 0; iScale < scale_num; iScale++) {
			//尺度0不缩放，其余尺度使用插值方法缩放
			if(iScale == 0)
				grey.copyTo(grey_pyr[0]);
			else
				resize(grey_pyr[iScale-1], grey_pyr[iScale], grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);

			int width = grey_pyr[iScale].cols;
			int height = grey_pyr[iScale].rows;

			// compute the integral histograms计算积分直方图
			DescMat* hogMat = InitDescMat(height+1, width+1, hogInfo.nBins);
			HogComp(prev_grey_pyr[iScale], hogMat->desc, hogInfo);

			DescMat* hofMat = InitDescMat(height+1, width+1, hofInfo.nBins);
			HofComp(flow_pyr[iScale], hofMat->desc, hofInfo);

			DescMat* mbhMatX = InitDescMat(height+1, width+1, mbhInfo.nBins);
			DescMat* mbhMatY = InitDescMat(height+1, width+1, mbhInfo.nBins);
			MbhComp(flow_pyr[iScale], mbhMatX->desc, mbhMatY->desc, mbhInfo);

			// track feature points in each scale separately分别跟踪每个尺度中的特征点
			std::list<Track>& tracks = xyScaleTracks[iScale];
			int k=0;
			for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end();iTrack++) {
				int index = iTrack->index;
				Point2f prev_point = iTrack->point[index];
				
				int x = std::min<int>(std::max<int>(cvRound(prev_point.x), 0), width-1);
				int y = std::min<int>(std::max<int>(cvRound(prev_point.y), 0), height-1);

				Point2f point;
				point.x = prev_point.x + flow_pyr[iScale].ptr<float>(y)[2*x];
				point.y = prev_point.y + flow_pyr[iScale].ptr<float>(y)[2*x+1];
			

				if(point.x <= 0 || point.x >= width || point.y <= 0 || point.y >= height) {
					iTrack = tracks.erase(iTrack);
					continue;
				}

				// get the descriptors for the feature point
				RectInfo rect;
				GetRect(prev_point, rect, width, height, hogInfo);
				GetDesc(hogMat, rect, hogInfo, iTrack->hog, index);
				GetDesc(hofMat, rect, hofInfo, iTrack->hof, index);
				GetDesc(mbhMatX, rect, mbhInfo, iTrack->mbhX, index);
				GetDesc(mbhMatY, rect, mbhInfo, iTrack->mbhY, index);
				iTrack->addPoint(point);
 				



				// draw the trajectories at the first scale在原始尺度上可视化轨迹
				//if(show_track == 1 && iScale == 0)
					//DrawTrack(iTrack->point, iTrack->index, fscales[iScale], image);
					//DrawCircle(iTrack->point, iTrack->index, fscales[iScale], image);

				// if the trajectory achieves the maximal length
				 // 若轨迹的长度达到了预设长度,在iDT中应该是设置为15
                // 达到长度后就可以输出各个特征了
				if(iTrack->index >= trackInfo.length) {
					std::vector<Point2f> trajectory(trackInfo.length+1);
					for(int i = 0; i <= trackInfo.length; ++i)
						trajectory[i] = iTrack->point[i]*fscales[iScale];

				

					float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);
					//IsValid(trajectory, mean_x, mean_y, var_x, var_y, length)&&(fscales[iScale]==1)
					double mean_cos_distance(0),  var_cos_distance(0), max_cos_distance(0), min_cos_distance(2);
					if(IsValid(trajectory, mean_x, mean_y, var_x, var_y, length,mean_cos_distance, var_cos_distance, max_cos_distance, min_cos_distance)&&length>10&&(fscales[iScale]==1)) {
						//给输出轨迹编号
						//orientation(trajectory, mean_cos_distance, var_cos_distance, max_cos_distance, min_cos_distance);
						DrawTrack(iTrack->point, iTrack->index, fscales[iScale], image);
					    DrawCircle(iTrack->point, iTrack->index, fscales[iScale], image);

						char str[4];
				        sprintf(str, "%d", k);
						putText(image,  str, cv::Point2f(mean_x, mean_y), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(255, 0, 0), 1);
						//printf("0 1:%.3f 2:%.3f 3:%.3f 4:%.3f 5:%.3f 6:%.3f frame:%d num:%d\n", min_start_distance, normalized_min_start_distance ,mean_x, mean_y,var_x, var_y, length, fscales[iScale]);
                        //k++;
						

						// for spatio-temporal pyramid
						//printf("%f\t", std::min<float>(std::max<float>(mean_x/float(seqInfo.width), 0), 0.999));
						//printf("%f\t", std::min<float>(std::max<float>(mean_y/float(seqInfo.height), 0), 0.999));
						//printf("%f\n", std::min<float>(std::max<float>((frame_num - trackInfo.length/2.0 - start_frame)/float(seqInfo.length), 0), 0.999));
					
					
                        
			           //找到相距轨迹起点和终点最近的框
						double temp_delta_x=trajectory[0].x-result_vec[0].x-0.5*result_vec[0].w;
						double temp_delta_y=trajectory[0].y-result_vec[0].y-0.5*result_vec[0].h;
						double min_start_distance=sqrt(temp_delta_x*temp_delta_x + temp_delta_y*temp_delta_y);
                        unsigned int box_x=result_vec[0].x+0.5*result_vec[0].w;
						unsigned int box_y=result_vec[0].y+0.5*result_vec[0].h;
						unsigned int box_w=result_vec[0].w;
						unsigned int box_h=result_vec[0].h;
						//std::string person1_string ("person");
						

						for (auto &i : result_vec) {
							//std::string obj1_name = obj_names[i.obj_id];  //person1_string.compare (obj1_name)==0
							if(1){
								double delta_x=trajectory[0].x-i.x-0.5*i.w;
								double delta_y=trajectory[0].y-i.y-0.5*i.h;
								double start_distance=sqrt(delta_x*delta_x + delta_y*delta_y);
								if(min_start_distance>start_distance)
								{
										box_x=i.x+0.5*i.w;
										box_y=i.y+0.5*i.h;
										box_w=i.w;
										box_h=i.h;
								}
								min_start_distance=(min_start_distance<start_distance)?min_start_distance:start_distance;
							}
	                    }
		                
						//轨迹终点相距人体距离
						double end_delta_x=trajectory[trackInfo.length-1].x-box_x;
						double end_delta_y=trajectory[trackInfo.length-1].y-box_y;
						double min_end_distance=sqrt(end_delta_x*end_delta_x + end_delta_y*end_delta_y);
					    double normalized_min_end_distance=min_end_distance/box_w;
                        //轨迹起点相距人体距离
						double normalized_min_start_distance=min_start_distance/box_w;
						double normalized_length=length/box_w;
						//printf("start distance of this trajectory is%f\n",min_start_distance);
						//printf("归一化的轨迹起始距离%f\n",normalized_min_start_distance);
						//printf("归一化的轨迹终点距离%f\n",normalized_min_end_distance);
						//printf("归一化的轨迹长度%f\n",normalized_length);
						//printf("轨迹的余弦距离%f\n",mean_cos_distance);
						//printf("轨迹的余弦距离方差%f\n",var_cos_distance);
						//printf("轨迹的最长余弦距离%f\n",max_cos_distance);
						//printf("轨迹的最短余弦距离%f\n",min_cos_distance);
						//printf("the closeet bbox locates at%d\t%d\n",box_x,box_y);
                        
						/*判断抛投轨迹并画出
						double input[3]={normalized_min_start_distance,0,normalized_length};
						double prob=score(input);
						if(prob>0.5)
                              DrawThrowTrack(iTrack->point, iTrack->index, fscales[iScale], image);
                         
						*/

		            	printf("0 1:%.3f 2:%.3f 3:%.3f 4:%.3f 5:%.3f 6:%.3f \t frame:%d num:%d\n",normalized_min_start_distance ,normalized_min_end_distance,normalized_length,mean_cos_distance,var_cos_distance,max_cos_distance,frame_num, k);
						k++;
						
						
			
					}

					iTrack = tracks.erase(iTrack);
					continue;
				}
				++iTrack;
			}
			ReleDescMat(hogMat);
			ReleDescMat(hofMat);
			ReleDescMat(mbhMatX);
			ReleDescMat(mbhMatY);

			if(init_counter != trackInfo.gap)
				continue;

			// detect new feature points every initGap frames
			//在每个间隙帧中检测新特征点
			std::vector<Point2f> points(0);
			for(std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); iTrack++)
				points.push_back(iTrack->point[iTrack->index]);

			DenseSample(grey_pyr[iScale], points, quality, min_distance);
		
			// save the new feature points
			for(i = 0; i < points.size(); i++)
				tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
		
		}

       //这里有好多个copyTo prev_xxx
        //因为计算光流，surf匹配等都需要上一帧的信息，故在每帧处理完后保存该帧信息，用作下一帧计算时用
		init_counter = 0;
		grey.copyTo(prev_grey);
		for(i = 0; i < scale_num; i++) {
			grey_pyr[i].copyTo(prev_grey_pyr[i]);
			poly_pyr[i].copyTo(prev_poly_pyr[i]);
		}

		frame_num++;
        output_video.write(image);
		if( show_track == 1 ) {
			//在输出视频上打印当前帧数
			 sprintf(frame_num_string, "%d", frame_num);
			putText(image, frame_num_string,cv::Point2f(10,10) , cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(255, 0, 0), 1);
			
			imshow( "DenseTrack", image);
			//3毫秒后显示窗口关闭
			c = cvWaitKey(3);
			//将图像逐帧输出
			output_video<<image;
			
			//按esc退出程序
			if((char)c == 27) 
			break;
		   
		}
	}





    output_video.release();

	if( show_track == 1 )
		destroyWindow("DenseTrack");
   
	return 0;
}