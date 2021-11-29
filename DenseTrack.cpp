#include<stdio.h>
#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>


#define OPENCV
#define GPU


#include "DenseTrack.h"
#include "Initialize.h"
#include "Descriptors.h"
#include "OpticalFlow.h"
#include "IOUtracker.hpp"
#include "yolo_v2_class.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracker.hpp>
#include "opencv2/highgui/highgui.hpp"

 
// set show_track = 1, if you want to visualize the trajectories
int show_track = 1; 


  // initiate active tracks and finished tracks
std::vector<Track_box> active_tracks;
std::vector<Track_box> finished_tracks;
std::vector<cv::Rect2d> track_boxes;




// tracker thresholds   
float sigma_l = 0;
float sigma_h = 0.3;
float sigma_iou = 0.2;
int t_min = 3;

 int id=0;
 int revive=0;
 int ttl=30;
char frame_num_string[6];



 
std::vector<std::string> objects_names_from_file(std::string const filename) 
{
    std::ifstream file(filename);
    std::vector<std::string> file_lines;
    if (!file.is_open()) return file_lines;
    for (std::string line; getline(file, line);) file_lines.push_back(line);
    std::cout << "object names loaded \n";
    return file_lines;
}


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
	 std::string out_videofile = "output_files/result_video/throw/kd_8_8_4_1_throw.avi";
    bool const save_output_videofile = true;   // true - for saving history
	
	//Detector initialization
	std::string names_file = "yolo_files/coco.names";
    std::string cfg_file = "yolo_files/yolov4.cfg";
    std::string weights_file = "yolo_files/yolov4.weights";
	

    Detector detector(cfg_file, weights_file, 0);

    //Create multitracker 
	cv::Ptr<cv::MultiTracker> multiTracker = cv::MultiTracker::create();

    //Load opencv tracker
    //cv::Ptr<cv::TrackerMIL> tracker= cv::TrackerMIL::create();
    //Ptr<TrackerTLD> tracker= TrackerTLD::create();
    //cv::Ptr<cv::TrackerKCF> tracker1 = cv::TrackerKCF::create();
    //cv::Ptr<cv::TrackerMedianFlow> tracker = cv::TrackerMedianFlow::create();
    //Ptr<TrackerBoosting> tracker= TrackerBoosting::create();

	bool updated;     // Whether if a track was updated or not
    int IOU_index;      // Index of the box with the highest IOU
    int track_id = 0;// Starting ID for the Tracks

	 //读入分类对象文件
	auto obj_names = objects_names_from_file(names_file);


	cv::VideoCapture capture;
	char* video = argv[1];
	int video_flag = arg_parse(argc, argv);
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



	if(video_flag)
		seqInfo.length = end_frame - start_frame + 1;



	if(show_track == 1)
		cv::namedWindow("DenseTrack", 0);

	cv::Mat image, prev_grey, grey;

	std::vector<float> fscales(0);
	std::vector<cv::Size> sizes(0);

	std::vector<cv::Mat> prev_grey_pyr(0), grey_pyr(0), flow_pyr(0);
	std::vector<cv::Mat> prev_poly_pyr(0), poly_pyr(0); // for optical flow

	std::vector<std::list<Track> > xyScaleTracks;
	int init_counter = 0; // indicate when to detect new feature points
	
	//检测视频保存
	cv::VideoWriter output_video;
	int frame_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);   //宽和高保持不变
    int frame_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);

	#ifdef CV_VERSION_EPOCH // OpenCV 2.x
                video_fps = capture.get(CV_CAP_PROP_FPS);
		#else
                video_fps = capture.get(cv::CAP_PROP_FPS);
		#endif

	 if (save_output_videofile)
		#ifdef CV_VERSION_EPOCH // OpenCV 2.x
                    output_video.open(out_videofile, CV_FOURCC('D', 'I', 'V', 'X'), std::max(30, video_fps), Size(frame_width,frame_height), true);
		#else
                    output_video.open(out_videofile, cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), std::max(30, video_fps), cv::Size(frame_width,frame_height), true);
		#endif
	
    //printf("(1,1) \t \n");
	while(true) {
		cv::Mat frame;
		int i, j, c;

		// get a new frame
		capture >> frame;

        //yolo检测
		std::vector<bbox_t> result_vec = detector.detect(frame);
		//仅保留检测结果为人的框
		for (int p = 0; p < result_vec.size(); p++)
        {
                if(result_vec[p].obj_id!=0)//人的id是0
                {
                    result_vec.erase(result_vec.begin() + p);
                    p--;
                }

        }

    
        //std::cout<<"frame:"<< frame_num<<std::endl;
         //std::cout<<"检测框的坐标为: "<< std::endl;
         //for (bbox_t box: result_vec)
        //{
        //    std::cout<<"X : "<< box.x<<", Y : "<< box.y <<", width : "<< box.w<<", Height : "<<box.h<<", prob : "<<box.prob<<", box_id: "<<box.obj_id<<std::endl;

        //}	

		if(frame.empty())
			break;

		if(frame_num < start_frame || frame_num > end_frame) {
			frame_num++;
			continue;
		}

		 /*-----------------------对第一帧做处理-------------------------*/
        //由于光流需要两帧进行计算，故第一帧不计算光流
		if(frame_num == start_frame) 
		{
			//光流计算初始化
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
					resize(prev_grey_pyr[iScale-1], prev_grey_pyr[iScale], prev_grey_pyr[iScale].size(), 0, 0, cv::INTER_LINEAR);

				// dense sampling feature points
				std::vector<cv::Point2f> points(0);
				DenseSample(prev_grey_pyr[iScale], points, quality, min_distance);
				
				

				// save the feature points
				std::list<Track>& tracks = xyScaleTracks[iScale];
				for(i = 0; i < points.size(); i++)
					tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
					
			}

			// compute polynomial expansion计算多项式展开
			my::FarnebackPolyExpPyr(prev_grey, prev_poly_pyr, fscales, 7, 1.5);

            // add new tracks in active tracks
			//初始化跟踪轨迹
            for ( auto box : result_vec)
            {
                std::vector<bbox_t> new_box;

                new_box.push_back(box);

                Track_box t = {new_box, box.prob, frame_num, id,revive};
                
                active_tracks.push_back(t);
                id++;
                
               cv::Rect2d bbox;
               bbox.x=box.x;
               bbox.y=box.y;
               bbox.width=box.w;
               bbox.height=box.h;
               track_boxes.push_back(bbox);

            }
            //std::cout<<"初始化轨迹数量 "<<active_tracks.size()<<std::endl;

               // initialize multitracker 初始化
            for (int i = 0; i <track_boxes.size(); i++)
            {
                multiTracker->add(cv::TrackerKCF::create(), frame, track_boxes[i]);
            }
       
			frame_num++;
			continue;
		}

		init_counter++;
		frame.copyTo(image);
		cvtColor(image, grey, CV_BGR2GRAY);

        if(frame_num>start_frame)
		{
        	multiTracker->update(frame);
			//std::cout<<"轨迹数量 "<<active_tracks.size()<<std::endl;
			//跟踪部分
			// for each track in active tracks
			for (int i = 0; i < active_tracks.size(); i++)
			{
				Track_box track = active_tracks[i];
				updated = false;
				// the index of box with highest iou
				IOU_index = Highest_iou(track.boxes.back(), result_vec);
				//std::cout<<"匹配到的bbox编号为"<<IOU_index<<"最大匹配iou值为"<< find_IOU(track.boxes.back(), result_vec[IOU_index])<<std::endl;
				//std::cout<<"最大匹配iou值为"<< find_IOU(track.boxes.back(), frame_boxes[index])<<std::endl;
				// if box is found and its iou greater than sigma_iou 
				if (IOU_index != -1 && find_IOU(track.boxes.back(), result_vec[IOU_index]) >= sigma_iou)
				{
					track.boxes.push_back(result_vec[IOU_index]);
					track.revive_id=revive;

					if (track.max_prob < result_vec[IOU_index].prob)
					{
						// update the prob in tracks
						track.max_prob = result_vec[IOU_index].prob;
					}
				result_vec.erase(result_vec.begin() + IOU_index);

					// updating the track
					active_tracks[i] = track;
					updated = true;

				}
					
			
				int box_index = frame_num - active_tracks[i].start_frame;
				//BoundingBox b = frameBoxes[j];
				if (box_index < active_tracks[i].boxes.size() )
					DrawTrack(image, active_tracks[i].boxes[box_index], active_tracks[i].track_id);
					
				
					//std::cout<<active_tracks[i].track_id<<"号轨迹中跟踪框数量 "<<active_tracks[i].boxes.size()<<std::endl;
				

				// if not updated, use kcf tracks or append them into finished tracks
				if (!updated)
				{
					if(track.revive_id<ttl&&track.max_prob >= sigma_h)
					{
						bbox_t tempbox=track.boxes.back();
					IOU_index = track.track_id;
						
						if (IOU_index != -1 )
						{
							tempbox.x= multiTracker->getObjects()[IOU_index].x;
							tempbox.y= multiTracker->getObjects()[IOU_index].y;
							tempbox.w= multiTracker->getObjects()[IOU_index].width;
							tempbox.h= multiTracker->getObjects()[IOU_index].height;
						}
						track.revive_id++;
						track.boxes.push_back(tempbox);
						active_tracks[i] = track;         
						DrawTrack(image, active_tracks[i].boxes.back(), active_tracks[i].track_id);            
					}  
				
					if(track.revive_id>=ttl)
					{
						if (track.max_prob >= sigma_h && track.boxes.size() >= t_min)           
									finished_tracks.push_back(track);   
					
								active_tracks.erase(active_tracks.begin() + i);
								i--;
					}             
				
		
				}
			}

			/// Create new tracks
			for (auto box : result_vec)
			{
				std::vector<bbox_t> b;
				b.push_back(box);
				// Track_id is set to 0 because we dont know if this track will
				// "survive" or not
				Track_box t = { b, box.prob, frame_num,  id , revive};
				active_tracks.push_back(t);
				//将新人体框加入kcf跟踪
					//将新人体框加入kcf跟踪
					if(box.prob>0.8){
				cv::Rect2d temp_bbox;
				temp_bbox.x=box.x;
				temp_bbox.y=box.y;
				temp_bbox.width=box.w;
				temp_bbox.height=box.h;
					multiTracker->add(cv::TrackerKCF::create(), frame, temp_bbox);           
					}
				id++;
			} 
		}




         //光流计算部分从这里开始
		// compute optical flow for all scales once
		my::FarnebackPolyExpPyr(grey, poly_pyr, fscales, 7, 1.5);
		my::calcOpticalFlowFarneback(prev_poly_pyr, poly_pyr, flow_pyr, 10, 2);

        //在每个尺度分别计算特征
		for(int iScale = 0; iScale < scale_num; iScale++) {
			//尺度0不缩放，其余尺度使用插值方法缩放
			if(iScale == 0)
				grey.copyTo(grey_pyr[0]);
			else
				resize(grey_pyr[iScale-1], grey_pyr[iScale], grey_pyr[iScale].size(), 0, 0, cv::INTER_LINEAR);

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
				cv::Point2f prev_point = iTrack->point[index];
				
				int x = std::min<int>(std::max<int>(cvRound(prev_point.x), 0), width-1);
				int y = std::min<int>(std::max<int>(cvRound(prev_point.y), 0), height-1);

				cv::Point2f point;
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
					//draw_boxes(image, result_vec, obj_names);//只对人进行框出

				// if the trajectory achieves the maximal length
				 // 若轨迹的长度达到了预设长度,在iDT中应该是设置为15
                // 达到长度后就可以输出各个特征了
				if(iTrack->index >= trackInfo.length) {
					std::vector<cv::Point2f> trajectory(trackInfo.length+1);
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

						

						// for spatio-temporal pyramid
						//printf("%f\t", std::min<float>(std::max<float>(mean_x/float(seqInfo.width), 0), 0.999));
						//printf("%f\t", std::min<float>(std::max<float>(mean_y/float(seqInfo.height), 0), 0.999));
						//printf("%f\n", std::min<float>(std::max<float>((frame_num - trackInfo.length/2.0 - start_frame)/float(seqInfo.length), 0), 0.999));
					
					
                        
			           //找到相距轨迹起点和终点最近的框
					   bbox_t temp_box=active_tracks[0].boxes.back();
						double temp_delta_x=trajectory[0].x-temp_box.x-0.5*temp_box.w;
						double temp_delta_y=trajectory[0].y-temp_box.y-0.5*temp_box.h;
						double min_start_distance=sqrt(temp_delta_x*temp_delta_x + temp_delta_y*temp_delta_y);
                        unsigned int box_x=temp_box.x+0.5*temp_box.w;
						unsigned int box_y=temp_box.y+0.5*temp_box.h;
						unsigned int box_w=temp_box.w;
						unsigned int box_h=temp_box.h;
						//std::string person1_string ("person");
						

						for (auto &i : active_tracks) 
						{
							//std::string obj1_name = obj_names[i.obj_id];  //person1_string.compare (obj1_name)==0
							if(1)
							{
								double delta_x=trajectory[0].x-i.boxes.back().x-0.5*i.boxes.back().w;
								double delta_y=trajectory[0].y-i.boxes.back().y-0.5*i.boxes.back().h;
								double start_distance=sqrt(delta_x*delta_x + delta_y*delta_y);
								if(min_start_distance>start_distance)
								{
										box_x=i.boxes.back().x+0.5*i.boxes.back().w;
										box_y=i.boxes.back().y+0.5*i.boxes.back().h;
										box_w=i.boxes.back().w;
										box_h=i.boxes.back().h;
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
						//away_flag为1代表轨迹远离人体框
						int away_flag=(min_start_distance<min_end_distance)?1:0;
                        
						/*判断抛投轨迹并画出
						double input[3]={normalized_min_start_distance,0,normalized_length};
						double prob=score(input);
						if(prob>0.5)
                              DrawThrowTrack(iTrack->point, iTrack->index, fscales[iScale], image);
                         
						*/
                        
		            	printf("frame:%d num:%d\t1\t1:%.3f\t2:%.3f\t3:%.3f\t4:%.3f\t5:%.3f\t6:%.3f \t 7:%d \t",frame_num, k,normalized_min_start_distance ,normalized_min_end_distance,normalized_length,mean_cos_distance,var_cos_distance,max_cos_distance,away_flag);
						//printf(" 0 1:%.3f 2:%.3f 3:%.3f 4:%.3f 5:%.3f 6:%.3f \t frame:%d num:%d",normalized_min_start_distance ,normalized_min_end_distance,normalized_length,mean_cos_distance,var_cos_distance,max_cos_distance,frame_num, k);
						/*PrintDesc(iTrack->hog, hogInfo, trackInfo);
						printf("hog end\t");
						PrintDesc(iTrack->mbhX, mbhInfo, trackInfo);
						printf("mbhx end\t");
						PrintDesc(iTrack->mbhY, mbhInfo, trackInfo);
						printf("mbhy end\t");*/
						printf("\n");
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
			std::vector<cv::Point2f> points(0);
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
		cv::destroyWindow("DenseTrack");
   
	return 0;
}
