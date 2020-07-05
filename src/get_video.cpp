#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include "mynteye/api/api.h"

using namespace mynteye;
using namespace cv;

int main(int argc, char *argv[]) {
    auto &&api = API::Create(argc, argv);
    if (!api) return 1;

    bool ok;
    auto &&request = api->SelectStreamRequest(&ok);
    if (!ok) return 1;
    api->ConfigStreamRequest(request);
    api->EnableStreamData(Stream::LEFT_RECTIFIED);
    api->EnableStreamData(Stream::RIGHT_RECTIFIED);
    api->Start(Source::VIDEO_STREAMING);

    double fps;
    double t = 0.01;
    std::cout << "fps:" << std::endl;

    Mat H_matrix =(Mat_<double>(3,3)<<1.00476859e+00,-3.45690534e-02, -2.62218249e+01,
                                      3.57667123e-02, 9.92150127e-01, -4.21958744e+00,
                                      1.06401679e-04, -8.28318658e-05, 1.0);

    cv::namedWindow("frame");
    cv::namedWindow("transformation");
    std::int32_t count = 0;
    // TODO Change File Name each time
    std::string file_name1 = "ref.avi";
    std::string file_name2 = "left_trans.avi";

    VideoWriter video_right(file_name1, CV_FOURCC('M','J','P','G'),30, Size(640,400)); 
    VideoWriter video_left_trans(file_name2,CV_FOURCC('M','J','P','G'),30, Size(640,400)); 
    VideoWriter video_left("left.avi", CV_FOURCC('M','J','P','G'),30, Size(640,400)); 

    std::cout << "Press 's' to start record video." << std::endl;
    bool flag_start = false;
    while (true) {
        api->WaitForStreams();

        auto &&left_data = api->GetStreamData(Stream::LEFT_RECTIFIED);
        auto &&right_data = api->GetStreamData(Stream::RIGHT_RECTIFIED);
        cv::Mat left_trans = cv::Mat(left_data.frame.rows, left_data.frame.cols, CV_8UC3, Scalar(0, 0, 0));
        if (!left_data.frame.empty() && !right_data.frame.empty()) {
            double t_c = cv::getTickCount() / cv::getTickFrequency();
            fps = 1.0/(t_c - t);
            printf("\b\b\b\b\b\b\b\b\b%.2f", fps);
            t = t_c;

            cv::Mat img;
            cv::Mat img2;
            cv::warpPerspective(left_data.frame, left_trans, H_matrix, left_data.frame.size());
            cv::hconcat(left_data.frame, right_data.frame,img);
            cv::hconcat(right_data.frame, left_trans, img2);
            cv::imshow("frame", img);
            cv::imshow("transformation", img2);
            if (flag_start){
                if (count++ == 0)
                    std::cout << "Start To Record Video!!!" << std::endl;
                video_right.write(right_data.frame);
                video_left.write(left_data.frame);
                video_left_trans.write(left_trans);
            }
        }

        char key = static_cast<char>(cv::waitKey(1));
        if (key == 27 || key == 'q' || key == 'Q') {  // ESC/Q
            break;
        } 
        else if (key == 32 || key == 's' || key == 'S')
            flag_start = true;
        else if ((key == 32 || key == 's' || key == 'S') && (flag_start == true))
        {
            flag_start = false;
            break;
        }
            /*
            if (!left_data.frame.empty()
                && !right_data.frame.empty()) {
                char l_name[20];
                char r_name[20];
                char trans_l_name[20];
                ++count;
                snprintf(l_name, sizeof(l_name), "left_%d.jpg", count);
                snprintf(r_name, sizeof(r_name), "right_%d.jpg", count);
                snprintf(trans_l_name, sizeof(trans_l_name), "trans_left_%d.jpg", count);
                cv::imwrite(l_name, left_data.frame);
                cv::imwrite(r_name, right_data.frame);
                cv::imwrite(trans_l_name, left_trans);
                
                std::cout << "Saved " << l_name << " " << r_name << " " << trans_l_name << " to current directory" << std::endl;
            */
    }
    
    api->Stop(Source::VIDEO_STREAMING);
    video_right.release();
    video_left_trans.release(); 
    video_left.release();
    destroyAllWindows();
    return 0;

}
