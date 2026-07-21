#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <set>
#include <functional>

class NotesRender {
public:
    NotesRender(std::function<std::pair<int,int>(int)> get_side_x_func);

    void update_in_temp(cv::Mat& in_tmp_img, cv::Mat& in_tmp_img_mask, int step_pix,
                        const std::map<int, std::map<std::string, std::vector<int>>>& notes,
                        const std::vector<cv::Scalar>& color_map);

    int radius;
    int border_thickness;
    int gap;
    int shrink;
    int keep_for_out_pix;
    int min_update_pix;
    int off_region_pix;

    std::function<std::pair<int,int>(int)> get_side_x_func;
    std::vector<int> last_update_pix;
    std::vector<int> last_on_border_pix;
};