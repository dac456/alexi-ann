#ifndef __TRAINING_SET_HPP
#define __TRAINING_SET_HPP

#include "common.hpp"

struct frame_data{
    bool warped;
    double dppx, dppy;

    double rw, rl;

    double x, y;
    double px, py;
    double dx, dy;
    double dx_last, dy_last;
    double theta;
    double dtheta;
    double dtheta_last;
    double pitch;

    double v;
    double w;
    double left;
    double right;
    double left_last;
    double right_last;

    int pw, ph;
};

enum TRAINING_TYPE{
    DX,
    DY,
    DTHETA,
    TERRAIN,
    ALL
};

class training_set{
private:
    fs::path _set_path;
    TRAINING_TYPE _type;

    std::vector<frame_data> _frames;
    std::vector<std::array<double,256>> _images;
    std::vector<std::array<double,256>> _diff_images;
    blaze::DynamicMatrix<double> _input_set;
    blaze::DynamicMatrix<double> _target_set;

    frame_data _last_frame;

public:
    //training_set(fs::path p, TRAINING_TYPE type);
    training_set(std::vector<frame_data> frames, std::vector<std::array<double,256>> images, std::vector<std::array<double,256>> diff_images, TRAINING_TYPE type);

    void save_fann_data(fs::path file);
    void save_blaze_data(fs::path file);

    std::vector<frame_data> get_frames();
    blaze::DynamicMatrix<double> get_input_set();
    blaze::DynamicMatrix<double> get_target_set();

};

#endif
