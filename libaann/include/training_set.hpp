#ifndef __TRAINING_SET_HPP
#define __TRAINING_SET_HPP

#include "common.hpp"

struct frame_data{
    bool warped;

    double rw, rl;

    double x, y;
    double dx, dy;
    double theta;
    double dtheta;
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
    ALL
};

class training_set{
private:
    fs::path _set_path;

    std::vector<frame_data> _frames;
    blaze::DynamicMatrix<double> _input_set;
    blaze::DynamicMatrix<double> _target_set;

    frame_data _last_frame;

public:
    //training_set(fs::path p, TRAINING_TYPE type);
    training_set(std::vector<frame_data> frames, TRAINING_TYPE type);

    void save_fann_data(fs::path file);

    std::vector<frame_data> get_frames();
    blaze::DynamicMatrix<double> get_input_set();
    blaze::DynamicMatrix<double> get_target_set();

};

#endif
