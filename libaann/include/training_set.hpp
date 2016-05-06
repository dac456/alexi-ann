#ifndef __TRAINING_SET_HPP
#define __TRAINING_SET_HPP

#include "common.hpp"

struct frame_data{
    double rw, rl;

    double x, y;
    double theta;
    double pitch;

    double v;
    double w;

    int pw, ph;
};

class training_set{
private:
    fs::path _set_path;

    blaze::DynamicMatrix<double> _input_set;
    blaze::DynamicMatrix<double> _target_set;

    frame_data _last_frame;

public:
    training_set(fs::path p);

    void save_fann_data(fs::path file);

    blaze::DynamicMatrix<double> get_input_set();
    blaze::DynamicMatrix<double> get_target_set();

private:
    frame_data _parse_frame(fs::path file);

};

#endif