#ifndef __TRAINING_SET_HPP
#define __TRAINING_SET_HPP

#include "common.hpp"

struct frame_data{
    int x, y;
    double theta;

    double v;
    double w;

    int pw, ph;
};

class training_set{
private:
    fs::path _set_path;

    blaze::DynamicMatrix<double> _input_set;
    blaze::DynamicMatrix<double> _target_set;

public:
    training_set(fs::path p);

    blaze::DynamicMatrix<double> get_input_set();
    blaze::DynamicMatrix<double> get_target_set();

private:
    frame_data _parse_frame(fs::path file);

};

#endif
