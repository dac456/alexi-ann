#ifndef __PLATFORM_HPP
#define __PLATFORM_HPP

#include <memory>
#include <SDL/SDL.h>

#include "rnn.hpp"
#include "fann_ffn.hpp"
#include "fake_imu.hpp"

#include <map>

class platform{
private:
    size_t _ticks;
    double _rand;

    SDL_Surface* _display;
    std::map<std::string, std::shared_ptr<fann_ffn>> _ann;

    fake_imu_ptr _imu;

    double _pos_x;
    double _pos_y;
    double _yaw;
    size_t _pixel_pos_x;
    size_t _pixel_pos_y;

    double _desired_linear_velocity;
    double _desired_angular_velocity;
    double _left;
    double _right;
    double _last_dx;
    double _last_dy;
    double _last_dtheta;
    double _last_speed;
    float* _last_input;
    blaze::DynamicMatrix<double> _input_dx;
    blaze::DynamicMatrix<double> _input_dy;
    blaze::DynamicMatrix<double> _input_dtheta;
    size_t _num_inputs;

    static std::ofstream _log;

public:
    platform(SDL_Surface* disp, std::map<std::string, std::shared_ptr<fann_ffn>> ann, fake_imu_ptr imu, double init_x, double init_y);

    void step(double width, double height); //OK to pass this info here?

    double get_pos_x();
    double get_pos_y();
    double get_yaw();

    bool is_inclined();

    float* get_last_input();

private:
    void _move();
};

typedef std::shared_ptr<platform> platform_ptr;

#endif
