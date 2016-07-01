#include "platform.hpp"
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

std::default_random_engine generator;
std::uniform_real_distribution<double> distribution(0, 1);

platform::platform(SDL_Surface* disp, std::map<std::string, std::shared_ptr<fann_ffn>> ann, fake_imu_ptr imu, double init_x, double init_y)
    : _display(disp)
    , _ann(ann)
    , _imu(imu)
    , _pos_x(init_x)
    , _pos_y(init_y)
    , _yaw(0.0)
    , _left(0.0)
    , _last_dx(0.0)
    , _last_dy(0.0)
    , _right(0.0)
{
}

void platform::step(double width, double height){
    /*if(is_inclined()){
        _desired_linear_velocity = (3.0f);
        _desired_angular_velocity = (0.0f);
    }
    else{
        _desired_linear_velocity = (2.0f);


        double r = distribution(generator);

        if(r < 0.25) _desired_angular_velocity = (1.57);
        else if(r >= 0.25 && r < 0.5) _desired_angular_velocity = (-1.57);
        else if(r > 0.75) _desired_angular_velocity = (0.0f);
    }*/
    _desired_linear_velocity = 2.0f;
    _desired_angular_velocity = 0.0f;

    std::cout << "pitch: " << _imu->get_accel_pitch() << std::endl;
    std::cout << _left << " " << _right << std::endl;
    float in[5] = {_left, _right, _imu->get_accel_pitch(), _last_dx, _last_dy};
    float* out_dx = _ann["dx"]->predict(in);
    float* out_dy = _ann["dy"]->predict(in);
    float* out_dtheta = _ann["dtheta"]->predict(in);
    _last_dx = out_dx[0];
    _last_dy = out_dy[0];

    float speed = sqrt( pow(out_dx[0], 2.0) + pow(out_dy[0], 2.0) );
    if(_desired_linear_velocity < 0.0f){
        speed *= -1.0f;
    }

    _yaw += out_dtheta[0];
    _pos_x += speed*cos(_yaw);
    _pos_y += speed*sin(_yaw);

    if(_pos_x > width * 0.5){
        _pos_x -= width;
    }
    if(_pos_x < -width * 0.5){
        _pos_x += width;
    }
    if(_pos_y > height * 0.5){
        _pos_y -= height;
    }
    if(_pos_y < -height * 0.5){
        _pos_y += height;
    }

    _move();
    _last_input = in;
}

void platform::_move(){
    const double r = 0.5; //wheel radius (m)
    const double L = 3.2679; //wheel base (m)

    _left = (_desired_linear_velocity - (L*_desired_angular_velocity*0.5)) / r;
    _right = (_desired_linear_velocity + (L*_desired_angular_velocity*0.5)) / r;
}

double platform::get_pos_x(){
    return _pos_x;
}

double platform::get_pos_y(){
    return _pos_y;
}

double platform::get_yaw(){
    return _yaw;
}

bool platform::is_inclined(){
    if(fabs((_imu->get_accel_pitch()*180.0) / M_PI) > 20.0) return true;
    else return false;
}

float* platform::get_last_input(){
    return _last_input;
}
