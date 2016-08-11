#include "fake_imu.hpp"
#include <math.h>

fake_imu::fake_imu(terrain_ptr terrain)
    : _terrain(terrain)
    , _yaw(0.0)
{

}

void fake_imu::update(std::pair<size_t,size_t> pixel_pos, double yaw){
    _pixel_pos_x = pixel_pos.first;
    _pixel_pos_y = pixel_pos.second;
    _yaw = yaw;
}

double fake_imu::get_accel_pitch(){
    size_t forward_x = 8;
    size_t forward_y = 0;
    size_t forward_rx = forward_x * cos(_yaw) - forward_y * sin(_yaw);
    size_t forward_ry = forward_x * sin(_yaw) + forward_y * cos(_yaw);

    unsigned char h1 = 0;
    unsigned char h2 = 0;
    for(int i = -2; i < 2; i++){
        for(int j = -2; j < 2; j++){
            h1 += _terrain->get_height_at(_pixel_pos_x - forward_rx + i, _pixel_pos_y - forward_ry + j);
            h2 += _terrain->get_height_at(_pixel_pos_x + forward_rx + i, _pixel_pos_y + forward_ry + j);
        }
    }
    h1 /= 16;
    h2 /= 16;
    //unsigned char h1 = _terrain->get_height_at(_pixel_pos_x - forward_rx, _pixel_pos_y - forward_ry);
    //unsigned char h2 = _terrain->get_height_at(_pixel_pos_x + forward_rx, _pixel_pos_y + forward_ry);

    double h_delta = static_cast<double>(h2 - h1);
    double x_delta = 16.0;

    _pitch_samples.push_back(atan(h_delta / x_delta));
    if(_pitch_samples.size() > 20){
        _pitch_samples.erase(_pitch_samples.begin());
    }

    double sum = 0.0;
    for(double pitch : _pitch_samples){
        sum += pitch;
    }
    sum /= _pitch_samples.size();

    return sum;
}

double fake_imu::get_accel_roll(){
    size_t right_x = 0;
    size_t right_y = 8;
    size_t right_rx = right_x * cos(_yaw) - right_y * sin(_yaw);
    size_t right_ry = right_x * sin(_yaw) + right_y * cos(_yaw);

    unsigned char h1 = 0;
    unsigned char h2 = 0;
    for(int i = -2; i < 2; i++){
        for(int j = -2; j < 2; j++){
            h1 += _terrain->get_height_at(_pixel_pos_x - right_rx + i, _pixel_pos_y - right_ry + j);
            h2 += _terrain->get_height_at(_pixel_pos_x + right_rx + i, _pixel_pos_y + right_ry + j);
        }
    }
    h1 /= 16;
    h2 /= 16;
    //unsigned char h1 = _terrain->get_height_at(_pixel_pos_x - right_rx, _pixel_pos_y - right_ry);
    //unsigned char h2 = _terrain->get_height_at(_pixel_pos_x + right_rx, _pixel_pos_y + right_ry);

    double h_delta = static_cast<double>(h2 - h1);
    double y_delta = 16.0;

    _roll_samples.push_back(atan(h_delta / y_delta));
    if(_roll_samples.size() > 20){
        _roll_samples.erase(_roll_samples.begin());
    }

    double sum = 0.0;
    for(double roll : _roll_samples){
        sum += roll;
    }
    sum /= _roll_samples.size();

    return sum;
}

std::pair<std::pair<size_t,size_t>,std::pair<size_t,size_t>> fake_imu::get_debug_forward(){
    size_t forward_x = 8;
    size_t forward_y = 0;
    size_t forward_rx = forward_x * cos(_yaw) - forward_y * sin(_yaw);
    size_t forward_ry = forward_x * sin(_yaw) + forward_y * cos(_yaw);

    return std::make_pair(std::make_pair(_pixel_pos_x - forward_rx, _pixel_pos_y - forward_ry), std::make_pair(_pixel_pos_x + forward_rx, _pixel_pos_y + forward_ry));
}

std::pair<std::pair<size_t,size_t>,std::pair<size_t,size_t>> fake_imu::get_debug_right(){
    size_t right_x = 0;
    size_t right_y = 8;
    size_t right_rx = right_x * cos(_yaw) - right_y * sin(_yaw);
    size_t right_ry = right_x * sin(_yaw) + right_y * cos(_yaw);

    return std::make_pair(std::make_pair(_pixel_pos_x - right_rx, _pixel_pos_y - right_ry), std::make_pair(_pixel_pos_x + right_rx, _pixel_pos_y + right_ry));
}
