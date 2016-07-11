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

    unsigned char h1 = _terrain->get_height_at(_pixel_pos_x - forward_rx, _pixel_pos_y - forward_ry);
    unsigned char h2 = _terrain->get_height_at(_pixel_pos_x + forward_rx, _pixel_pos_y + forward_ry);

    double h_delta = static_cast<double>(h2 - h1);
    double x_delta = 4.0;

    return atan(h_delta / x_delta);
}
