#include "fake_imu.hpp"
#include <math.h>

fake_imu::fake_imu(terrain_ptr terrain)
    : _terrain(terrain)
{

}

void fake_imu::update(std::pair<size_t,size_t> pixel_pos){
    _pixel_pos_x = pixel_pos.first;
    _pixel_pos_y = pixel_pos.second;
}

double fake_imu::get_accel_pitch(){
    unsigned char h1 = _terrain->get_height_at(_pixel_pos_x - 2, _pixel_pos_y);
    unsigned char h2 = _terrain->get_height_at(_pixel_pos_x + 2, _pixel_pos_y);

    double h_delta = static_cast<double>(h2 - h1);
    double x_delta = 4.0;

    return atan(h_delta / x_delta);
}
