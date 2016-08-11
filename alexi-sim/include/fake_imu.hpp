#ifndef __FAKE_IMU_HPP
#define __FAKE_IMU_HPP

#include <memory>
#include <queue>
#include "terrain.hpp"

class fake_imu{
private:
    terrain_ptr _terrain;

    size_t _pixel_pos_x;
    size_t _pixel_pos_y;
    double _yaw;

    std::vector<double> _pitch_samples;
    std::vector<double> _roll_samples;

public:
    fake_imu(terrain_ptr terrain);

    void update(std::pair<size_t,size_t> pixel_pos, double yaw);

    double get_accel_pitch();
    double get_accel_roll();

    std::pair<std::pair<size_t,size_t>,std::pair<size_t,size_t>> get_debug_forward();
    std::pair<std::pair<size_t,size_t>,std::pair<size_t,size_t>> get_debug_right();

};

typedef std::shared_ptr<fake_imu> fake_imu_ptr;

#endif
