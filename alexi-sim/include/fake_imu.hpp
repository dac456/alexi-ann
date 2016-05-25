#ifndef __FAKE_IMU_HPP
#define __FAKE_IMU_HPP

#include <memory>
#include "terrain.hpp"

class fake_imu{
private:
    terrain_ptr _terrain;

    size_t _pixel_pos_x;
    size_t _pixel_pos_y;

public:
    fake_imu(terrain_ptr terrain);

    void update(std::pair<size_t,size_t> pixel_pos);

    double get_accel_pitch();

};

typedef std::shared_ptr<fake_imu> fake_imu_ptr;

#endif
