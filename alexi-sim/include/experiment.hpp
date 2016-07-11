#ifndef __EXPERIMENT_HPP
#define __EXPERIMENT_HPP

#include <SDL/SDL.h>

#include <map>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include "terrain.hpp"
#include "platform.hpp"
#include "fake_imu.hpp"

#include "fann_ffn.hpp"

class experiment{
private:
    SDL_Surface* _display;

    std::map<std::string, std::shared_ptr<fann_ffn>> _ann;

    terrain_ptr _terrain;
    platform_ptr _platform;
    fake_imu_ptr _imu;
    float* _last_terrain_update;

    double _scale;
    double _particle_radius;

public:
    experiment(SDL_Surface* disp, fs::path cfg);

    void step();

    terrain_ptr get_terrain();

private:
    std::pair<size_t,size_t> _real_pos_to_pixel_pos(std::pair<double,double> real_pos);

};

#endif
