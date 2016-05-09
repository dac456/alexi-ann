#include "experiment.hpp"

#include <iostream>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <SDL/SDL_gfxPrimitives.h>

experiment::experiment(SDL_Surface* disp, fs::path cfg)
    : _display(disp)
{
    fs::path full_cfg_path = fs::canonical(cfg);

    if(fs::exists(full_cfg_path)){
        po::options_description desc;
        desc.add_options()
            ("output_directory_prefix", po::value<std::string>()->required(), "Directory prefix for various simulation output")
            ("chrono_data_path", po::value<std::string>()->default_value("../data"), "Path for chrono data files")

            ("vehicle.x", po::value<double>())
            ("vehicle.y", po::value<double>())
            ("vehicle.z", po::value<double>())
            ("vehicle.r", po::value<double>())
            ("vehicle.p", po::value<double>())
            ("vehicle.h", po::value<double>())

            ("map.filename", po::value<std::string>())
            ("map.scale", po::value<double>())
            ("map.particle_radius", po::value<double>()->default_value(0.15))

            ("raygrid.resolution", po::value<double>())

            ("experiment.name", po::value<std::string>())
            ("experiment.algorithm", po::value<std::string>())
            ("experiment.linear", po::value<double>())
            ("experiment.angular", po::value<double>())
        ;

        po::variables_map opts;
        std::ifstream cfg(full_cfg_path.string());

        po::store(po::parse_config_file(cfg , desc), opts);
        po::notify(opts);

        _scale = opts["raygrid.resolution"].as<double>();
        _particle_radius = opts["map.particle_radius"].as<double>();

        fs::path map_file = full_cfg_path;
        map_file.remove_filename();
        map_file /= fs::path(opts["chrono_data_path"].as<std::string>()) / fs::path(opts["map.filename"].as<std::string>());

        _terrain = std::make_shared<terrain>(_display, map_file, _scale);

        //std::pair<size_t,size_t> vehicle_position = _real_pos_to_pixel_pos(std::make_pair(opts["vehicle.x"].as<double>(), opts["vehicle.z"].as<double>()));
        _ann = std::make_shared<fann_ffn>("fann.net");
        _platform = std::make_shared<platform>(_display, _ann, opts["vehicle.x"].as<double>(), opts["vehicle.z"].as<double>());
    }
}

void experiment::step(){
    if(_platform){
        _platform->step();
    }

    if(_terrain){
        _terrain->update();
    }

    std::pair<double,double> real_pos = std::make_pair(_platform->get_pos_x(), _platform->get_pos_y());
    std::pair<size_t,size_t> plot_pos = _real_pos_to_pixel_pos(real_pos);
    circleRGBA(_display, plot_pos.first, plot_pos.second, 16, 255, 0, 0, 255);

    SDL_Flip(_display);
}

terrain_ptr experiment::get_terrain(){
    return _terrain;
}

std::pair<size_t,size_t> experiment::_real_pos_to_pixel_pos(std::pair<double,double> real_pos){
    double real_width = _terrain->get_width() * _particle_radius * 2.0;
    double real_height = _terrain->get_height() * _particle_radius * 2.0;
    double half_width = real_width * 0.5;
    double half_length = real_height * 0.5;

    size_t px = ((real_pos.first + half_width) / real_width) * (_terrain->get_display_width());
    size_t py = ((real_pos.second + half_length) / real_height) * (_terrain->get_display_height());

    return std::make_pair(px, py);
}
