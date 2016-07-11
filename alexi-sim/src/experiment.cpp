#include "experiment.hpp"

#include <iostream>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <SDL/SDL_gfxPrimitives.h>

experiment::experiment(SDL_Surface* disp, fs::path cfg)
    : _display(disp)
    , _last_terrain_update(nullptr)
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
        _ann["dx"] = std::make_shared<fann_ffn>("fann_dx.net");
        _ann["dy"] = std::make_shared<fann_ffn>("fann_dy.net");
        _ann["dtheta"] = std::make_shared<fann_ffn>("fann_dtheta.net");
        _ann["terrain"] = std::make_shared<fann_ffn>("fann_terrain.net");
        _imu = std::make_shared<fake_imu>(_terrain);
        _platform = std::make_shared<platform>(_display, _ann, _imu, opts["vehicle.x"].as<double>(), opts["vehicle.z"].as<double>());
    }
}

void experiment::step(){
    if(_imu){
        _imu->update(_real_pos_to_pixel_pos(std::make_pair(_platform->get_pos_x(), _platform->get_pos_y())), _platform->get_yaw());
    }

    if(_platform){
        _platform->step(_terrain->get_width() * _particle_radius * 2.0, _terrain->get_height() * _particle_radius * 2.0);
    }

    std::pair<double,double> real_pos = std::make_pair(_platform->get_pos_x(), _platform->get_pos_y());
    std::pair<size_t,size_t> plot_pos = _real_pos_to_pixel_pos(real_pos);

    if(_terrain){
        float* in = _platform->get_last_input();
        if(in){
            float inTerrain[262];
            for(size_t i = 0; i < 6; i++) inTerrain[i] = in[i];
            
            if(_last_terrain_update == nullptr){
                for(size_t i = 6; i < 262; i++) inTerrain[i] = 0.0f;
            }
            else{
                for(size_t i = 6; i < 262; i++) inTerrain[i] = _last_terrain_update[i-6];
            }

            float* out_terrain = _ann["terrain"]->predict(in);
            if(out_terrain){
                _last_terrain_update = out_terrain;

                int px = plot_pos.first;
                int py = plot_pos.second;
                int start_x = (px - 8);
                int start_y = (py - 8);

                #pragma omp parallel for
                for(size_t wy = 0; wy < 4; wy++){
                    #pragma omp parallel for
                    for(size_t wx = 0; wx < 4; wx++){

                        float avg = 0.0f;
                        float c = 0.0f;
                        for(size_t y = wy*4; y < (wy*4) + 4; y++){
                            for(size_t x = wx*4; x < (wx*4) + 4; x++){
                                avg += out_terrain[x + (y * 16)];
                                c += 1.0f;
                            }
                        }
                        avg /= c;

                        for(size_t y = wy*4; y < (wy*4) + 4; y++){
                            for(size_t x = wx*4; x < (wx*4) + 4; x++){
                                out_terrain[x + (y * 16)] = avg;
                            }
                        }
                    }
                }

                size_t idx = 0;
                for(size_t i = 0; i < 16; i++){
                    int y = (start_y + i) /*% _terrain->get_display_height()*/;
                    if(y < 0){
                        y = _terrain->get_display_height() + y;
                    }
                    else if(y >= _terrain->get_display_height()){
                        y = y - _terrain->get_display_height();
                    }
                    for(size_t j = 0; j < 16; j++){
                        int x = (start_x + j) /*% _terrain->get_display_width()*/;
                        if(x < 0){
                            x = _terrain->get_display_width() + x;
                        }
                        else if(x >= _terrain->get_display_width()){
                            x = x - _terrain->get_display_width();
                        }
                        _terrain->update_pixel_by_delta(x, y, static_cast<int>(255.0f*out_terrain[idx]));
                        idx++;
                    }
                }
            }
        }
        _terrain->update();
    }

    circleRGBA(_display, plot_pos.first, plot_pos.second, 16, 255, 0, 0, 255);

    const double yaw = _platform->get_yaw();

    size_t forward_x = 16;
    size_t forward_y = 0;
    size_t forward_rx = forward_x * cos(yaw) - forward_y * sin(yaw);
    size_t forward_ry = forward_x * sin(yaw) + forward_y * cos(yaw);
    circleRGBA(_display, plot_pos.first + forward_rx, plot_pos.second + forward_ry, 4, 0, 255, 0, 255);

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
