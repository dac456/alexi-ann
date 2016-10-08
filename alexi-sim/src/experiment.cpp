#include "experiment.hpp"
#include "data_preprocessor.hpp"

#include <iostream>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <SDL/SDL_gfxPrimitives.h>

double sigmoid(double value){
    return 1.0 / (1.0 + exp(-value));
}

double sigmoid_dx(double value){
    return sigmoid(value) * (1.0 - sigmoid(value));
}

double linear(double value){
    return value;
}

double linear_dx(double value){
    return 1.0;
}

double sigmoid_tanh(double value){
    return tanh(value);
}

double tanh_dx(double value){
    return 1.0 - pow(tanh(value), 2.0);
}

double softsign(double value){
    return value / (fabs(value) + 1.0);
}

double softsign_dx(double value){
    return 1.0 / pow(fabs(value) + 1, 2.0);
}

double log_semisig(double value){
    if(value > 0){
        return log(value + 1.0);
    }
    else{
        return -log(fabs(value) + 1.0);
    }
}

double log_semisig_dx(double value){
    if(value > 0){
        return 1.0 / (value + 1.0);
    }
    else{
        return -1.0 / (fabs(value) + 1.0);
    }
}

double softplus(double value){
    return log(1 + exp(value));
}

double ramp(double value){
    return std::max(value, 0.0);
}

double ramp_dx(double value){
    if(value < 0) {
        return 0;
    } else {
        return 1;
    }
}

double sigmoid_sinh(double value){
    return (exp(value) - exp(-value)) * 0.5;
}

double sigmoid_cosh(double value){
    return (exp(value) + exp(-value)) * 0.5;
}

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

        fs::path frame_path = full_cfg_path;
        frame_path.replace_extension("");
        frame_path = fs::canonical(frame_path / "framedata");
        std::cout << frame_path.string() << std::endl;

        std::vector<fs::path> paths;
        paths.push_back(frame_path);
        data_preprocessor preproc(paths);
        std::vector<frame_data> frames = preproc.get_frames();
        for(auto frame : frames){
            _ref_path.push_back(std::make_pair(frame.px, frame.py));
        }

        po::store(po::parse_config_file(cfg , desc), opts);
        po::notify(opts);

        _scale = opts["raygrid.resolution"].as<double>();
        _particle_radius = opts["map.particle_radius"].as<double>();

        fs::path map_file = full_cfg_path;
        map_file.remove_filename();
        map_file /= fs::path(opts["chrono_data_path"].as<std::string>()) / fs::path(opts["map.filename"].as<std::string>());

        _terrain = std::make_shared<terrain>(_display, map_file, _scale);

        //std::pair<size_t,size_t> vehicle_position = _real_pos_to_pixel_pos(std::make_pair(opts["vehicle.x"].as<double>(), opts["vehicle.z"].as<double>()));
        /*_ann["dx"] = std::make_shared<fann_ffn>("rnn_dx");
        _ann["dx"]->set_hidden_activation_function(sigmoid_sinh);
        _ann["dx"]->set_hidden_activation_function_dx(sigmoid_cosh);
        _ann["dx"]->set_output_activation_function(linear);
        _ann["dx"]->set_output_activation_function_dx(linear_dx);

        _ann["dy"] = std::make_shared<fann_ffn>("rnn_dy");
        _ann["dy"]->set_hidden_activation_function(sigmoid_sinh);
        _ann["dy"]->set_hidden_activation_function_dx(sigmoid_cosh);
        _ann["dy"]->set_output_activation_function(linear);
        _ann["dy"]->set_output_activation_function_dx(linear_dx);

        _ann["dtheta"] = std::make_shared<fann_ffn>("rnn_dtheta");
        _ann["dtheta"]->set_hidden_activation_function(linear);
        _ann["dtheta"]->set_hidden_activation_function_dx(linear_dx);
        _ann["dtheta"]->set_output_activation_function(linear);
        _ann["dtheta"]->set_output_activation_function_dx(linear_dx);*/
        _ann["dx"] = std::make_shared<fann_ffn>("fann_dx.net");
        _ann["dy"] = std::make_shared<fann_ffn>("fann_dy.net");
        _ann["dtheta"] = std::make_shared<fann_ffn>("fann_dtheta.net");

        //_ann["terrain"] = std::make_shared<fann_ffn>("fann_terrain.net");
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
        /*float* in = _platform->get_last_input();
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
                    int y = (start_y + i);
                    if(y < 0){
                        y = _terrain->get_display_height() + y;
                    }
                    else if(y >= _terrain->get_display_height()){
                        y = y - _terrain->get_display_height();
                    }
                    for(size_t j = 0; j < 16; j++){
                        int x = (start_x + j);
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
        }*/
        _terrain->update();
    }


    _path.push_back(std::make_pair(plot_pos.first, plot_pos.second));

    //Draw paths
    for(size_t i = 0; i < _ref_path.size(); i += 4){
        circleRGBA(_display, _ref_path[i].second, _ref_path[i].first, 2, 0, 0, 255, 127);
    }

    for(size_t i = 0; i < _path.size(); i += 4){
        circleRGBA(_display, _path[i].first, _path[i].second, 2, 255, 255, 0, 127);
    }

    //Draw vehicle
    circleRGBA(_display, plot_pos.first, plot_pos.second, 16, 255, 0, 0, 255);

    const double yaw = _platform->get_yaw();

    size_t forward_x = 16;
    size_t forward_y = 0;
    size_t forward_rx = forward_x * cos(yaw) - forward_y * sin(yaw);
    size_t forward_ry = forward_x * sin(yaw) + forward_y * cos(yaw);
    circleRGBA(_display, plot_pos.first + forward_rx, plot_pos.second + forward_ry, 4, 0, 255, 0, 255);

    /*circleRGBA(_display, _imu->get_debug_forward().first.first, _imu->get_debug_forward().first.second, 4, 255, 255, 255, 255);
    circleRGBA(_display, _imu->get_debug_forward().second.first, _imu->get_debug_forward().second.second, 4, 255, 255, 255, 255);
    circleRGBA(_display, _imu->get_debug_right().first.first, _imu->get_debug_right().first.second, 4, 255, 255, 255, 255);
    circleRGBA(_display, _imu->get_debug_right().second.first, _imu->get_debug_right().second.second, 4, 255, 255, 255, 255);*/

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
