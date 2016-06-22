#include "data_preprocessor.hpp"
#include "../../contrib/alphanum.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "../../contrib/stb/stb_image.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

data_preprocessor::data_preprocessor(fs::path set_path){
    std::vector<std::string> frames;
    if(fs::is_directory(set_path)){
        fs::directory_iterator itr{set_path};
        while(itr != fs::directory_iterator{}){
            frames.push_back((*itr++).path().string());
        }

        std::sort(frames.begin(), frames.end(), doj::alphanum_less<std::string>());
    }
    else{

    }

    std::vector<std::string> images;
    fs::path raygrid_path = set_path / "../diff/";
    if(fs::is_directory(raygrid_path)){
        fs::directory_iterator itr{raygrid_path};
        while(itr != fs::directory_iterator{}){
            images.push_back(fs::canonical((*itr++).path()).string());
        }

        std::sort(images.begin(), images.end(), doj::alphanum_less<std::string>());
    }

    for(size_t i = 0; i < frames.size(); i++){
        frame_data data = _parse_frame(frames[i]);
        if(!data.warped){
            _frames.push_back(data);
        }
        else{
            std::cout << "excluding frame " << i << std::endl;
        }
    }

    std::array<double,256> zero;
    zero.fill(0.0);
    _diff_images.push_back(zero);

    for(size_t i = 1; i < images.size(); i++){
        frame_data data = _parse_frame(frames[i]);
        if(!data.warped){
            //TODO: correct order?
            size_t w = data.rw / data.dppx;
            size_t h = data.rl / data.dppy;

            std::ifstream fin(images[i], std::ios::in);
            std::vector<double> values;
            while(!fin.eof()){
                double v;
                fin >> v;
                values.push_back(v);
            }
            fin.close();

            int px = data.px;
            int py = data.py;
            int start_x = (px - 8);
            int start_y = (py - 8);

            std::array<double,256> img;
            img.fill(0.0);

            size_t idx = 0;
            for(size_t i = 0; i < 16; i++){
                size_t y = (start_y + i) % h;
                for(size_t j = 0; j < 16; j++){
                    size_t x = (start_x + j) % w;
                    img[idx] = values[x + (w * y)];
                    idx++;
                }
            }
            _diff_images.push_back(img);
        }
        else{
            std::cout << "excluding frame image " << i << std::endl;
        }
    }
}

void data_preprocessor::run_processor(PREPROCESSOR proc_type){
    switch(proc_type){
        case AVERAGE:
        _average_frames(5);
        break;

        case THRESHOLD:
        _threshold_frames(0.01);
        break;
    }
}

std::vector<frame_data> data_preprocessor::get_frames(){
    return _frames;
}

std::vector<std::array<double,256>> data_preprocessor::get_diff_images(){
    return _diff_images;
}

void data_preprocessor::_average_frames(size_t block_size){
    //average frame data
    std::vector<frame_data> new_frames;

    for(size_t i = 0; i < _frames.size(); i += block_size){
        frame_data f = _frames[i];

        double avg_dx = 0.0;
        double avg_dy = 0.0;
        double avg_dtheta = 0.0;
        for(size_t j = i; j < i + block_size; j++){
            avg_dx += _frames[j].dx;
            avg_dy += _frames[j].dy;
            avg_dtheta += _frames[j].dtheta;
        }
        avg_dx /= static_cast<double>(block_size);
        avg_dy /= static_cast<double>(block_size);
        avg_dtheta /= static_cast<double>(block_size);

        f.dx = avg_dx;
        f.dy = avg_dy;
        f.dtheta = avg_dtheta;

        new_frames.push_back(f);
    }

    _frames = new_frames;

    //average image deltas
    std::vector<std::array<double,256>> new_img;
    
    for(size_t i = 0; i < _diff_images.size(); i += block_size){
        std::array<double,256> avg;
        avg.fill(0.0);

        for(size_t j = i; j < i + block_size; j++){
            for(size_t k = 0; k < 256; k++){
                avg[k] += _diff_images[j][k];
            }
        }
        for(size_t k = 0; k < 256; k++){
            avg[k] /= static_cast<double>(block_size);
        }

        new_img.push_back(avg);
    }

    _diff_images = new_img;
}

void data_preprocessor::_threshold_frames(double interval){
    for(size_t i = 0; i < _frames.size(); i++){
        if(_frames[i].dx < 0.01) _frames[i].dx = 0.0;
        if(_frames[i].dx > 0.01) _frames[i].dx = 0.01;
        if(_frames[i].dy < 0.01) _frames[i].dy = 0.0;
        if(_frames[i].dy > 0.01) _frames[i].dy = 0.01;
        if(_frames[i].dtheta < 0.05) _frames[i].dtheta = 0.0;
        if(_frames[i].dtheta > 0.05) _frames[i].dtheta = 0.05;
    }
}

frame_data data_preprocessor::_parse_frame(fs::path file){
    po::variables_map vm;
    po::options_description desc("Configuration Options");
    desc.add_options()
        ("dppx", po::value<double>(), "Distance per pixel X")
        ("dppy", po::value<double>(), "Distance per pixel Y")
        ("rw", po::value<double>(), "Real width")
        ("rl", po::value<double>(), "Real length")

        ("warped", po::value<bool>(), "Did vehicle warp between this frame and last")

        ("vx", po::value<int>(), "Pixel-space x position")
        ("vy", po::value<int>(), "Pixel-space y position")
        ("vxr", po::value<double>(), "Real x position")
        ("vyr", po::value<double>(), "Real y position")
        ("vdxr", po::value<double>(), "Real x position delta")
        ("vdyr", po::value<double>(), "Real y position delta")
        ("vtheta", po::value<double>())
        ("vdtheta", po::value<double>())
        ("vpitch", po::value<double>())

        ("vdl", po::value<double>())
        ("vda", po::value<double>())

        ("vleft", po::value<double>())
        ("vright", po::value<double>())
        ("vleftlast", po::value<double>())
        ("vrightlast", po::value<double>())


        ("gx1x", po::value<int>())
        ("gx1y", po::value<int>())
        ("gx2x", po::value<int>())
        ("gx2y", po::value<int>())
        ("gy1x", po::value<int>())
        ("gy1y", po::value<int>())
        ("gy2x", po::value<int>())
        ("gy2y", po::value<int>())
    ;

    std::ifstream cfg(file.string());

    po::store(po::parse_config_file(cfg , desc), vm);
    po::notify(vm);

    frame_data out;
    out.warped = vm["warped"].as<bool>();
    out.dppx = vm["dppx"].as<double>();
    out.dppy = vm["dppy"].as<double>();

    out.rw = vm["rw"].as<double>();
    out.rl = vm["rl"].as<double>();

    out.x = vm["vxr"].as<double>();
    out.y = vm["vyr"].as<double>();
    out.px = vm["vx"].as<int>();
    out.py = vm["vy"].as<int>();
    out.dx = vm["vdxr"].as<double>();
    out.dy = vm["vdyr"].as<double>();
    out.theta = vm["vtheta"].as<double>();
    out.dtheta = vm["vdtheta"].as<double>();
    out.pitch = vm["vpitch"].as<double>();

    out.v = vm["vdl"].as<double>();
    out.w = vm["vda"].as<double>();

    out.left = vm["vleft"].as<double>();
    out.right = vm["vright"].as<double>();

    out.pw = vm["rw"].as<double>() / vm["dppx"].as<double>();
    out.ph = vm["rl"].as<double>() / vm["dppy"].as<double>();

    return out;
}
