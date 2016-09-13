#include "data_preprocessor.hpp"
#include "../../contrib/alphanum.hpp"

//#define STB_IMAGE_IMPLEMENTATION
#include "../../contrib/stb/stb_image.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

data_preprocessor::data_preprocessor(std::vector<fs::path> set_paths){
    for(auto set_path : set_paths){
        std::cout << "Processing path " << set_path.string() << "..." << std::endl;

        std::vector<std::string> frames;
        if(fs::is_directory(set_path)){
            fs::directory_iterator itr{set_path};
            while(itr != fs::directory_iterator{}){
                fs::path frame_path = (*itr++).path();
                std::cout << "Adding frame " << frame_path.filename().string() << "..." << std::endl;
                frames.push_back(frame_path.string());
            }

            std::sort(frames.begin(), frames.end(), doj::alphanum_less<std::string>());
        }
        else{

        }
#if 0
        std::vector<std::string> images;
        fs::path raygrid_path = set_path / "../raygrid/";
        if(fs::is_directory(raygrid_path)){
            fs::directory_iterator itr{raygrid_path};
            while(itr != fs::directory_iterator{}){
                fs::path img_path = (*itr++).path();
                std::cout << "Adding raygrid image " << img_path.filename().string() << "..." << std::endl;
                images.push_back(fs::canonical(img_path).string());
            }

            std::sort(images.begin(), images.end(), doj::alphanum_less<std::string>());
        }

        std::vector<std::string> diff_images;
        fs::path diff_path = set_path / "../diff/";
        if(fs::is_directory(diff_path)){
            fs::directory_iterator itr{diff_path};
            while(itr != fs::directory_iterator{}){
                fs::path img_path = (*itr++).path();
                std::cout << "Adding diff image " << img_path.filename().string() << "..." << std::endl;
                diff_images.push_back(fs::canonical(img_path).string());
            }

            std::sort(diff_images.begin(), diff_images.end(), doj::alphanum_less<std::string>());
        }
#endif
        //Process frames
        for(size_t i = 0; i < frames.size(); i++){
            std::cout << "Parsing frame " << frames[i] << "..." << std::endl;
            frame_data data = _parse_frame(frames[i]);
            if(i == 0){
                data.dx_last = 0.0;
                data.dy_last = 0.0;
                data.dtheta_last = 0.0;
            }
            else{
                data.dx_last = _frames[i-1].dx;
                data.dy_last = _frames[i-1].dy;
                data.dtheta_last = _frames[i-1].dtheta;
            }
            if(!data.warped){
                _frames.push_back(data);
            }
            else{
                std::cout << "excluding frame " << i << " from path " << set_path.string() <<  std::endl;
            }
        }

        //Raygrid images
        #if 0
        std::array<double,256> zero;
        zero.fill(0.0);
        _images.push_back(zero);

        for(size_t i = 1; i < images.size(); i++){
            std::cout << "Processing raygrid image " << images[i-1] << "..." << std::endl;

            frame_data data = _parse_frame(frames[i-1]);
            if(!data.warped){
                int w, h, n;
                boost::shared_array<unsigned char> img_in = boost::shared_array<unsigned char>(stbi_load(images[i-1].c_str(), &w, &h, &n, 1));

                int px = data.px;
                int py = data.py;
                int start_x = (px - 8);
                int start_y = (py - 8);

                std::array<double,256> img_out;

                size_t idx = 0;
                #pragma omp parallel for
                for(size_t i = 0; i < 16; i++){
                    //int y = (start_y + i) % h;
                    int y = _wrap_value((start_y + i), h);
                    for(size_t j = 0; j < 16; j++){
                        //int x = (start_x + j) % w;
                        int x = _wrap_value((start_x + j), w);
                        img_out[idx] = static_cast<double>(img_in[x + (w * y)]) / 255.0;
                        idx++;
                    }
                }
                _images.push_back(img_out);
            }
        }
        #endif
#if 0
        //Diff images
        std::array<double,256> zero;
        zero.fill(0.0);
        _diff_images.push_back(zero);

        //#pragma omp parallel for
        for(size_t i = 1; i < diff_images.size(); i++){
            std::cout << "Processing diff image " << diff_images[i] << "..." << std::endl;

            frame_data data = _parse_frame(frames[i]);
            if(!data.warped){
                //TODO: correct order?
                size_t w = data.rw / data.dppx;
                size_t h = data.rl / data.dppy;

                std::ifstream fin(diff_images[i], std::ios::in);
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
                #pragma omp parallel for
                for(size_t i = 0; i < 16; i++){
                    //size_t y = (start_y + i) % h;
                    int y = _wrap_value((start_y + i), h);
                    for(size_t j = 0; j < 16; j++){
                        //size_t x = (start_x + j) % w;
                        int x = _wrap_value((start_x + j), w);
                        img[idx] = values[x + (w * y)];
                        idx++;
                    }
                }
                _diff_images.push_back(img);
            }
            else{
                std::cout << "excluding frame image " << i << " from path " << set_path.string() << std::endl;
            }
        }

        //Last diff
        _images.push_back(zero);
        for(size_t i = 1; i < _diff_images.size(); i++){
            _images.push_back(_diff_images[i-1]);
        }
#endif
    }
}

void data_preprocessor::run_processor(PREPROCESSOR proc_type){
    switch(proc_type){
        case AVERAGE:
        _average_frames(50);
        break;

        case THRESHOLD:
        _threshold_frames(0.01);
        break;

        case ACCUMULATE:
        _accumulate_frames(25);
        break;

        case FILTER:
        _filter_frames();
        break;

        case NORMALIZE:
        _normalize_frames(1);
        break;

        case LOWPASS:
        _lowpass_frames(0.3);
        break;
    }
}

void data_preprocessor::write_csv(fs::path p, int mode) {
    std::ofstream fout(p.string());

    for(auto frame : _frames) {
        if(!frame.warped) {
            switch(mode) {
                case 0:
                    fout << frame.left << "," << frame.right << "," << frame.dx << std::endl;
                break;

                case 1:
                    fout << frame.left << "," << frame.right << "," << frame.dy << std::endl;
                break;

                case 2:
                    fout << frame.left << "," << frame.right << "," << frame.dtheta << std::endl;
                break;

                case 3:
                    fout << frame.pitch << std::endl;
                break;
            }
        }
    }

    fout.close();
}

std::vector<frame_data> data_preprocessor::get_frames(){
    return _frames;
}

std::vector<std::array<double,256>> data_preprocessor::get_images(){
    return _images;
}

std::vector<std::array<double,256>> data_preprocessor::get_diff_images(){
    return _diff_images;
}

int data_preprocessor::_wrap_value(int value, int size){
    int out = value;

    if(value < 0){
        out = size + value;
    }
    else if(value >= size){
        out = value - size;
    }

    return out;
}

void data_preprocessor::_average_frames(size_t block_size){
    //average frame data
    std::vector<frame_data> new_frames;

    for(size_t i = 0; i < _frames.size(); i += block_size){
        frame_data f = _frames[i];

        double avg_dx = 0.0;
        double avg_dy = 0.0;
        double avg_dx_last = 0.0;
        double avg_dy_last = 0.0;
        double avg_dtheta = 0.0;
        double avg_theta = 0.0;
        for(size_t j = i; j < i + block_size; j++){
            avg_dx += _frames[j].dx;
            avg_dy += _frames[j].dy;
            avg_dx_last += _frames[j].dx_last;
            avg_dy_last += _frames[j].dy_last;
            avg_dtheta += _frames[j].dtheta;
            avg_theta += _frames[j].theta;
        }
        avg_dx /= static_cast<double>(block_size);
        avg_dy /= static_cast<double>(block_size);
        avg_dx_last /= static_cast<double>(block_size);
        avg_dy_last /= static_cast<double>(block_size);
        avg_dtheta /= static_cast<double>(block_size);
        avg_theta /= static_cast<double>(block_size);

        f.dx = avg_dx;
        f.dy = avg_dy;
        f.dx_last = avg_dx_last;
        f.dy_last = avg_dy_last;
        f.dtheta = avg_dtheta;
        f.theta = avg_theta;

        new_frames.push_back(f);
    }

    _frames = new_frames;

    //average raygrid images
    std::vector<std::array<double,256>> new_img;

    for(size_t i = 0; i < _images.size(); i += block_size){
        std::array<double,256> avg;
        avg.fill(0.0);

        for(size_t j = i; j < i + block_size; j++){
            for(size_t k = 0; k < 256; k++){
                avg[k] += _images[j][k];
            }
        }
        for(size_t k = 0; k < 256; k++){
            avg[k] /= static_cast<double>(block_size);
        }

        new_img.push_back(avg);
    }

    _images = new_img;

    //average diff image deltas
    std::vector<std::array<double,256>> new_diff_img;

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

        new_diff_img.push_back(avg);
    }

    _diff_images = new_diff_img;
}

void data_preprocessor::_threshold_frames(double interval){
    for(auto& frame : _frames) {
        if(frame.dtheta >= -0.05 && frame.dtheta <= 0.05) {
            frame.dtheta = 0.0;

            frame.dtheta_class[0] = 0;
            frame.dtheta_class[1] = 0;
            frame.dtheta_class[2] = 0;
            frame.dtheta_class[3] = 0;
        } else if(frame.dtheta > 0.05 && frame.dtheta <= 0.2) {
            frame.dtheta = 0.15;

            frame.dtheta_class[0] = 0;
            frame.dtheta_class[1] = 0;
            frame.dtheta_class[2] = 0;
            frame.dtheta_class[3] = 1;
        } else if(frame.dtheta > 0.2 && frame.dtheta <= 0.4) {
            frame.dtheta = 0.3;

            frame.dtheta_class[0] = 0;
            frame.dtheta_class[1] = 0;
            frame.dtheta_class[2] = 1;
            frame.dtheta_class[3] = 0;
        } else if(frame.dtheta > 0.4) {
            frame.dtheta = 0.6;

            frame.dtheta_class[0] = 0;
            frame.dtheta_class[1] = 1;
            frame.dtheta_class[2] = 0;
            frame.dtheta_class[3] = 0;
        } else if(frame.dtheta < -0.05 && frame.dtheta >= -0.2) {
            frame.dtheta = -0.15;

            frame.dtheta_class[0] = 1;
            frame.dtheta_class[1] = 0;
            frame.dtheta_class[2] = 0;
            frame.dtheta_class[3] = 1;
        } else if(frame.dtheta < -0.2 && frame.dtheta >= -0.4) {
            frame.dtheta = -0.3;

            frame.dtheta_class[0] = 1;
            frame.dtheta_class[1] = 0;
            frame.dtheta_class[2] = 1;
            frame.dtheta_class[3] = 0;
        } else if(frame.dtheta < -0.4) {
            frame.dtheta = -0.6;

            frame.dtheta_class[0] = 1;
            frame.dtheta_class[1] = 1;
            frame.dtheta_class[2] = 0;
            frame.dtheta_class[3] = 0;
        }
    }
}

void data_preprocessor::_accumulate_frames(size_t block_size){
    std::vector<frame_data> new_frames;

    for(size_t i = 0; i < _frames.size(); i += block_size){
        frame_data f = _frames[i];

        double accum_dx = 0.0;
        double accum_dy = 0.0;
        double accum_dtheta = 0.0;

        double avg_left = 0.0;
        double avg_right = 0.0;

        for(size_t j = i; j < i + block_size; j++){
            accum_dx += _frames[i].dx;
            accum_dy += _frames[i].dy;
            accum_dtheta += _frames[i].dtheta;

            avg_left += _frames[i].left;
            avg_right += _frames[i].right;
        }
        avg_left /= static_cast<double>(block_size);
        avg_right /= static_cast<double>(block_size);

        f.dx = accum_dx;
        f.dy = accum_dy;
        f.dtheta = accum_dtheta;
        f.left = avg_left;
        f.right = avg_right;

        new_frames.push_back(f);
    }

    _frames = new_frames;
}

void data_preprocessor::_filter_frames() {
    bool restart = true;

    while(restart) {
        bool found = false;
        for(size_t i = 0; i < _frames.size(); i++) {
            if(_frames[i].dtheta > 0.25 || _frames[i].dtheta < -0.25) {
                _frames.erase(_frames.begin() + i);

                found = true;
                break;
            }
        }

        if(!found) {
            restart = false;
        }
    }
}

void data_preprocessor::_normalize_frames(int mode) {
    std::ofstream fout("./stats.dat");
    double mean = 0.0;
    double dev = 0.0;
    double min = 0.0;
    double max = 0.0;

    switch(mode) {
        case 0:
        //vLeft
        for(auto& frame : _frames) {
            mean += frame.left;
        }
        mean /= (double)_frames.size();

        for(auto& frame : _frames) {
            dev += pow(frame.left - mean, 2.0);
        }
        dev /= (double)_frames.size();
        dev = sqrt(dev);

        for(auto& frame : _frames) {
            frame.left = (frame.left - mean) / dev;
        }
        fout << mean << " " << dev << std::endl;


        //vRight
        mean = 0.0;
        dev = 0.0;

        for(auto& frame : _frames) {
            mean += frame.right;
        }
        mean /= (double)_frames.size();

        for(auto& frame : _frames) {
            dev += pow(frame.right - mean, 2.0);
        }
        dev /= (double)_frames.size();
        dev = sqrt(dev);

        for(auto& frame : _frames) {
            frame.right = (frame.right - mean) / dev;
        }
        fout << mean << " " << dev << std::endl;

        //pitch
        mean = 0.0;
        dev = 0.0;

        for(auto& frame : _frames) {
            mean += frame.pitch;
        }
        mean /= (double)_frames.size();

        for(auto& frame : _frames) {
            dev += pow(frame.pitch - mean, 2.0);
        }
        dev /= (double)_frames.size();
        dev = sqrt(dev);

        for(auto& frame : _frames) {
            frame.pitch = (frame.pitch - mean) / dev;
        }
        fout << mean << " " << dev << std::endl;
        break;

        case 1:
        //vleft
        max = 0.0;
        min = 0.0;

        for(auto frame : _frames) {
            if(frame.left > max) max = frame.left;
            if(frame.left < min) min = frame.left;
        }
        for(auto& frame : _frames) {
            frame.left = (frame.left - ((max+min)/2.0)) / ((max-min)/2.0);
        }
        fout << max << " " << min << std::endl;

        //vright
        max = 0.0;
        min = 0.0;

        for(auto frame : _frames) {
            if(frame.right > max) max = frame.right;
            if(frame.right < min) min = frame.right;
        }
        for(auto& frame : _frames) {
            frame.right = (frame.right - ((max+min)/2.0)) / ((max-min)/2.0);
        }
        fout << max << " " << min << std::endl;

        //pitch
        max = 0.0;
        min = 0.0;

        for(auto frame : _frames) {
            if(frame.pitch > max) max = frame.pitch;
            if(frame.pitch < min) min = frame.pitch;
        }
        for(auto& frame : _frames) {
            frame.pitch = (frame.pitch - ((max+min)/2.0)) / ((max-min)/2.0);
        }
        fout << max << " " << min << std::endl;

        //roll
        max = 0.0;
        min = 0.0;

        for(auto frame : _frames) {
            if(frame.roll > max) max = frame.roll;
            if(frame.roll < min) min = frame.roll;
        }
        for(auto& frame : _frames) {
            frame.roll = (frame.roll - ((max+min)/2.0)) / ((max-min)/2.0);
        }
        fout << max << " " << min << std::endl;

        //dtheta
        max = 0.0;
        min = 0.0;

        for(auto frame : _frames) {
            if(frame.dtheta > max) max = frame.dtheta;
            if(frame.dtheta < min) min = frame.dtheta;
        }
        for(auto& frame : _frames) {
            frame.dtheta = (frame.dtheta - ((max+min)/2.0)) / ((max-min)/2.0);
        }
        fout << max << " " << min << std::endl;
        break;
    }

    fout.close();

}

void data_preprocessor::_lowpass_frames(double alpha) {
    double beta = 1.0 - alpha;

    /*double avg = 0.0;
    for(auto frame : _frames) {
        avg += frame.dtheta;
    }
    avg /= _frames.size();*/
    std::vector<double> s;
    s.resize(_frames.size());
    s[0] = _frames[0].dtheta;

    for(size_t i = 1; i < _frames.size(); i++) {
        s[i] = alpha*_frames[i].dtheta + beta*s[i-1];
    }

    for(size_t i = 1; i < _frames.size(); i++) {
        _frames[i].dtheta = s[i];
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
        ("vroll", po::value<double>()->default_value(0.0))

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
    out.roll = vm["vroll"].as<double>();

    out.vdl = vm["vdl"].as<double>();
    out.vda = vm["vda"].as<double>();

    out.v = vm["vdl"].as<double>();
    out.w = vm["vda"].as<double>();

    out.left = vm["vleft"].as<double>();
    out.right = vm["vright"].as<double>();

    out.pw = vm["rw"].as<double>() / vm["dppx"].as<double>();
    out.ph = vm["rl"].as<double>() / vm["dppy"].as<double>();

    return out;
}
