#include "training_set.hpp"
#include "../contrib/alphanum.hpp"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

training_set::training_set(fs::path path)
    : _set_path(path)
{
    std::vector<std::string> frames;
    if(fs::is_directory(path)){
        fs::directory_iterator itr{path};
        while(itr != fs::directory_iterator{}){
            frames.push_back((*itr++).path().string());
        }

        std::sort(frames.begin(), frames.end(), doj::alphanum_less<std::string>());
    }
    else{

    }

    if(frames.size() % 2 != 0){
        frames.pop_back();
    }

    _input_set.resize(5, (frames.size()/2)-1);
    _target_set.resize(3, (frames.size()/2)-1);

    size_t ij = 0;
    size_t tj = 0;
    for(size_t i = 0; i < frames.size()-2; i++){
        frame_data data = _parse_frame(frames[i]);
        if(i % 2 == 0){
            column(_input_set, ij) = blaze::StaticVector<double, 5UL, blaze::columnVector>(data.x, data.y, data.theta, data.v, data.w);
            ij++;
        }
        else{
            column(_target_set, tj) = blaze::StaticVector<double, 3UL, blaze::columnVector>(data.x, data.y, data.theta);
            tj++;
        }
    }
}

blaze::DynamicMatrix<double> training_set::get_input_set(){
    return _input_set;
}

blaze::DynamicMatrix<double> training_set::get_target_set(){
    return _target_set;
}

frame_data training_set::_parse_frame(fs::path file){
    po::variables_map vm;
    po::options_description desc("Configuration Options");
    desc.add_options()
        ("dppx", po::value<double>(), "Distance per pixel X")
        ("dppy", po::value<double>(), "Distance per pixel Y")
        ("rw", po::value<double>(), "Real width")
        ("rl", po::value<double>(), "Real length")

        ("warped", po::value<bool>(), "Did vehicle warp between this frame and last")

        ("vx", po::value<int>())
        ("vy", po::value<int>())
        ("vtheta", po::value<double>())

        ("vdl", po::value<double>())
        ("vda", po::value<double>())

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
    out.x = vm["vx"].as<int>();
    out.y = vm["vy"].as<int>();
    out.theta = vm["vtheta"].as<double>();

    out.v = vm["vdl"].as<double>();
    out.w = vm["vda"].as<double>();

    return out;
}
