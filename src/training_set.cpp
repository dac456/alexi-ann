#include "training_set.hpp"
#include "../contrib/alphanum.hpp"

#include <fstream>
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

    size_t num_frames = frames.size();
    size_t set_size = frames.size()/2;
    if(set_size % 2 != 0){
        set_size -= 1;
        num_frames -= 2;
    }

    _input_set.resize(2, set_size);
    _target_set.resize(2, set_size);
    //std::vector< blaze::StaticVector<double, 2UL, blaze::columnVector> > input_vector;
    //std::vector< blaze::StaticVector<double, 2UL, blaze::columnVector> > target_vector;

    size_t ij = 0;
    size_t tj = 0;
    bool assign_as_input = true;
    for(size_t i = 0; i < num_frames; i++){
        frame_data data = _parse_frame(frames[i]);
    //    if(i > 0){
    //        if(data.x == _last_frame.x && data.y == _last_frame.y){
    //            i++;
    //        }
    //        else{

                if(assign_as_input){
                    column(_input_set, ij) = blaze::StaticVector<double, 2UL, blaze::columnVector>(data.x, data.y);
                    //input_vector.push_back(blaze::StaticVector<double, 2UL, blaze::columnVector>(data.x, data.y));
                    ij++;
                    //i++;
                    assign_as_input = false;
                }
                else{
                    column(_target_set, tj) = blaze::StaticVector<double, 2UL, blaze::columnVector>(data.x, data.y);
                    //target_vector.push_back(blaze::StaticVector<double, 2UL, blaze::columnVector>(data.x, data.y));
                    tj++;
                    assign_as_input = true;
                }
        //    }
        //}
        //else{
        //    //column(_input_set, ij) = blaze::StaticVector<double, 5UL, blaze::columnVector>(data.x/double(data.pw), data.y/double(data.ph), data.theta/6.28, data.v, data.w);
        //    input_vector.push_back(blaze::StaticVector<double, 2UL, blaze::columnVector>(data.x, data.y));
        //    ij++;
        //    i++;
        //    assign_as_input = false;
        //}

        //_last_frame = data;
    }

    /*if(input_vector.size() < target_vector.size()){
        target_vector.resize(input_vector.size());
    }
    else if(target_vector.size() < input_vector.size()){
        input_vector.resize(target_vector.size());
    }

    _input_set.resize(2, input_vector.size());
    _target_set.resize(2, target_vector.size());
    for(size_t i = 0; i < input_vector.size(); i++){
        column(_input_set, i) = input_vector[i];
    }
    for(size_t i = 0; i < target_vector.size(); i++){
        column(_target_set, i) = target_vector[i];
    }*/
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

        ("vx", po::value<int>(), "Pixel-space x position")
        ("vy", po::value<int>(), "Pixel-space y position")
        ("vxr", po::value<double>(), "Real x position")
        ("vyr", po::value<double>(), "Real y position")
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
    out.rw = vm["rw"].as<double>();
    out.rl = vm["rl"].as<double>();

    out.x = vm["vxr"].as<double>();
    out.y = vm["vyr"].as<double>();
    out.theta = vm["vtheta"].as<double>();

    out.v = vm["vdl"].as<double>();
    out.w = vm["vda"].as<double>();

    out.pw = vm["rw"].as<double>() / vm["dppx"].as<double>();
    out.ph = vm["rl"].as<double>() / vm["dppy"].as<double>();

    return out;
}
