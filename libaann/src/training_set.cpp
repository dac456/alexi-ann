#include "training_set.hpp"
#include "../../contrib/alphanum.hpp"

#include <fstream>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

training_set::training_set(fs::path path, TRAINING_TYPE type)
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

    /*if(frames.size() % 2 != 0){
        frames.pop_back();
    }

    size_t num_frames = frames.size();
    size_t set_size = frames.size()/2;
    if(set_size % 2 != 0){
        set_size -= 1;
        num_frames -= 2;
    }*/

    _input_set.resize(3, frames.size());
    _target_set.resize(1, frames.size());
    //std::vector< blaze::StaticVector<double, 2UL, blaze::columnVector> > input_vector;
    //std::vector< blaze::StaticVector<double, 2UL, blaze::columnVector> > target_vector;

    size_t ij = 0;
    size_t tj = 0;
    bool assign_as_input = true;
    for(size_t i = 0; i < frames.size(); i++){
        frame_data data = _parse_frame(frames[i]);
        if(!data.warped){
            switch(type){
                case DX:
                column(_input_set, i) = blaze::StaticVector<double, 3UL, blaze::columnVector>(data.left, data.right, data.pitch);
                column(_target_set, i) = blaze::StaticVector<double, 1UL, blaze::columnVector>(data.dx);
                break;

                case DY:
                column(_input_set, i) = blaze::StaticVector<double, 3UL, blaze::columnVector>(data.left, data.right, data.pitch);
                column(_target_set, i) = blaze::StaticVector<double, 1UL, blaze::columnVector>(data.dy);
                break;

                case DTHETA:
                column(_input_set, i) = blaze::StaticVector<double, 3UL, blaze::columnVector>(data.left, data.right, data.pitch);
                column(_target_set, i) = blaze::StaticVector<double, 1UL, blaze::columnVector>(data.dtheta);
                break;
            }
    //    if(i > 0){
    //        if(data.x == _last_frame.x && data.y == _last_frame.y){
    //            i++;
    //        }
    //        else{

                //if(assign_as_input){
                    //column(_input_set, i) = blaze::StaticVector<double, 3UL, blaze::columnVector>(data.left, data.right, data.pitch);
                    //input_vector.push_back(blaze::StaticVector<double, 2UL, blaze::columnVector>(data.x, data.y));
                //    ij++;
                    //i++;
                //    assign_as_input = false;
                //}
                //else{
                    //column(_target_set, i) = blaze::StaticVector<double, 2UL, blaze::columnVector>(data.dx, data.dy);
                    //target_vector.push_back(blaze::StaticVector<double, 2UL, blaze::columnVector>(data.x, data.y));
                //    tj++;
                //    assign_as_input = true;
                //}
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
        else{
            std::cout << "excluding frame " << i << std::endl;
        }
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

void training_set::save_fann_data(fs::path file){
    std::ofstream fout(file.string());

    fout << _input_set.columns() << " 3 1" << std::endl;
    for(size_t i = 0; i < _input_set.columns(); i++){
        fout << _input_set(0,i) << " " << _input_set(1,i) << " " << _input_set(2,i) /*<< " " << _input_set(3,i)*/ << std::endl;
        fout << _target_set(0,i) /*<< " " << _target_set(1,i)*/ << std::endl;
    }

    fout.close();
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

    out.rw = vm["rw"].as<double>();
    out.rl = vm["rl"].as<double>();

    out.x = vm["vxr"].as<double>();
    out.y = vm["vyr"].as<double>();
    out.dx = vm["vdxr"].as<double>();
    out.dy = vm["vdyr"].as<double>();
    out.theta = vm["vtheta"].as<double>();
    out.dtheta = vm["vdtheta"].as<double>();
    //out.pitch = vm["vpitch"].as<double>();
    out.pitch = 0.0;

    out.v = vm["vdl"].as<double>();
    out.w = vm["vda"].as<double>();

    out.left = vm["vleft"].as<double>();
    out.right = vm["vright"].as<double>();

    out.pw = vm["rw"].as<double>() / vm["dppx"].as<double>();
    out.ph = vm["rl"].as<double>() / vm["dppy"].as<double>();

    return out;
}
