#include "training_set.hpp"

#include <fstream>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

/*training_set::training_set(fs::path path, TRAINING_TYPE type)
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

    _input_set.resize(3, frames.size());
    _target_set.resize(1, frames.size());

    for(size_t i = 0; i < frames.size(); i++){
        frame_data data = _parse_frame(frames[i]);
        if(!data.warped){
            _frames.push_back(data);

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
        }
        else{
            std::cout << "excluding frame " << i << std::endl;
        }
    }
    std::cout << _target_set.columns() << std::endl;
}*/

training_set::training_set(std::vector<frame_data> frames, TRAINING_TYPE type)
    : _frames(frames)
{
    _input_set.resize(3, frames.size());

    switch(type){
        case DX:
        case DY:
        case DTHETA:
        _target_set.resize(1, frames.size());
        break;

        case ALL:
        _target_set.resize(3, frames.size());
        break;
    }

    for(size_t i = 0; i < _frames.size(); i++){
        frame_data data = _frames[i];
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
        }
        else{
            std::cout << "excluding frame " << i << std::endl;
        }
    }
    std::cout << "Set size: " << _target_set.columns() << std::endl;
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

std::vector<frame_data> training_set::get_frames(){
    return _frames;
}

blaze::DynamicMatrix<double> training_set::get_input_set(){
    return _input_set;
}

blaze::DynamicMatrix<double> training_set::get_target_set(){
    return _target_set;
}
