#include "training_set.hpp"

#include <fstream>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <blaze/util/serialization/Archive.h>

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

training_set::training_set(std::vector<frame_data> frames, std::vector<std::array<double,256>> images, std::vector<std::array<double,256>> diff_images, TRAINING_TYPE type)
    : _frames(frames)
    , _images(images)
    , _diff_images(diff_images)
    , _type(type)
{
    if(type != TERRAIN){
        _input_set.resize(4, frames.size());
    }
    else{
        _input_set.resize(262, frames.size());
    }

    switch(type){
        case DX:
        case DY:
        case DTHETA:
        _target_set.resize(1, frames.size());
        break;

        case TERRAIN:
        _target_set.resize(256, diff_images.size());
        break;

        case ALL:
        _target_set.resize(3, frames.size());
        break;
    }

    if(_type != TERRAIN){
        for(size_t i = 0; i < _frames.size(); i++){
            frame_data data = _frames[i];
            column(_input_set, i) = blaze::StaticVector<double, 4UL, blaze::columnVector>(data.left, data.right, data.pitch, data.roll);
            //column(_input_set, i) = blaze::StaticVector<double, 3UL, blaze::columnVector>(data.vdl, data.vda, data.pitch/*, data.roll*/);

            switch(type){
                case DX:
                //column(_input_set, i) = blaze::StaticVector<double, 5UL, blaze::columnVector>(data.left, data.right, data.pitch, data.roll, data.dx_last);
                column(_target_set, i) = blaze::StaticVector<double, 1UL, blaze::columnVector>(data.dx);
                break;

                case DY:
                //column(_input_set, i) = blaze::StaticVector<double, 5UL, blaze::columnVector>(data.left, data.right, data.pitch, data.roll, data.dy_last);
                column(_target_set, i) = blaze::StaticVector<double, 1UL, blaze::columnVector>(data.dy);
                break;

                case DTHETA:
                //column(_input_set, i) = blaze::StaticVector<double, 5UL, blaze::columnVector>(data.left, data.right, data.pitch, data.roll, data.dtheta_last);
                column(_target_set, i) = blaze::StaticVector<double, 1UL, blaze::columnVector>(data.dtheta);
                break;
            }
        }
    }
    else{
        for(size_t i = 0; i < _diff_images.size(); i++){
            frame_data data = _frames[i];
            //column(_input_set, i) = blaze::StaticVector<double, 261UL, blaze::columnVector>(data.left, data.right, data.pitch, data.dx_last, data.dy_last);
            _input_set(0,i) = data.left;
            _input_set(1,i) = data.right;
            _input_set(2,i) = data.pitch;
            _input_set(3,i) = data.dx_last;
            _input_set(4,i) = data.dy_last;
            _input_set(5,i) = data.theta;
            for(int y = 6; y < 262; y++){
                _input_set(y,i) = _images[i][y-5];
            }

            switch(type){
                case TERRAIN:
                for(int y = 0; y < 256; y++){
                    _target_set(y,i) = _diff_images[i][y];
                }
                break;
            }
        }
    }
    std::cout << "Set size: " << _target_set.columns() << std::endl;
}

void training_set::save_fann_data(fs::path file){
    std::ofstream fout(file.string());

    if(_type != TERRAIN){
        fout << _input_set.columns() << " 4 1" << std::endl;
        for(size_t i = 0; i < _input_set.columns(); i++){
            fout << _input_set(0,i) << " " << _input_set(1,i) << " " << _input_set(2,i) << " " << _input_set(3,i) /*<< " " << _input_set(4,i)*/ << std::endl;
            fout << _target_set(0,i) /*<< " " << _target_set(1,i)*/ << std::endl;
        }
    }
    else{
        fout << _input_set.columns() << " 262 256" << std::endl;
        for(size_t i = 0; i < _input_set.columns(); i++){
            //fout << _input_set(0,i) << " " << _input_set(1,i) << " " << _input_set(2,i) << " " << _input_set(3,i) << " " << _input_set(4,i) << std::endl;
            for(size_t r = 0; r < _input_set.rows(); r++){
                fout << _input_set(r,i) << " ";
            }
            fout << std::endl;

            for(size_t r = 0; r < _target_set.rows(); r++){
                fout << _target_set(r,i) << " ";
            }
            fout << std::endl;
        }
    }

    fout.close();
}

void training_set::save_blaze_data(fs::path file){
    std::string output = file.string() + ".blaze.data";

    blaze::Archive<std::ofstream> archive(output);
    archive << _input_set << _target_set;
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
