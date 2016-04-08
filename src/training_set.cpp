#include "training_set.hpp"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

training_set::training_set(fs::path path)
    : _set_path(path)
{
    if(fs::is_directory(path)){
        
    }
    else{

    }
}

frame_data training_set::_parse_frame(fs::path file){
    po::variables_map vm;
    po::options_description desc("Configuration Options");
    desc.add_options()
        ("vx", po::value<int>())
    ;

    std::ifstream cfg(file.string());

    po::store(po::parse_config_file(cfg , desc), vm);
    po::notify(vm);

    frame_data out;
    out.x = vm["vx"].as<int>();

    return out;
}
