#ifndef __DATA_PREPROCESSOR_HPP
#define __DATA_PREPROCESSOR_HPP

#include "common.hpp"
#include "training_set.hpp"

enum PREPROCESSOR{
    AVERAGE,
    THRESHOLD
};

class data_preprocessor{
private:
    std::vector<frame_data> _frames;

public:
    data_preprocessor(fs::path set_path);

    void run_processor(PREPROCESSOR proc_type);
    std::vector<frame_data> get_frames();

private:
    void _average_frames(size_t block_size);
    void _threshold_frames(double interval);

    frame_data _parse_frame(fs::path file);
};

#endif
