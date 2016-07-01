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
    std::vector<std::array<double,256>> _diff_images;

public:
    data_preprocessor(std::vector<fs::path> set_paths);

    void run_processor(PREPROCESSOR proc_type);
    std::vector<frame_data> get_frames();
    std::vector<std::array<double,256>> get_diff_images();

private:
    void _average_frames(size_t block_size);
    void _threshold_frames(double interval);

    frame_data _parse_frame(fs::path file);
};

#endif
