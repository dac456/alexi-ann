#ifndef __DATA_PREPROCESSOR_HPP
#define __DATA_PREPROCESSOR_HPP

#include "common.hpp"
#include "training_set.hpp"

enum PREPROCESSOR{
    AVERAGE,
    THRESHOLD,
    ACCUMULATE,
    FILTER,
    NORMALIZE,
    LOWPASS,
    NOISE,
};

class data_preprocessor{
private:
    std::vector<frame_data> _frames;
    std::vector<std::array<double,1024>> _images;
    std::vector<std::array<double,1024>> _diff_images;

public:
    data_preprocessor(std::vector<fs::path> set_paths);

    void run_processor(PREPROCESSOR proc_type);
    void write_csv(fs::path p, int mode);

    std::vector<frame_data> get_frames();
    std::vector<std::array<double,1024>> get_images();
    std::vector<std::array<double,1024>> get_diff_images();


private:
    int _wrap_value(int value, int size);

    void _average_frames(size_t block_size);
    void _threshold_frames(double interval);
    void _accumulate_frames(size_t block_size);
    void _filter_frames();
    void _normalize_frames(int mode);
    void _lowpass_frames(double alpha);
    void _add_noise(double min, double max);

    frame_data _parse_frame(fs::path file);
};

#endif
