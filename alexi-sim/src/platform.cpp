#include "platform.hpp"

platform::platform(SDL_Surface* disp, std::shared_ptr<fann_ffn> ann, double init_x, double init_y)
    : _display(disp)
    , _ann(ann)
    , _pos_x(init_x)
    , _pos_y(init_y)
{

}

void platform::step(){
    float in[4] = {_pos_x, _pos_y, 2.0f, 0.0f};
    float* out = _ann->predict(in);

    std::cout << out[0] << " " << out[1] << std::endl;
    _pos_x = out[0];
    _pos_y = out[1];
}

double platform::get_pos_x(){
    return _pos_x;
}

double platform::get_pos_y(){
    return _pos_y;
}
