#ifndef __PLATFORM_HPP
#define __PLATFORM_HPP

#include <memory>
#include <SDL/SDL.h>

#include "fann_ffn.hpp"

class platform{
private:
    SDL_Surface* _display;
    std::shared_ptr<fann_ffn> _ann;

    double _pos_x;
    double _pos_y;

public:
    platform(SDL_Surface* disp, std::shared_ptr<fann_ffn> ann, double init_x, double init_y);

    void step();

    double get_pos_x();
    double get_pos_y();
};

typedef std::shared_ptr<platform> platform_ptr;

#endif
