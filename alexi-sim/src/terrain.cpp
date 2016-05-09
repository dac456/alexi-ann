#include "terrain.hpp"

#include <iostream>
#include <SDL/SDL_rotozoom.h>
#include <SDL/SDL_gfxPrimitives.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../../contrib/stb/stb_image.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

terrain::terrain(SDL_Surface* display, fs::path map_file, double scale)
    : _display(display)
    , _scale(scale)
{
    if(fs::exists(map_file)){
        int w, h, n;
        _heights = boost::shared_array<unsigned char>(stbi_load(map_file.string().c_str(), &w, &h, &n, 1));
        _width = w;
        _height = h;

        int h2 = h * _scale;
        int w2 = w * _scale;
        int x_ratio = static_cast<int>((w << 16) / w2) + 1;
        int y_ratio = static_cast<int>((h << 16) / h2) + 1;
        boost::shared_array<unsigned char> temp(new unsigned char[h2 * w2]);

        for(int y = 0; y < h2; y++) {
            for(int x = 0; x < w2; x++) {
                int x2 = ((x * x_ratio) >> 16) ;
                int y2 = ((y * y_ratio) >> 16) ;
                temp[(y * w2) + x] = _heights[(y2 * w) + x2] ;
            }
        }

        _heights = temp;
        _display_width = w2;
        _display_height = h2;
    }
}

void terrain::update(){
    for(int y = 0; y < _display_height; y++){
        for(int x = 0; x < _display_width; x++){
            size_t idx = x + (_display_width * y);
            pixelRGBA(_display, x, y, _heights[idx], _heights[idx], _heights[idx], 255);
        }
    }
}

void terrain::update_pixel_by_delta(size_t x, size_t y, unsigned char delta){
    size_t idx = x + (_width * y);
    size_t d = _heights[idx] + delta;

    if(d > 255) d = 255;
    if(d < 0) d = 0;

    _heights[idx] = static_cast<unsigned char>(d);
}

int terrain::get_width(){
    return _width;
}

int terrain::get_height(){
    return _height;
}

int terrain::get_display_width(){
    return _display_width;
}

int terrain::get_display_height(){
    return _display_height;
}
