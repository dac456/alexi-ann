#ifndef __TERRAIN_HPP
#define __TERRAIN_HPP

#include <memory>

#include <SDL/SDL.h>
#include <SDL/SDL_image.h>

#include <boost/smart_ptr.hpp>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

class terrain{
private:
    SDL_Surface* _surf;
    SDL_Surface* _display;

    boost::shared_array<unsigned char> _heights;
    size_t _width;
    size_t _height;
    size_t _display_width;
    size_t _display_height;
    double _real_width;
    double _real_height;

    double _scale;

public:
    terrain(SDL_Surface* display, fs::path map_file, double scale);

    void update();
    void update_pixel_by_delta(size_t x, size_t y, unsigned char delta);

    int get_width();
    int get_height();
    int get_display_width();
    int get_display_height();
};

typedef std::shared_ptr<terrain> terrain_ptr;

#endif
