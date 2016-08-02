#include <SDL/SDL.h>
#include <SDL/SDL_image.h>

#include <chrono>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "experiment.hpp"

void resize_window(SDL_Surface* surf, size_t w, size_t h){
    SDL_Surface* tmp = SDL_SetVideoMode(w, h, 32, SDL_HWSURFACE | SDL_DOUBLEBUF);
    SDL_FreeSurface(surf);
    surf = tmp;
}

int main(int argc, char* argv[]){
    //Get options
    po::options_description desc("supported options");
    desc.add_options()
        ("experiment", po::value<std::string>()->required(), "Path to experiment data to load.")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    SDL_Init(SDL_INIT_EVERYTHING);

    SDL_Surface* surf = SDL_SetVideoMode(1, 1, 32, SDL_HWSURFACE | SDL_DOUBLEBUF);
    if(!surf){
        return 1;
    }
    else{

        SDL_Event evt;
        bool running = true;

        experiment expr(surf, vm["experiment"].as<std::string>());
        resize_window(surf, expr.get_terrain()->get_display_width(), expr.get_terrain()->get_display_height());
        std::cout << expr.get_terrain()->get_display_width() << " " << expr.get_terrain()->get_display_height() << std::endl;

        std::chrono::steady_clock::time_point last_time = std::chrono::steady_clock::now();
        while(running){
            while(SDL_PollEvent(&evt)){
                if(evt.type == SDL_QUIT){
                    running = false;
                }
            }

            if(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - last_time) >= std::chrono::milliseconds(1)){
                expr.step();
                last_time = std::chrono::steady_clock::now();
            }
        }

    }

    SDL_Quit();
    return 0;
}
