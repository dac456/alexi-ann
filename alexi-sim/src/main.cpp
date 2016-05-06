#include <SDL/SDL.h>
#include <SDL/SDL_image.h>

void render(SDL_Surface* disp){
    SDL_Flip(disp);
}

int main(int argc, char* argv[]){
    SDL_Init(SDL_INIT_EVERYTHING);

    SDL_Surface* surf = SDL_SetVideoMode(640, 480, 32, SDL_HWSURFACE | SDL_DOUBLEBUF);
    if(!surf){
        return 1;
    }
    else{

        SDL_Event evt;
        bool running = true;

        while(running){
            while(SDL_PollEvent(&evt)){
                if(evt.type == SDL_QUIT){
                    running = false;
                }
            }

            render(surf);
        }

    }

    SDL_Quit();
    return 0;
}
