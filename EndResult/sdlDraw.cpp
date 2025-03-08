//g++ -o sdlDraw.out sdlDraw.cpp $(sdl2-config --cflags --libs)

#include <SDL2/SDL.h>
#include <iostream>
#include <vector>

const int WINDOW_WIDTH = 280;
const int WINDOW_HEIGHT = 280;
const int PIXEL_SIZE = 10;  // Each "pixel" will be 10x10

// Function to draw the grid
void drawCanvas(SDL_Renderer* renderer, std::vector<std::vector<bool>>& pixels) {
    for (int i = 0; i < WINDOW_WIDTH / PIXEL_SIZE; ++i) {
        for (int j = 0; j < WINDOW_HEIGHT / PIXEL_SIZE; ++j) {
            if (pixels[i][j]) {
                SDL_Rect rect = {i * PIXEL_SIZE, j * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE};
                SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255); // White for drawing
                SDL_RenderFillRect(renderer, &rect);
            } else {
                SDL_Rect rect = {i * PIXEL_SIZE, j * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE};
                SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255); // Black for background
                SDL_RenderFillRect(renderer, &rect);
            }
        }
    }
}

// Function to print pixel values
void printPixelValues(const std::vector<std::vector<bool>>& pixels) {
    for (int i = 0; i < pixels.size(); ++i) {
        for (int j = 0; j < pixels[i].size(); ++j) {
            std::cout << (pixels[j][i] * 255) / 255.5 << " "; // 1 for white, 0 for black
        }
        //std::cout << std::endl; // Newline after each row
    }

    std::cout << std::endl;
}

int main() {
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
        return -1;
    }

    // Create window
    SDL_Window* window = SDL_CreateWindow("Draw on Canvas", 
                                          SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                          WINDOW_WIDTH, WINDOW_HEIGHT, 
                                          SDL_WINDOW_SHOWN);
    if (window == nullptr) {
        std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return -1;
    }

    // Create renderer
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (renderer == nullptr) {
        std::cerr << "Renderer could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }

    // Canvas to track drawn pixels (2D vector of booleans)
    std::vector<std::vector<bool>> pixels(WINDOW_WIDTH / PIXEL_SIZE, 
                                           std::vector<bool>(WINDOW_HEIGHT / PIXEL_SIZE, false));

    bool quit = false;
    SDL_Event e;
    bool isDrawing = false;

    // Main loop
    while (!quit) {
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) {
                quit = true;
            } else if (e.type == SDL_MOUSEBUTTONDOWN) {
                if (e.button.button == SDL_BUTTON_LEFT) {
                    // Start drawing
                    isDrawing = true;
                    int x = e.button.x / PIXEL_SIZE;
                    int y = e.button.y / PIXEL_SIZE;
                    if (x < WINDOW_WIDTH / PIXEL_SIZE && y < WINDOW_HEIGHT / PIXEL_SIZE) {
                        pixels[x][y] = true;
                    }
                } else if (e.button.button == SDL_BUTTON_RIGHT) {
                    // Right-click: print the pixel values and quit
                    printPixelValues(pixels);
                    quit = true;
                }
            } else if (e.type == SDL_MOUSEMOTION) {
                if (isDrawing) {
                    // Continue drawing while mouse is moving
                    int x = e.motion.x / PIXEL_SIZE;
                    int y = e.motion.y / PIXEL_SIZE;
                    if (x < WINDOW_WIDTH / PIXEL_SIZE && y < WINDOW_HEIGHT / PIXEL_SIZE) {
                        pixels[x][y] = true;
                    }
                }
            } else if (e.type == SDL_MOUSEBUTTONUP) {
                if (e.button.button == SDL_BUTTON_LEFT) {
                    // Stop drawing
                    isDrawing = false;
                }
            }
        }

        // Render the canvas
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);  // Set background color to black
        SDL_RenderClear(renderer);
        
        // Draw the updated canvas
        drawCanvas(renderer, pixels);

        // Present the renderer
        SDL_RenderPresent(renderer);
    }

    // Cleanup and close
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}

