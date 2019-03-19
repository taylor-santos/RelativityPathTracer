#pragma once
#include "Mesh.h"
#define cimg_use_jpeg
#include "CImg.h"
#include <CL/cl.hpp>
#include <chrono>
#include <fstream>
#include <sstream>

extern unsigned int framenumber;
extern bool downKeys[9];
extern cl_float3 cameraVelocity;
extern cl_float4 cameraPos;
extern bool stopTime;
extern int interval;
extern bool changedTime;
extern bool changedInterval;
extern float currTime;
extern cl_float3 white_point;
extern float ambient;
extern std::chrono::time_point<std::chrono::high_resolution_clock> clock_start, clock_end, clock_prev;
extern Mesh theMesh;
extern std::vector<unsigned char> textures;
extern std::vector<int> textureValues; // {index, width, height}

void keyDown(unsigned char key, int x, int y);
void keyUp(unsigned char key, int x, int y);
void render();
void inputScene();
bool ReadTexture(std::string path);
bool ReadOBJ(std::string path, Mesh &mesh);
