#pragma once
#include <CL/cl.hpp>

// OpenCL objects
extern cl::Device device;
extern cl::CommandQueue queue;
extern cl::Kernel kernel;
extern cl::Context context;
extern cl::Program program;
extern cl::Buffer cl_output;
extern cl::Buffer cl_objects;
extern cl::Buffer cl_vertices;
extern cl::Buffer cl_normals;
extern cl::Buffer cl_uvs;
extern cl::Buffer cl_triangles;
extern cl::Buffer cl_octrees;
extern cl::Buffer cl_octreeTris;
extern cl::Buffer cl_textures;
extern cl::BufferGL cl_vbo;
extern std::vector<cl::Memory> cl_vbos;

void pickPlatform(cl::Platform& platform, const std::vector<cl::Platform>& platforms);
void pickDevice(cl::Device& device, const std::vector<cl::Device>& devices);
void initOpenCL();
void initCLKernel();
void runKernel();

void cleanUp();
