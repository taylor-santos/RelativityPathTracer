#define __CL_ENABLE_EXCEPTIONS

#include "CLSetup.h"
#include "Object.h"
#include "Render.h"
#include "gl_interop.h"
#include <iostream>
#include <Windows.h>
#include <fstream>

cl::Device device;
cl::CommandQueue queue;
cl::Kernel kernel;
cl::Context context;
cl::Program program;
cl::Buffer cl_output;
cl::Buffer cl_objects;
cl::Buffer cl_vertices;
cl::Buffer cl_normals;
cl::Buffer cl_uvs;
cl::Buffer cl_triangles;
cl::Buffer cl_octrees;
cl::Buffer cl_octreeTris;
cl::Buffer cl_textures;
cl::BufferGL cl_vbo;
std::vector<cl::Memory> cl_vbos;

void pickPlatform(cl::Platform& platform, const std::vector<cl::Platform>& platforms) {

	if (platforms.size() == 1) platform = platforms[0];
	else {
		int input = 0;
		std::cout << "\nChoose an OpenCL platform: ";
		std::cin >> input;

		// handle incorrect user input
		while (input < 1 || input > platforms.size()) {
			std::cin.clear(); //clear errors/bad flags on cin
			std::cin.ignore(std::cin.rdbuf()->in_avail(), '\n'); // ignores exact number of chars in cin buffer
			std::cout << "No such option. Choose an OpenCL platform: ";
			std::cin >> input;
		}
		platform = platforms[input - 1];
	}
}

void pickDevice(cl::Device& device, const std::vector<cl::Device>& devices) {

	if (devices.size() == 1) device = devices[0];
	else {
		int input = 0;
		std::cout << "\nChoose an OpenCL device: ";
		std::cin >> input;

		// handle incorrect user input
		while (input < 1 || input > devices.size()) {
			std::cin.clear(); //clear errors/bad flags on cin
			std::cin.ignore(std::cin.rdbuf()->in_avail(), '\n'); // ignores exact number of chars in cin buffer
			std::cout << "No such option. Choose an OpenCL device: ";
			std::cin >> input;
		}
		device = devices[input - 1];
	}
}

void initOpenCL(std::string filename)
{
	// Get all available OpenCL platforms (e.g. AMD OpenCL, Nvidia CUDA, Intel OpenCL)
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	std::cout << "Available OpenCL platforms : " << std::endl << std::endl;
	for (int i = 0; i < platforms.size(); i++)
		std::cout << "\t" << i + 1 << ": " << platforms[i].getInfo<CL_PLATFORM_NAME>() << std::endl;

	std::cout << std::endl << "WARNING: " << std::endl << std::endl;
	std::cout << "OpenCL-OpenGL interoperability is only tested " << std::endl;
	std::cout << "on discrete GPUs from Nvidia and AMD" << std::endl;
	std::cout << "Other devices (such as Intel integrated GPUs) may fail" << std::endl << std::endl;

	// Pick one platform
	cl::Platform platform;
	//pickPlatform(platform, platforms);
	platform = platforms[0];
	std::cout << "\nUsing OpenCL platform: \t" << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

	// Get available OpenCL devices on platform
	std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

	std::cout << "Available OpenCL devices on this platform: " << std::endl << std::endl;
	for (int i = 0; i < devices.size(); i++) {
		std::cout << "\t" << i + 1 << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
		std::cout << "\t\tMax compute units: " << devices[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
		std::cout << "\t\tMax work group size: " << devices[i].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl << std::endl;
	}


	// Pick one device
	//Device device;
	pickDevice(device, devices);
	std::cout << "\nUsing OpenCL device: \t" << device.getInfo<CL_DEVICE_NAME>() << std::endl;

	// Create an OpenCL context on that device.
	// Windows specific OpenCL-OpenGL interop
	cl_context_properties properties[] =
	{
		CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
		CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
		CL_CONTEXT_PLATFORM, (cl_context_properties)platform(),
		0
	};

	try {
		context = cl::Context({device}, properties);
	}
	catch (cl::Error &e) {
		std::cerr << e.what() << ": " << e.err() << std::endl;
		throw e;
	}

	// Create a command queue
	queue = cl::CommandQueue(context, device);


	// Convert the OpenCL source code to a string// Convert the OpenCL source code to a string

	std::ifstream file(filename);
	if (!file) {
		std::cout << "\nNo OpenCL file found!" << std::endl << "Exiting..." << std::endl;
		system("PAUSE");
		exit(1);
	}
	std::string source{ std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>() };

	const char* kernel_source = source.c_str();

	// Create an OpenCL program with source
	program = cl::Program(context, kernel_source);

	// Build the program for the selected device 
	cl_int result = program.build({ device }); // "-cl-fast-relaxed-math"
	if (result) std::cout << "Error during compilation OpenCL code!!!\n (" << result << ")" << std::endl;
}



void initRelativityKernel() {

	// pick a rendermode
	unsigned int rendermode = 1;

	// Create a kernel (entry point in the OpenCL source program)
	kernel = cl::Kernel(program, "render_kernel");

	int object_count = cpu_objects.size();

	// specify OpenCL kernel arguments
	kernel.setArg(0, cl_objects);
	kernel.setArg(1, object_count);
	kernel.setArg(2, cl_vertices);
	kernel.setArg(3, cl_normals);
	kernel.setArg(4, cl_uvs);
	kernel.setArg(5, cl_triangles);
	kernel.setArg(6, cl_octrees);
	kernel.setArg(7, cl_octreeTris);
	kernel.setArg(8, cl_textures);
	kernel.setArg(9, white_point);
	kernel.setArg(10, ambient);
	kernel.setArg(11, window_width);
	kernel.setArg(12, window_height);
	kernel.setArg(13, interval);
	kernel.setArg(14, cl_vbo);
}

void runKernel() {
	// every pixel in the image has its own thread or "work item",
	// so the total amount of work items equals the number of pixels
	std::size_t global_work_size = window_width * window_height;
	std::size_t local_work_size;
	try {
		local_work_size = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
	}
	catch (cl::Error &e) {
		std::cerr << e.what() << ": " << e.err() << std::endl;
		throw e;
	}

	// Ensure the global work size is a multiple of local work size
	if (global_work_size % local_work_size != 0)
		global_work_size = (global_work_size / local_work_size + 1) * local_work_size;

	//Make sure OpenGL is done using the VBOs
	glFinish();

	//this passes in the vector of VBO buffer objects 
	queue.enqueueAcquireGLObjects(&cl_vbos);
	queue.finish();

	// launch the kernel
	queue.enqueueNDRangeKernel(kernel, NULL, global_work_size, local_work_size); // local_work_size
	queue.finish();

	//Release the VBOs so OpenGL can play with them
	queue.enqueueReleaseGLObjects(&cl_vbos);
	queue.finish();
}

void cleanUp() {
	//	delete cpu_output;
}
