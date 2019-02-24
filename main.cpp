// OpenCL ray tracing tutorial by Sam Lapere, 2016
// http://raytracey.blogspot.com

#include <iostream>
#include <fstream>
#include <vector>
#include <Windows.h>
#include "gl_interop.h"
#include <CL\cl.hpp>

// TODO
// cleanup()
// check for cl-gl interop

using namespace std;
using namespace cl;

const int sphere_count = 9;


// OpenCL objects
Device device;
CommandQueue queue;
Kernel kernel;
Context context;
Program program;
Buffer cl_output;
Buffer cl_spheres;
BufferGL cl_vbo;
vector<Memory> cl_vbos;

// image buffer (not needed with real-time viewport)
cl_float4* cpu_output;
cl_int err;
unsigned int framenumber = 0;


// padding with dummy variables are required for memory alignment
// float3 is considered as float4 by OpenCL
// alignment can also be enforced by using __attribute__ ((aligned (16)));
// see https://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/attributes-variables.html

struct Sphere
{
	cl_float4 M[4];
	cl_float4 InvM[4];
	cl_float3 color;
	cl_float3 emission;
};

Sphere cpu_spheres[sphere_count];

void pickPlatform(Platform& platform, const vector<Platform>& platforms){

	if (platforms.size() == 1) platform = platforms[0];
	else{
		int input = 0;
		cout << "\nChoose an OpenCL platform: ";
		cin >> input;

		// handle incorrect user input
		while (input < 1 || input > platforms.size()){
			cin.clear(); //clear errors/bad flags on cin
			cin.ignore(cin.rdbuf()->in_avail(), '\n'); // ignores exact number of chars in cin buffer
			cout << "No such option. Choose an OpenCL platform: ";
			cin >> input;
		}
		platform = platforms[input - 1];
	}
}

void pickDevice(Device& device, const vector<Device>& devices){

	if (devices.size() == 1) device = devices[0];
	else{
		int input = 0;
		cout << "\nChoose an OpenCL device: ";
		cin >> input;

		// handle incorrect user input
		while (input < 1 || input > devices.size()){
			cin.clear(); //clear errors/bad flags on cin
			cin.ignore(cin.rdbuf()->in_avail(), '\n'); // ignores exact number of chars in cin buffer
			cout << "No such option. Choose an OpenCL device: ";
			cin >> input;
		}
		device = devices[input - 1];
	}
}

void printErrorLog(const Program& program, const Device& device){

	// Get the error log and print to console
	string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
	cerr << "Build log:" << std::endl << buildlog << std::endl;

	// Print the error log to a file
	FILE *log = fopen("errorlog.txt", "w");
	fprintf(log, "%s\n", buildlog);
	cout << "Error log saved in 'errorlog.txt'" << endl;
	system("PAUSE");
	exit(1);
}

void initOpenCL()
{
	// Get all available OpenCL platforms (e.g. AMD OpenCL, Nvidia CUDA, Intel OpenCL)
	vector<Platform> platforms;
	Platform::get(&platforms);
	cout << "Available OpenCL platforms : " << endl << endl;
	for (int i = 0; i < platforms.size(); i++)
		cout << "\t" << i + 1 << ": " << platforms[i].getInfo<CL_PLATFORM_NAME>() << endl;

	cout << endl << "WARNING: " << endl << endl;
	cout << "OpenCL-OpenGL interoperability is only tested " << endl;
	cout << "on discrete GPUs from Nvidia and AMD" << endl;
	cout << "Other devices (such as Intel integrated GPUs) may fail" << endl << endl;

	// Pick one platform
	Platform platform;
	pickPlatform(platform, platforms);
	cout << "\nUsing OpenCL platform: \t" << platform.getInfo<CL_PLATFORM_NAME>() << endl;

	// Get available OpenCL devices on platform
	vector<Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

	cout << "Available OpenCL devices on this platform: " << endl << endl;
	for (int i = 0; i < devices.size(); i++){
		cout << "\t" << i + 1 << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << endl;
		cout << "\t\tMax compute units: " << devices[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << endl;
		cout << "\t\tMax work group size: " << devices[i].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << endl << endl;
	}


	// Pick one device
	//Device device;
	pickDevice(device, devices);
	cout << "\nUsing OpenCL device: \t" << device.getInfo<CL_DEVICE_NAME>() << endl;

	// Create an OpenCL context on that device.
	// Windows specific OpenCL-OpenGL interop
	cl_context_properties properties[] =
	{
		CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
		CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
		CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 
		0
	};

	context = Context(device, properties);

	// Create a command queue
	queue = CommandQueue(context, device);

	
	// Convert the OpenCL source code to a string// Convert the OpenCL source code to a string
	string source;
	ifstream file("opencl_kernel.cl");
	if (!file){
		cout << "\nNo OpenCL file found!" << endl << "Exiting..." << endl;
		system("PAUSE");
		exit(1);
	}
	while (!file.eof()){
		char line[256];
		file.getline(line, 255);
		source += line;
	}

	const char* kernel_source = source.c_str();

	// Create an OpenCL program with source
	program = Program(context, kernel_source);

	// Build the program for the selected device 
	cl_int result = program.build({ device }); // "-cl-fast-relaxed-math"
	if (result) cout << "Error during compilation OpenCL code!!!\n (" << result << ")" << endl;
	if (result == CL_BUILD_PROGRAM_FAILURE) printErrorLog(program, device);
}

#define float3(x, y, z) {{x, y, z}}  // macro to replace ugly initializer braces
#define float4(x, y, z, w) {{x, y, z, w}}

bool calcInvM(Sphere *sphere) {
	float A2323 = sphere->M[2].z * sphere->M[3].w - sphere->M[2].w * sphere->M[3].z;
	float A1323 = sphere->M[2].y * sphere->M[3].w - sphere->M[2].w * sphere->M[3].y;
	float A1223 = sphere->M[2].y * sphere->M[3].z - sphere->M[2].z * sphere->M[3].y;
	float A0323 = sphere->M[2].x * sphere->M[3].w - sphere->M[2].w * sphere->M[3].x;
	float A0223 = sphere->M[2].x * sphere->M[3].z - sphere->M[2].z * sphere->M[3].x;
	float A0123 = sphere->M[2].x * sphere->M[3].y - sphere->M[2].y * sphere->M[3].x;
	float A2313 = sphere->M[1].z * sphere->M[3].w - sphere->M[1].w * sphere->M[3].z;
	float A1313 = sphere->M[1].y * sphere->M[3].w - sphere->M[1].w * sphere->M[3].y;
	float A1213 = sphere->M[1].y * sphere->M[3].z - sphere->M[1].z * sphere->M[3].y;
	float A2312 = sphere->M[1].z * sphere->M[2].w - sphere->M[1].w * sphere->M[2].z;
	float A1312 = sphere->M[1].y * sphere->M[2].w - sphere->M[1].w * sphere->M[2].y;
	float A1212 = sphere->M[1].y * sphere->M[2].z - sphere->M[1].z * sphere->M[2].y;
	float A0313 = sphere->M[1].x * sphere->M[3].w - sphere->M[1].w * sphere->M[3].x;
	float A0213 = sphere->M[1].x * sphere->M[3].z - sphere->M[1].z * sphere->M[3].x;
	float A0312 = sphere->M[1].x * sphere->M[2].w - sphere->M[1].w * sphere->M[2].x;
	float A0212 = sphere->M[1].x * sphere->M[2].z - sphere->M[1].z * sphere->M[2].x;
	float A0113 = sphere->M[1].x * sphere->M[3].y - sphere->M[1].y * sphere->M[3].x;
	float A0112 = sphere->M[1].x * sphere->M[2].y - sphere->M[1].y * sphere->M[2].x;

	float det = 
		  sphere->M[0].x * (sphere->M[1].y * A2323 - sphere->M[1].z * A1323 + sphere->M[1].w * A1223)
		- sphere->M[0].y * (sphere->M[1].x * A2323 - sphere->M[1].z * A0323 + sphere->M[1].w * A0223)
		+ sphere->M[0].z * (sphere->M[1].x * A1323 - sphere->M[1].y * A0323 + sphere->M[1].w * A0123)
		- sphere->M[0].w * (sphere->M[1].x * A1223 - sphere->M[1].y * A0223 + sphere->M[1].z * A0123);
	if (det == 0.0f) {
		return false;
	}
	det = 1 / det;

	sphere->InvM[0] = float4(
		det * (sphere->M[1].y * A2323 - sphere->M[1].z * A1323 + sphere->M[1].w * A1223),
		det * -(sphere->M[0].y * A2323 - sphere->M[0].z * A1323 + sphere->M[0].w * A1223),
		det * (sphere->M[0].y * A2313 - sphere->M[0].z * A1313 + sphere->M[0].w * A1213),
		det * -(sphere->M[0].y * A2312 - sphere->M[0].z * A1312 + sphere->M[0].w * A1212)
	);
	sphere->InvM[1] = float4(
		det * -(sphere->M[1].x * A2323 - sphere->M[1].z * A0323 + sphere->M[1].w * A0223),
		det * (sphere->M[0].x * A2323 - sphere->M[0].z * A0323 + sphere->M[0].w * A0223),
		det * -(sphere->M[0].x * A2313 - sphere->M[0].z * A0313 + sphere->M[0].w * A0213),
		det * (sphere->M[0].x * A2312 - sphere->M[0].z * A0312 + sphere->M[0].w * A0212)
	);
	sphere->InvM[2] = float4(
		det * (sphere->M[1].x * A1323 - sphere->M[1].y * A0323 + sphere->M[1].w * A0123),
		det * -(sphere->M[0].x * A1323 - sphere->M[0].y * A0323 + sphere->M[0].w * A0123),
		det * (sphere->M[0].x * A1313 - sphere->M[0].y * A0313 + sphere->M[0].w * A0113),
		det * -(sphere->M[0].x * A1312 - sphere->M[0].y * A0312 + sphere->M[0].w * A0112)
	);
	sphere->InvM[3] = float4(
		det * -(sphere->M[1].x * A1223 - sphere->M[1].y * A0223 + sphere->M[1].z * A0123),
		det * (sphere->M[0].x * A1223 - sphere->M[0].y * A0223 + sphere->M[0].z * A0123),
		det * -(sphere->M[0].x * A1213 - sphere->M[0].y * A0213 + sphere->M[0].z * A0113),
		det * (sphere->M[0].x * A1212 - sphere->M[0].y * A0212 + sphere->M[0].z * A0112)
	);
	return true;
}

float sqr_magnitude(const cl_float3 v) {
	return v.x*v.x + v.y*v.y + v.z*v.z;
}

float magnitude(const cl_float3 v) {
	return sqrt(sqr_magnitude(v));
}

cl_float3 normalize(const cl_float3 v) {
	float m = magnitude(v);
	return float3(v.x / m, v.y / m, v.z / m);
}

void TRS(Sphere *sphere, cl_float3 translation, float angle, cl_float3 axis, cl_float3 scale) {
	cl_float3 R[3];
	float c = cos(angle);
	float s = sin(angle);
	cl_float3 u = normalize(axis);
	R[0] = float3(c + u.x*u.x*(1 - c), u.x*u.y*(1 - c) - u.z*s, u.x*u.z*(1 - c) + u.y*s);
	R[1] = float3(u.y*u.x*(1 - c) + u.z*s, c + u.y*u.y*(1 - c), u.y*u.z*(1 - c) - u.x*s);
	R[2] = float3(u.z*u.x*(1 - c) - u.y*s, u.z*u.y*(1 - c) + u.x*s, c + u.z*u.z*(1 - c));
	sphere->M[0] = float4(R[0].x * scale.x, R[0].y * scale.y, R[0].z * scale.z, translation.x);
	sphere->M[1] = float4(R[1].x * scale.x, R[1].y * scale.y, R[1].z * scale.z, translation.y);
	sphere->M[2] = float4(R[2].x * scale.x, R[2].y * scale.y, R[2].z * scale.z, translation.z);
	sphere->M[3] = float4(0, 0, 0, 1);
	calcInvM(sphere);
}

void initScene(Sphere* cpu_spheres){

	// left wall
	cpu_spheres[0].color = float3(0.75f, 0.25f, 0.25f);
	cpu_spheres[0].emission = float3(0.0f, 0.0f, 0.0f);
	TRS(&cpu_spheres[0], float3(-6, 0, 10), 0, float3(0, 1, 0), float3(0.1f, 10, 10));

	// right wall
	cpu_spheres[1].color = float3(0.25f, 0.25f, 0.75f);
	cpu_spheres[1].emission = float3(0.0f, 0.0f, 0.0f);
	TRS(&cpu_spheres[1], float3(6, 0, 10), 0, float3(0, 1, 0), float3(0.1f, 10, 10));

	// floor
	cpu_spheres[2].color = float3(0.9f, 0.8f, 0.7f);
	cpu_spheres[2].emission = float3(0.0f, 0.0f, 0.0f);
	TRS(&cpu_spheres[2], float3(0, -6, 10), 0, float3(0, 1, 0), float3(10, 0.1f, 10));

	// ceiling
	cpu_spheres[3].color = float3(0.9f, 0.8f, 0.7f);
	cpu_spheres[3].emission = float3(0.0f, 0.0f, 0.0f);
	TRS(&cpu_spheres[3], float3(0, 6, 10), 0, float3(0, 1, 0), float3(10, 0.1f, 10));

	// back wall
	cpu_spheres[4].color = float3(0.9f, 0.8f, 0.7f);
	cpu_spheres[4].emission = float3(0.0f, 0.0f, 0.0f);
	TRS(&cpu_spheres[4], float3(0, 0, -1), 0, float3(0, 1, 0), float3(10, 10, 0.1f));

	// front wall 
	cpu_spheres[5].color = float3(0.9f, 0.8f, 0.7f);
	cpu_spheres[5].emission = float3(0.0f, 0.0f, 0.0f);
	TRS(&cpu_spheres[5], float3(0, 0, 16), 0, float3(0, 1, 0), float3(10, 10, 0.1f));

	// left sphere
	cpu_spheres[6].color = float3(0.9f, 0.8f, 0.7f);
	cpu_spheres[6].emission = float3(0.0f, 0.0f, 0.0f);
	TRS(&cpu_spheres[6], float3(-3, -4.75f, 12), 0, float3(0, 1, 0), float3(1, 1, 1));

	// right sphere
	cpu_spheres[7].color = float3(0.9f, 0.8f, 0.7f);
	cpu_spheres[7].emission = float3(0.0f, 0.0f, 0.0f);
	TRS(&cpu_spheres[7], float3(0.25f, -0.14f, 1.1f), 0, float3(0, 1, 0), float3(0.16f, 0.16f, 0.16f));

	// lightsource
	cpu_spheres[8].color = float3(0.0f, 0.0f, 0.0f);
	cpu_spheres[8].emission = float3(9.0f, 8.0f, 6.0f);
	TRS(&cpu_spheres[8], float3(0, 0.5f, 1), 0, float3(0, 1, 0), float3(0.1f, 0.1f, 0.1f));
}

void initCLKernel(){

	// pick a rendermode
	unsigned int rendermode = 1;

	// Create a kernel (entry point in the OpenCL source program)
	kernel = Kernel(program, "render_kernel");

	// specify OpenCL kernel arguments
	//kernel.setArg(0, cl_output);
	kernel.setArg(0, cl_spheres);
	kernel.setArg(1, window_width);
	kernel.setArg(2, window_height);
	kernel.setArg(3, sphere_count);
	kernel.setArg(4, cl_vbo);
	kernel.setArg(5, framenumber);
}

void runKernel(){
	// every pixel in the image has its own thread or "work item",
	// so the total amount of work items equals the number of pixels
	std::size_t global_work_size = window_width * window_height;
	std::size_t local_work_size = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);;

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


// hash function to calculate new seed for each frame
// see http://www.reedbeta.com/blog/2013/01/12/quick-and-easy-gpu-random-numbers-in-d3d11/
unsigned int WangHash(unsigned int a) {
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}


void render(){
	framenumber++;

	int new_window_width = glutGet(GLUT_WINDOW_WIDTH),
	    new_window_height = glutGet(GLUT_WINDOW_HEIGHT);
	if (new_window_width != window_width || new_window_height != window_height) {
		window_width = new_window_width;
		window_height = new_window_height;

		glClearColor(0.0, 0.0, 0.0, 1.0);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluOrtho2D(0.0, window_width, 0.0, window_height);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		unsigned int size = window_width * window_height * sizeof(cl_float3);
		glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
		cl_vbo = BufferGL(context, CL_MEM_WRITE_ONLY, vbo);
		cl_vbos[0] = cl_vbo;
		initCLKernel();
	}

	queue.enqueueWriteBuffer(cl_spheres, CL_TRUE, 0, sphere_count * sizeof(Sphere), cpu_spheres);

	kernel.setArg(0, cl_spheres);
	kernel.setArg(5, WangHash(framenumber));

	runKernel();

	drawGL();
}

void cleanUp(){
//	delete cpu_output;
}

void main(int argc, char** argv){

	// initialise OpenGL (GLEW and GLUT window + callback functions)
	initGL(argc, argv);
	cout << "OpenGL initialized \n";

	// initialise OpenCL
	initOpenCL();

	// create vertex buffer object
	createVBO(&vbo);

	// call Timer():
	Timer(0);
	
	//make sure OpenGL is finished before we proceed
	glFinish();

	// initialise scene
	initScene(cpu_spheres);

	cl_spheres = Buffer(context, CL_MEM_READ_ONLY, sphere_count * sizeof(Sphere));
	queue.enqueueWriteBuffer(cl_spheres, CL_TRUE, 0, sphere_count * sizeof(Sphere), cpu_spheres);

	// create OpenCL buffer from OpenGL vertex buffer object
	cl_vbo = BufferGL(context, CL_MEM_WRITE_ONLY, vbo);
	cl_vbos.push_back(cl_vbo);

	// intitialise the kernel
	initCLKernel();

	// start rendering continuously
	glutMainLoop();

	// release memory
	cleanUp();

	system("PAUSE");
}
