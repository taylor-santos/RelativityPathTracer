// OpenCL ray tracing tutorial by Sam Lapere, 2016
// http://raytracey.blogspot.com

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <Windows.h>
#include "gl_interop.h"
#include <CL\cl.hpp>

// TODO
// cleanup()
// check for cl-gl interop

using namespace std;
using namespace cl;

const int object_count = 9;


// OpenCL objects
Device device;
CommandQueue queue;
Kernel kernel;
Context context;
Program program;
Buffer cl_output;
Buffer cl_objects;
Buffer cl_vertices;
Buffer cl_normals;
Buffer cl_uvs;
Buffer cl_triangles;
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

enum objectType{SPHERE, CUBE};

struct Object
{
	cl_double4 M[4];
	cl_double4 InvM[4];
	cl_float3 color;
	enum objectType type;
	cl_float dummy1;
	cl_float dummy2;
};

Object cpu_objects[object_count];
std::vector<cl_float3> vertices;
std::vector<unsigned int> triangles;
std::vector<cl_float2> uvs;
std::vector<cl_float3> normals;

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
	
	ifstream file("opencl_kernel.cl");
	if (!file){
		cout << "\nNo OpenCL file found!" << endl << "Exiting..." << endl;
		system("PAUSE");
		exit(1);
	}
	string source{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};

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
#define double3(x, y, z) {{x, y, z}}
#define double4(x, y, z, w) {{x, y, z, w}}

float sqr_magnitude(const cl_float3 v) {
	return v.x*v.x + v.y*v.y + v.z*v.z;
}
double sqr_magnitude(const cl_double3 v) {
	return v.x*v.x + v.y*v.y + v.z*v.z;
}

float magnitude(const cl_float3 v) {
	return sqrt(sqr_magnitude(v));
}
double magnitude(const cl_double3 v) {
	return sqrt(sqr_magnitude(v));
}

cl_float3 normalize(const cl_float3 v) {
	float m = magnitude(v);
	return float3(v.x / m, v.y / m, v.z / m);
}
cl_double3 normalize(const cl_double3 v) {
	double m = magnitude(v);
	return double3(v.x / m, v.y / m, v.z / m);
}

bool OBJReader(
	std::string path,
	std::vector<cl_float3> &vertices,
	std::vector<unsigned int> &triangles,
	std::vector<cl_float2> &uvs,
	std::vector<cl_float3> &normals
) {
	if (path.substr(path.size() - 4, 4) != ".obj") return false;
	ifstream file(path);
	if (!file) {
		perror("Error opening OBJ file");
		return false;
	}
	vertices.clear();
	triangles.clear();
	uvs.clear();
	normals.clear();
	std::string line;
	int lineno = 0;
	while (std::getline(file, line)) {
		std::istringstream stream(line);
		std::string prefix;
		stream >> prefix;
		if (prefix == "v") {
			cl_float3 vert;
			stream >> vert.x >> vert.y >> vert.z;
			if (stream.fail()) {
				std::cerr << "Error reading OBJ file \"" << path
					<< "\": Invalid syntax on line " << lineno << std::endl;
				return false;
			}
			vertices.push_back(vert);
		}
		else if (prefix == "vt") {
			cl_float2 uv;
			stream >> uv.x >> uv.y;
			if (stream.fail()) {
				std::cerr << "Error reading OBJ file \"" << path
					<< "\": Invalid syntax on line " << lineno << std::endl;
				return false;
			}
			uvs.push_back(uv);
		}
		else if (prefix == "vn") {
			cl_float3 norm;
			stream >> norm.x >> norm.y >> norm.z;
			if (stream.fail()) {
				std::cerr << "Error reading OBJ file \"" << path
					<< "\": Invalid syntax on line " << lineno << std::endl;
				return false;
			}
			norm = normalize(norm);
			normals.push_back(norm);
		}
		else if (prefix == "f") {
			std::string tri;
			for (int i = 0; i < 3; i++) {
				stream >> tri;
				std::istringstream vertstream(tri);
				std::string vert, uv, norm;
				std::getline(vertstream, vert, '/');
				if (!std::getline(vertstream, uv, '/')) {
					uv = "0";
				}
				if (!std::getline(vertstream, norm, '/')) {
					norm = "0";
				}
				triangles.push_back(stoul(vert));
				triangles.push_back(stoul(uv));
				triangles.push_back(stoul(norm));
			}
		}
		lineno++;
	}
	return true;
}

bool calcInvM(Object *object) {
	double A2323 = object->M[2].z * object->M[3].w - object->M[2].w * object->M[3].z;
	double A1323 = object->M[2].y * object->M[3].w - object->M[2].w * object->M[3].y;
	double A1223 = object->M[2].y * object->M[3].z - object->M[2].z * object->M[3].y;
	double A0323 = object->M[2].x * object->M[3].w - object->M[2].w * object->M[3].x;
	double A0223 = object->M[2].x * object->M[3].z - object->M[2].z * object->M[3].x;
	double A0123 = object->M[2].x * object->M[3].y - object->M[2].y * object->M[3].x;
	double A2313 = object->M[1].z * object->M[3].w - object->M[1].w * object->M[3].z;
	double A1313 = object->M[1].y * object->M[3].w - object->M[1].w * object->M[3].y;
	double A1213 = object->M[1].y * object->M[3].z - object->M[1].z * object->M[3].y;
	double A2312 = object->M[1].z * object->M[2].w - object->M[1].w * object->M[2].z;
	double A1312 = object->M[1].y * object->M[2].w - object->M[1].w * object->M[2].y;
	double A1212 = object->M[1].y * object->M[2].z - object->M[1].z * object->M[2].y;
	double A0313 = object->M[1].x * object->M[3].w - object->M[1].w * object->M[3].x;
	double A0213 = object->M[1].x * object->M[3].z - object->M[1].z * object->M[3].x;
	double A0312 = object->M[1].x * object->M[2].w - object->M[1].w * object->M[2].x;
	double A0212 = object->M[1].x * object->M[2].z - object->M[1].z * object->M[2].x;
	double A0113 = object->M[1].x * object->M[3].y - object->M[1].y * object->M[3].x;
	double A0112 = object->M[1].x * object->M[2].y - object->M[1].y * object->M[2].x;

	double det =
		object->M[0].x * (object->M[1].y * A2323 - object->M[1].z * A1323 + object->M[1].w * A1223)
		- object->M[0].y * (object->M[1].x * A2323 - object->M[1].z * A0323 + object->M[1].w * A0223)
		+ object->M[0].z * (object->M[1].x * A1323 - object->M[1].y * A0323 + object->M[1].w * A0123)
		- object->M[0].w * (object->M[1].x * A1223 - object->M[1].y * A0223 + object->M[1].z * A0123);
	if (det == 0.0) {
		return false;
	}
	det = 1 / det;

	object->InvM[0] = double4(
		det * (object->M[1].y * A2323 - object->M[1].z * A1323 + object->M[1].w * A1223),
		det * -(object->M[0].y * A2323 - object->M[0].z * A1323 + object->M[0].w * A1223),
		det * (object->M[0].y * A2313 - object->M[0].z * A1313 + object->M[0].w * A1213),
		det * -(object->M[0].y * A2312 - object->M[0].z * A1312 + object->M[0].w * A1212)
	);
	object->InvM[1] = double4(
		det * -(object->M[1].x * A2323 - object->M[1].z * A0323 + object->M[1].w * A0223),
		det * (object->M[0].x * A2323 - object->M[0].z * A0323 + object->M[0].w * A0223),
		det * -(object->M[0].x * A2313 - object->M[0].z * A0313 + object->M[0].w * A0213),
		det * (object->M[0].x * A2312 - object->M[0].z * A0312 + object->M[0].w * A0212)
	);
	object->InvM[2] = double4(
		det * (object->M[1].x * A1323 - object->M[1].y * A0323 + object->M[1].w * A0123),
		det * -(object->M[0].x * A1323 - object->M[0].y * A0323 + object->M[0].w * A0123),
		det * (object->M[0].x * A1313 - object->M[0].y * A0313 + object->M[0].w * A0113),
		det * -(object->M[0].x * A1312 - object->M[0].y * A0312 + object->M[0].w * A0112)
	);
	object->InvM[3] = double4(
		det * -(object->M[1].x * A1223 - object->M[1].y * A0223 + object->M[1].z * A0123),
		det * (object->M[0].x * A1223 - object->M[0].y * A0223 + object->M[0].z * A0123),
		det * -(object->M[0].x * A1213 - object->M[0].y * A0213 + object->M[0].z * A0113),
		det * (object->M[0].x * A1212 - object->M[0].y * A0212 + object->M[0].z * A0112)
	);
	return true;
}

void TRS(Object *object, cl_double3 translation, double angle, cl_double3 axis, cl_double3 scale) {
	cl_double3 R[3];
	double c = cos(angle);
	double s = sin(angle);
	cl_double3 u = normalize(axis);
	R[0] = double3(c + u.x*u.x*(1 - c), u.x*u.y*(1 - c) - u.z*s, u.x*u.z*(1 - c) + u.y*s);
	R[1] = double3(u.y*u.x*(1 - c) + u.z*s, c + u.y*u.y*(1 - c), u.y*u.z*(1 - c) - u.x*s);
	R[2] = double3(u.z*u.x*(1 - c) - u.y*s, u.z*u.y*(1 - c) + u.x*s, c + u.z*u.z*(1 - c));
	object->M[0] = double4(R[0].x * scale.x, R[0].y * scale.y, R[0].z * scale.z, translation.x);
	object->M[1] = double4(R[1].x * scale.x, R[1].y * scale.y, R[1].z * scale.z, translation.y);
	object->M[2] = double4(R[2].x * scale.x, R[2].y * scale.y, R[2].z * scale.z, translation.z);
	object->M[3] = double4(0, 0, 0, 1);
	calcInvM(object);
}

void initScene(Object* cpu_objects) {
	if (!OBJReader("models/bunny.obj", vertices, triangles, uvs, normals)) {
		exit(EXIT_FAILURE);
	}

	queue.enqueueWriteBuffer(cl_objects, CL_TRUE, 0, object_count * sizeof(Object), cpu_objects);

	// left wall
	cpu_objects[0].color = float3(0.75f, 0.25f, 0.25f);
	cpu_objects[0].type = CUBE;
	TRS(&cpu_objects[0], double3(-6, 0, 10), 0, double3(0, 1, 0), double3(0.1f, 10, 10));

	// right wall
	cpu_objects[1].color = float3(0.25f, 0.25f, 0.75f);
	cpu_objects[1].type = CUBE;
	TRS(&cpu_objects[1], double3(6, 0, 10), 0, double3(0, 1, 0), double3(0.1f, 10, 10));

	// floor
	cpu_objects[2].color = float3(0.9f, 0.8f, 0.7f);
	cpu_objects[2].type = CUBE;
	TRS(&cpu_objects[2], double3(0, -6, 10), 0, double3(0, 1, 0), double3(10, 0.1f, 10));

	// ceiling
	cpu_objects[3].color = float3(0.9f, 0.8f, 0.7f);
	cpu_objects[3].type = CUBE;
	TRS(&cpu_objects[3], double3(0, 6, 10), 0, double3(0, 1, 0), double3(10, 0.1f, 10));

	// back wall
	cpu_objects[4].color = float3(0.9f, 0.8f, 0.7f);
	cpu_objects[4].type = CUBE;
	//TRS(&cpu_objects[4], double3(0, 0, -1), 0, double3(0, 1, 0), double3(10, 10, 0.1f));

	// front wall 
	cpu_objects[5].color = float3(0.9f, 0.8f, 0.7f);
	cpu_objects[5].type = CUBE;
	TRS(&cpu_objects[5], double3(0, 0, 16), 0, double3(0, 1, 0), double3(10, 10, 0.1f));

	// left cube
	cpu_objects[6].color = float3(0.9f, 0.8f, 0.7f);
	cpu_objects[6].type = CUBE;
	TRS(&cpu_objects[6], double3(-3, -4.75f, 12), 0, double3(0, 1, 0), double3(1, 1, 1));

	// right sphere
	cpu_objects[7].color = float3(0.1f, 0.2f, 0.9f);
	cpu_objects[7].type = SPHERE;
	TRS(&cpu_objects[7], double3(0.25f, -0.14f, 1.1f), 0, double3(0, 1, 0), double3(0.05, 0.16f, 0.16f));

	// lightsource
	cpu_objects[8].color = float3(0.0f, 1.0f, 0.0f);
	cpu_objects[8].type = SPHERE;
	TRS(&cpu_objects[8], double3(0, 0.5f, 1), 0, double3(0, 1, 0), double3(0.1f, 0.1f, 0.1f));
}

void initCLKernel(){

	// pick a rendermode
	unsigned int rendermode = 1;

	// Create a kernel (entry point in the OpenCL source program)
	kernel = Kernel(program, "render_kernel");

	// specify OpenCL kernel arguments
	kernel.setArg(0, cl_objects);
	kernel.setArg(1, cl_vertices);
	kernel.setArg(2, cl_triangles);
	kernel.setArg(3, window_width);
	kernel.setArg(4, window_height);
	kernel.setArg(5, object_count);
	kernel.setArg(6, cl_vbo);
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

	TRS(&cpu_objects[6], double3(-3, -4.75f, 12), framenumber/100.0, double3(0, 1, 0), double3(1, 1, 1));

	queue.enqueueWriteBuffer(cl_objects, CL_TRUE, 0, object_count * sizeof(Object), cpu_objects);

	kernel.setArg(0, cl_objects);

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
	initScene(cpu_objects);

	cl_objects = Buffer(context, CL_MEM_READ_ONLY, object_count * sizeof(Object));
	queue.enqueueWriteBuffer(cl_objects, CL_TRUE, 0, object_count * sizeof(Object), cpu_objects);

	cl_vertices = Buffer(context, CL_MEM_READ_ONLY, vertices.size() * sizeof(cl_float3));
	queue.enqueueWriteBuffer(cl_vertices, CL_TRUE, 0, vertices.size() * sizeof(cl_float3), &vertices[0]);

	cl_triangles = Buffer(context, CL_MEM_READ_ONLY, triangles.size() * sizeof(unsigned int));
	queue.enqueueWriteBuffer(cl_vertices, CL_TRUE, 0, triangles.size() * sizeof(unsigned int), &triangles[0]);

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
