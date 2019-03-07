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
#include <chrono>

// TODO
// cleanup()
// check for cl-gl interop

using namespace std;
using namespace cl;

const int object_count = 9;

std::chrono::time_point<std::chrono::high_resolution_clock> clock_start, clock_end;

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
Buffer cl_octrees;
Buffer cl_octreeTris;
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

struct Octree
{
	cl_double3 min;
	cl_double3 max;
	int trisIndex,
		trisCount;
	int children[8]  = { -1, -1, -1, -1, -1, -1, -1, -1 };
	int neighbors[6] = { -1, -1, -1, -1, -1, -1 };
};

struct Mesh
{
	std::vector<cl_double3> vertices;
	std::vector<unsigned int> triangles;
	std::vector<cl_double2> uvs;
	std::vector<cl_double3> normals;
	std::vector<Octree> octree;
	std::vector<int> octreeTris;

	void GenerateOctree();
};

Mesh theMesh;

void pickPlatform(Platform& platform, const vector<Platform>& platforms) {

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

cl_double3 operator+(const cl_double3 v1, const cl_double3 v2) {
	return double3(
		v1.x + v2.x,
		v1.y + v2.y,
		v1.z + v2.z
	);
}

cl_double3 operator-(const cl_double3 v1, const cl_double3 v2) {
	return double3(
		v1.x - v2.x,
		v1.y - v2.y,
		v1.z - v2.z
	);
}

cl_double3 operator*(const cl_double3 v, const double c) {
	return double3(
		v.x * c,
		v.y * c,
		v.z * c
	);
}

cl_double3 operator/(const cl_double3 v, const double c) {
	return double3(
		v.x / c,
		v.y / c,
		v.z / c
	);
}

double dot(const cl_double3 a, const cl_double3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

cl_double3 cross(const cl_double3 a, const cl_double3 b) {
	return double3(
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	);
}

cl_double3 elementwise_min(const cl_double3 a, const cl_double3 b) {
	return double3(
		min(a.x, b.x),
		min(a.y, b.y),
		min(a.z, b.z)
	);
}

cl_double3 elementwise_max(const cl_double3 a, const cl_double3 b) {
	return double3(
		max(a.x, b.x),
		max(a.y, b.y),
		max(a.z, b.z)
	);
}

bool AABBTriangleIntersection(Mesh const& mesh, int octreeIndex, int triIndex) {
	const cl_double3 A = mesh.vertices[mesh.triangles[9 * triIndex + 3 * 0]];
	const cl_double3 B = mesh.vertices[mesh.triangles[9 * triIndex + 3 * 1]];
	const cl_double3 C = mesh.vertices[mesh.triangles[9 * triIndex + 3 * 2]];
	cl_double3 min = mesh.octree[octreeIndex].min;
	cl_double3 max = mesh.octree[octreeIndex].max;
	cl_double3 center = (min + max) / 2;
	cl_double3 extents = max - min;
	cl_double3 half_extents = extents / 2;

	cl_double3 offsetA = A - center;
	cl_double3 offsetB = B - center;
	cl_double3 offsetC = C - center;

	cl_double3 ba = offsetB - offsetA;
	cl_double3 cb = offsetC - offsetB;

	double x_ba_abs = abs(ba.x);
	double y_ba_abs = abs(ba.y);
	double z_ba_abs = abs(ba.z);
	{
		double min = ba.z * offsetA.y - ba.y * offsetA.z;
		double max = ba.z * offsetC.y - ba.y * offsetC.z;
		if (min > max) {
			double temp = min;
			min = max;
			max = temp;
		}
		double rad = z_ba_abs * extents.y + y_ba_abs * extents.z;
		if (min > rad || max < -rad) return false;
	}
	{
		double min = -ba.z * offsetA.x + ba.x * offsetA.z;
		double max = -ba.z * offsetC.x + ba.x * offsetC.z;
		if (min > max) {
			double temp = min;
			min = max;
			max = temp;
		}
		double rad = z_ba_abs * extents.x + x_ba_abs * extents.z;
		if (min > rad || max < -rad) return false;
	}
	{
		double min = ba.y * offsetB.x - ba.x * offsetB.y;
		double max = ba.y * offsetC.x - ba.x * offsetC.y;
		if (min > max) {
			double temp = min;
			min = max;
			max = temp;
		}
		double rad = y_ba_abs * extents.x + x_ba_abs * extents.y;
		if (min > rad || max < -rad) return false;
	}
	double x_cb_abs = abs(cb.x);
	double y_cb_abs = abs(cb.y);
	double z_cb_abs = abs(cb.z);
	{
		double min = cb.z * offsetA.y - cb.y * offsetA.z,
			max = cb.z * offsetC.y - cb.y * offsetC.z;
		if (min > max) {
			double temp = min;
			min = max;
			max = temp;
		}
		double rad = z_cb_abs * extents.y + y_cb_abs * extents.z;
		if (min > rad || max < -rad) return false;
	}
	{
		double min = -cb.z * offsetA.x + cb.x * offsetA.z,
			max = -cb.z * offsetC.x + cb.x * offsetC.z;
		if (min > max) {
			double temp = min;
			min = max;
			max = temp;
		}
		double rad = z_cb_abs * extents.x + x_cb_abs * extents.z;
		if (min > rad || max < -rad) return false;
	}
	{
		double min = cb.y * offsetA.x - cb.x * offsetA.y,
			max = cb.y * offsetB.x - cb.x * offsetB.y;
		if (min > max) {
			double temp = min;
			min = max;
			max = temp;
		}
		double rad = y_cb_abs * extents.x + x_cb_abs * extents.y;
		if (min > rad || max < -rad) return false;
	}
	cl_double3 ac = offsetA - offsetC;
	double x_ac_abs = abs(ac.x);
	double y_ac_abs = abs(ac.y);
	double z_ac_abs = abs(ac.z);
	{
		double min = ac.z * offsetA.y - ac.y * offsetA.z,
			max = ac.z * offsetB.y - ac.y * offsetB.z;
		if (min > max) {
			double temp = min;
			min = max;
			max = temp;
		}
		double rad = z_ac_abs * extents.y + y_ac_abs * extents.z;
		if (min > rad || max < -rad) return false;
	}
	{
		double min = -ac.z * offsetA.x + ac.x * offsetA.z,
			max = -ac.z * offsetB.x + ac.x * offsetB.z;
		if (min > max) {
			double temp = min;
			min = max;
			max = temp;
		}
		double rad = z_ac_abs * extents.x + x_ac_abs * extents.z;
		if (min > rad || max < -rad) return false;
	}
	{
		double min = ac.y * offsetB.x - ac.x * offsetB.y,
			max = ac.y * offsetC.x - ac.x * offsetC.y;
		if (min > max) {
			double temp = min;
			min = max;
			max = temp;
		}
		double rad = y_ac_abs * extents.x + x_ac_abs * extents.y;
		if (min > rad || max < -rad) return false;
	}
	{
		cl_double3 normal = cross(ba, cb);
		cl_double3 min, max;
		if (normal.x > 0) {
			min.x = -extents.x - offsetA.x;
			max.x = extents.x - offsetA.x;
		}
		else {
			min.x = extents.x - offsetA.x;
			max.x = -extents.x - offsetA.x;
		}
		if (normal.y > 0) {
			min.y = -extents.y - offsetA.y;
			max.y = extents.y - offsetA.y;
		}
		else {
			min.y = extents.y - offsetA.y;
			max.y = -extents.y - offsetA.y;
		}
		if (normal.z > 0) {
			min.z = -extents.z - offsetA.z;
			max.z = extents.z - offsetA.z;
		}
		else {
			min.z = extents.z - offsetA.z;
			max.z = -extents.z - offsetA.z;
		}
		if (dot(normal, min) > 0) return false;
		if (dot(normal, max) < 0) return false;
	}
	{
		cl_double3 min = elementwise_min(elementwise_min(offsetA, offsetB), offsetC);
		cl_double3 max = elementwise_max(elementwise_max(offsetA, offsetB), offsetC);
		if (min.x > extents.x || max.x < -extents.x) return false;
		if (min.y > extents.y || max.y < -extents.y) return false;
		if (min.z > extents.z || max.z < -extents.z) return false;
	}
	return true;
}

void Subdivide(Mesh &mesh, int octreeIndex, int minTris, int depth) {
	if (depth <= 0/* || mesh.octree[octreeIndex].trisCount <= minTris*/) return;
	cl_double3 extents = mesh.octree[octreeIndex].max - mesh.octree[octreeIndex].min;
	cl_double3 half_extents = extents / 2;
	cl_double3 ex = double3(half_extents.x, 0, 0);
	cl_double3 ey = double3(0, half_extents.y, 0);
	cl_double3 ez = double3(0, 0, half_extents.z);
	for (int x = 0; x < 2; x++) {
		for (int y = 0; y < 2; y++) {
			for (int z = 0; z < 2; z++) {
				Octree child;
				child.min = mesh.octree[octreeIndex].min + ex * x + ey * y + ez * z;
				child.max = child.min + half_extents;
				int trisStart = mesh.octree[octreeIndex].trisIndex;
				int trisCount = mesh.octree[octreeIndex].trisCount;
				child.trisIndex = mesh.octreeTris.size();
				child.trisCount = 0;
				
				mesh.octree[octreeIndex].children[z + 2 * y + 4 * x] = mesh.octree.size();
				mesh.octree.push_back(child);
				for (int tri = trisStart; tri < trisStart + trisCount; tri++) {
					int triIndex = mesh.octreeTris[tri];
					if (AABBTriangleIntersection(mesh, mesh.octree.size()-1, triIndex)) {
						mesh.octreeTris.push_back(triIndex);
						mesh.octree[mesh.octree.size()-1].trisCount++;
					}
				}
			}
		}
	}
	mesh.octree[mesh.octree[octreeIndex].children[0]].neighbors[0] = mesh.octree[octreeIndex].neighbors[0];
	mesh.octree[mesh.octree[octreeIndex].children[0]].neighbors[1] = mesh.octree[octreeIndex].children[1];
	mesh.octree[mesh.octree[octreeIndex].children[0]].neighbors[2] = mesh.octree[octreeIndex].neighbors[1];
	mesh.octree[mesh.octree[octreeIndex].children[0]].neighbors[3] = mesh.octree[octreeIndex].children[4];
	mesh.octree[mesh.octree[octreeIndex].children[0]].neighbors[4] = mesh.octree[octreeIndex].neighbors[5];
	mesh.octree[mesh.octree[octreeIndex].children[0]].neighbors[5] = mesh.octree[octreeIndex].children[2];
	mesh.octree[mesh.octree[octreeIndex].children[1]].neighbors[0] = mesh.octree[octreeIndex].children[0];
	mesh.octree[mesh.octree[octreeIndex].children[1]].neighbors[1] = mesh.octree[octreeIndex].neighbors[1];
	mesh.octree[mesh.octree[octreeIndex].children[1]].neighbors[2] = mesh.octree[octreeIndex].neighbors[2];
	mesh.octree[mesh.octree[octreeIndex].children[1]].neighbors[3] = mesh.octree[octreeIndex].children[5];
	mesh.octree[mesh.octree[octreeIndex].children[1]].neighbors[4] = mesh.octree[octreeIndex].neighbors[5];
	mesh.octree[mesh.octree[octreeIndex].children[1]].neighbors[5] = mesh.octree[octreeIndex].children[3];
	mesh.octree[mesh.octree[octreeIndex].children[2]].neighbors[0] = mesh.octree[octreeIndex].neighbors[0];
	mesh.octree[mesh.octree[octreeIndex].children[2]].neighbors[1] = mesh.octree[octreeIndex].children[3];
	mesh.octree[mesh.octree[octreeIndex].children[2]].neighbors[2] = mesh.octree[octreeIndex].neighbors[2];
	mesh.octree[mesh.octree[octreeIndex].children[2]].neighbors[3] = mesh.octree[octreeIndex].children[6];
	mesh.octree[mesh.octree[octreeIndex].children[2]].neighbors[4] = mesh.octree[octreeIndex].children[0];
	mesh.octree[mesh.octree[octreeIndex].children[2]].neighbors[5] = mesh.octree[octreeIndex].neighbors[4];
	mesh.octree[mesh.octree[octreeIndex].children[3]].neighbors[0] = mesh.octree[octreeIndex].children[2];
	mesh.octree[mesh.octree[octreeIndex].children[3]].neighbors[1] = mesh.octree[octreeIndex].neighbors[1];
	mesh.octree[mesh.octree[octreeIndex].children[3]].neighbors[2] = mesh.octree[octreeIndex].neighbors[2];
	mesh.octree[mesh.octree[octreeIndex].children[3]].neighbors[3] = mesh.octree[octreeIndex].children[7];
	mesh.octree[mesh.octree[octreeIndex].children[3]].neighbors[4] = mesh.octree[octreeIndex].children[1];
	mesh.octree[mesh.octree[octreeIndex].children[3]].neighbors[5] = mesh.octree[octreeIndex].neighbors[4];
	mesh.octree[mesh.octree[octreeIndex].children[4]].neighbors[0] = mesh.octree[octreeIndex].neighbors[0];
	mesh.octree[mesh.octree[octreeIndex].children[4]].neighbors[1] = mesh.octree[octreeIndex].children[5];
	mesh.octree[mesh.octree[octreeIndex].children[4]].neighbors[2] = mesh.octree[octreeIndex].children[0];
	mesh.octree[mesh.octree[octreeIndex].children[4]].neighbors[3] = mesh.octree[octreeIndex].neighbors[3];
	mesh.octree[mesh.octree[octreeIndex].children[4]].neighbors[4] = mesh.octree[octreeIndex].neighbors[5];
	mesh.octree[mesh.octree[octreeIndex].children[4]].neighbors[5] = mesh.octree[octreeIndex].children[6];
	mesh.octree[mesh.octree[octreeIndex].children[5]].neighbors[0] = mesh.octree[octreeIndex].children[4];
	mesh.octree[mesh.octree[octreeIndex].children[5]].neighbors[1] = mesh.octree[octreeIndex].neighbors[1];
	mesh.octree[mesh.octree[octreeIndex].children[5]].neighbors[2] = mesh.octree[octreeIndex].children[1];
	mesh.octree[mesh.octree[octreeIndex].children[5]].neighbors[3] = mesh.octree[octreeIndex].neighbors[3];
	mesh.octree[mesh.octree[octreeIndex].children[5]].neighbors[4] = mesh.octree[octreeIndex].neighbors[5];
	mesh.octree[mesh.octree[octreeIndex].children[5]].neighbors[5] = mesh.octree[octreeIndex].children[7];
	mesh.octree[mesh.octree[octreeIndex].children[6]].neighbors[0] = mesh.octree[octreeIndex].neighbors[0];
	mesh.octree[mesh.octree[octreeIndex].children[6]].neighbors[1] = mesh.octree[octreeIndex].children[7];
	mesh.octree[mesh.octree[octreeIndex].children[6]].neighbors[2] = mesh.octree[octreeIndex].children[2];
	mesh.octree[mesh.octree[octreeIndex].children[6]].neighbors[3] = mesh.octree[octreeIndex].neighbors[3];
	mesh.octree[mesh.octree[octreeIndex].children[6]].neighbors[4] = mesh.octree[octreeIndex].children[4];
	mesh.octree[mesh.octree[octreeIndex].children[6]].neighbors[5] = mesh.octree[octreeIndex].neighbors[4];
	mesh.octree[mesh.octree[octreeIndex].children[7]].neighbors[0] = mesh.octree[octreeIndex].children[6];
	mesh.octree[mesh.octree[octreeIndex].children[7]].neighbors[1] = mesh.octree[octreeIndex].neighbors[1];
	mesh.octree[mesh.octree[octreeIndex].children[7]].neighbors[2] = mesh.octree[octreeIndex].children[3];
	mesh.octree[mesh.octree[octreeIndex].children[7]].neighbors[3] = mesh.octree[octreeIndex].neighbors[3];
	mesh.octree[mesh.octree[octreeIndex].children[7]].neighbors[4] = mesh.octree[octreeIndex].children[5];
	mesh.octree[mesh.octree[octreeIndex].children[7]].neighbors[5] = mesh.octree[octreeIndex].neighbors[4];
	for (int i = 0; i < 8; i++) {
		Subdivide(mesh, mesh.octree[octreeIndex].children[i], minTris, depth - 1);
	}
}

void Mesh::GenerateOctree() {
	Octree newOctree;
	octreeTris.clear();
	newOctree.trisCount = 0;
	newOctree.trisIndex = 0;
	newOctree.min = vertices[triangles[0]];
	newOctree.max = vertices[triangles[0]];
	for (int i = 1; i < triangles.size() / 3; i++) {
		cl_double3 vert = vertices[triangles[3 * i]];
		newOctree.min = elementwise_min(newOctree.min, vert);
		newOctree.max = elementwise_max(newOctree.max, vert);
	}
	for (int i = 0; i < triangles.size() / 9; i++) {
		octreeTris.push_back(i);
		newOctree.trisCount++;
	}
	octree.push_back(newOctree);
	Subdivide(*this, 0, 20, 4);
}

bool OBJReader(std::string path, Mesh &mesh) {
	if (path.substr(path.size() - 4, 4) != ".obj") return false;
	ifstream file(path);
	if (!file) {
		perror("Error opening OBJ file");
		return false;
	}
	mesh.vertices.clear();
	mesh.triangles.clear();
	mesh.uvs.clear();
	mesh.normals.clear();
	std::string line;
	int lineno = 0;
	while (std::getline(file, line)) {
		std::istringstream stream(line);
		std::string prefix;
		stream >> prefix;
		if (prefix == "v") {
			cl_double3 vert;
			stream >> vert.x >> vert.y >> vert.z;
			if (stream.fail()) {
				std::cerr << "Error reading OBJ file \"" << path
					<< "\": Invalid syntax on line " << lineno << std::endl;
				return false;
			}
			mesh.vertices.push_back(vert);
		}
		else if (prefix == "vt") {
			cl_double2 uv;
			stream >> uv.x >> uv.y;
			if (stream.fail()) {
				std::cerr << "Error reading OBJ file \"" << path
					<< "\": Invalid syntax on line " << lineno << std::endl;
				return false;
			}
			mesh.uvs.push_back(uv);
		}
		else if (prefix == "vn") {
			cl_double3 norm;
			stream >> norm.x >> norm.y >> norm.z;
			if (stream.fail()) {
				std::cerr << "Error reading OBJ file \"" << path
					<< "\": Invalid syntax on line " << lineno << std::endl;
				return false;
			}
			norm = normalize(norm);
			mesh.normals.push_back(norm);
		}
		else if (prefix == "f") {
			std::string tri;
			bool needsNormal = false;
			for (int i = 0; i < 3; i++) {
				stream >> tri;
				std::istringstream vertstream(tri);
				std::string vert, uv, norm;
				std::getline(vertstream, vert, '/');
				if (!std::getline(vertstream, uv, '/')) {
					uv = "1";
				}
				if (!std::getline(vertstream, norm, '/')) {
					norm = std::to_string(mesh.normals.size()+1);
					needsNormal = true;
				}
				mesh.triangles.push_back(stoul(vert)-1);
				mesh.triangles.push_back(stoul(uv)-1);
				mesh.triangles.push_back(stoul(norm)-1);
			}
			if (needsNormal) {
				cl_double3 A = mesh.vertices[mesh.triangles[mesh.triangles.size() - 3]];
				cl_double3 B = mesh.vertices[mesh.triangles[mesh.triangles.size() - 6]];
				cl_double3 C = mesh.vertices[mesh.triangles[mesh.triangles.size() - 9]];
				cl_double3 N = normalize(cross(B - A, C - A));
				mesh.normals.push_back(N);
			}
		}
		lineno++;
	}
	mesh.GenerateOctree();
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
	if (!OBJReader("models/pear.obj", theMesh)) {
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
	cpu_objects[6].color = float3(1, 1, 1);
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
	kernel.setArg(1, object_count);
	kernel.setArg(2, cl_vertices);
	kernel.setArg(3, cl_normals);
	kernel.setArg(4, cl_triangles);
	kernel.setArg(5, (unsigned int)(theMesh.triangles.size()/9));
	kernel.setArg(6, cl_octrees);
	kernel.setArg(7, cl_octreeTris);
	kernel.setArg(8, window_width);
	kernel.setArg(9, window_height);
	kernel.setArg(10, cl_vbo);
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

	clock_end = std::chrono::high_resolution_clock::now();
	int ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock_end - clock_start).count();
	std::cout << 1000.0 * framenumber / ms << " fps \t" << ms / 1000.0 << "s" << std::endl;

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

	TRS(&cpu_objects[6], double3(0, 0, 12), ms / 1000.0, double3(1, 1, 0), double3(1, 1, 1));

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

	cl_vertices = Buffer(context, CL_MEM_READ_ONLY, theMesh.vertices.size() * sizeof(cl_double3));
	queue.enqueueWriteBuffer(cl_vertices, CL_TRUE, 0, theMesh.vertices.size() * sizeof(cl_double3), &theMesh.vertices[0]);

	cl_normals = Buffer(context, CL_MEM_READ_ONLY, theMesh.normals.size() * sizeof(cl_double3));
	queue.enqueueWriteBuffer(cl_normals, CL_TRUE, 0, theMesh.normals.size() * sizeof(cl_double3), &theMesh.normals[0]);

	cl_triangles = Buffer(context, CL_MEM_READ_ONLY, theMesh.triangles.size() * sizeof(unsigned int));
	queue.enqueueWriteBuffer(cl_triangles, CL_TRUE, 0, theMesh.triangles.size() * sizeof(unsigned int), &theMesh.triangles[0]);

	cl_octrees = Buffer(context, CL_MEM_READ_ONLY, theMesh.octree.size() * sizeof(Octree));
	queue.enqueueWriteBuffer(cl_octrees, CL_TRUE, 0, theMesh.octree.size() * sizeof(Octree), &theMesh.octree[0]);
	
	cl_octreeTris = Buffer(context, CL_MEM_READ_ONLY, theMesh.octreeTris.size() * sizeof(int));
	queue.enqueueWriteBuffer(cl_octreeTris, CL_TRUE, 0, theMesh.octreeTris.size() * sizeof(int), &theMesh.octreeTris[0]);


	// create OpenCL buffer from OpenGL vertex buffer object
	cl_vbo = BufferGL(context, CL_MEM_WRITE_ONLY, vbo);
	cl_vbos.push_back(cl_vbo);

	// intitialise the kernel
	initCLKernel();

	clock_start = std::chrono::high_resolution_clock::now();

	// start rendering continuously
	glutMainLoop();

	// release memory
	cleanUp();

	system("PAUSE");
}
