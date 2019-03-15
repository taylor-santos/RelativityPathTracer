// OpenCL ray tracing tutorial by Sam Lapere, 2016
// http://raytracey.blogspot.com

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <Windows.h>
#include <CL\cl.hpp>
#include <chrono>
#include <map>
#include <math.h>

#include "gl_interop.h"
#define cimg_use_jpeg
#include "CImg.h"

// TODO
// cleanup()
// check for cl-gl interop

const int object_count = 20;

std::chrono::time_point<std::chrono::high_resolution_clock> clock_start, clock_end, clock_prev;
double currTime = 0;

// OpenCL objects
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
cl_double3 white_point;
double ambient;

// image buffer (not needed with real-time viewport)
cl_float4* cpu_output;
cl_int err;
unsigned int framenumber = 0;


// padding with dummy variables are required for memory alignment
// float3 is considered as float4 by OpenCL
// alignment can also be enforced by using __attribute__ ((aligned (16)));
// see https://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/attributes-variables.html

enum objectType{SPHERE, CUBE, MESH};

struct Object
{
	cl_double4 M[4];
	cl_double4 InvM[4];
	cl_double4 Lorentz[4] = { {{1,0,0,0}},{{0,1,0,0}},{{0,0,1,0}},{0,0,0,1} };
	cl_double4 InvLorentz[4] = { {{1,0,0,0}},{{0,1,0,0}},{{0,0,1,0}},{0,0,0,1} };
	cl_double4 stationaryCam;
	cl_double3 color;
	enum objectType type;
	int meshIndex;
	int textureIndex = -1;
	int textureWidth;
	int textureHeight;
	bool light = false;
	char dummy[4];
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
	std::vector<int> meshIndices;

	void GenerateOctree(int firstTriIndex);
};

Mesh theMesh;

std::vector<unsigned char> textures;
std::vector<int> textureValues; // {index, width, height}

bool ReadTexture(std::string path) {
	using namespace cimg_library;
	cimg::exception_mode(0);
	try {
		CImg<unsigned char> image(path.c_str());
		image.permute_axes("cxyz");
		textureValues.push_back(textures.size());
		textureValues.push_back(image._height); // Texture width
		textureValues.push_back(image._depth);  // Texture height
		textures.insert(textures.end(), image._data, image._data + 3 * image._height * image._depth);
		return true;
	}
	catch (CImgException &e) {
		std::cerr << e.what() << std::endl;
		return false;
	}
}

void json(Mesh const& mesh, const unsigned int octreeIndex, std::ostream &out, const unsigned int indent) {
	out << "{" << std::endl;
	out << std::string(indent + 1, '\t') << "\"bounds\": {" << std::endl;
	out << std::string(indent + 2, '\t') << "\"min\": \"(" << mesh.octree[octreeIndex].min.x << ", "
		<< mesh.octree[octreeIndex].min.y << ", " << mesh.octree[octreeIndex].min.z << ")\"," << std::endl;
	out << std::string(indent + 2, '\t') << "\"max\": \"(" << mesh.octree[octreeIndex].max.x << ", "
		<< mesh.octree[octreeIndex].max.y << ", " << mesh.octree[octreeIndex].max.z << ")\"" << std::endl;
	out << std::string(indent + 1, '\t') << "}," << std::endl;
	out << std::string(indent + 1, '\t') << "\"trisCount\": " << mesh.octree[octreeIndex].trisCount << "," << std::endl;
	out << std::string(indent + 1, '\t') << "\"tris\": [";
	std::string sep = "";
	for (int i = 0; i < mesh.octree[octreeIndex].trisCount; i++) {
		out << sep << mesh.octreeTris[mesh.octree[octreeIndex].trisIndex + i];
		sep = ", ";
	}
	out << "]," << std::endl;
	out << std::string(indent + 1, '\t') << "\"Mathematica\": \"Show[Graphics3D[{Opacity[0.5],Cuboid[{"
		<< mesh.octree[octreeIndex].min.x << "," << mesh.octree[octreeIndex].min.y << "," << mesh.octree[octreeIndex].min.z << "},{"
		<< mesh.octree[octreeIndex].max.x << "," << mesh.octree[octreeIndex].max.y << "," << mesh.octree[octreeIndex].max.z
		<< "}]}],Graphics3D[Triangle[{";
	sep = "";
	for (int i = 0; i < mesh.octree[octreeIndex].trisCount; i++) {
		int triIndex = mesh.octreeTris[mesh.octree[octreeIndex].trisIndex + i];
		out << sep << "{{" << mesh.vertices[mesh.triangles[9*triIndex+3*0]].x
			<< "," << mesh.vertices[mesh.triangles[9 * triIndex + 3 * 0]].y
			<< "," << mesh.vertices[mesh.triangles[9 * triIndex + 3 * 0]].z << "},{"
			<< mesh.vertices[mesh.triangles[9 * triIndex + 3 * 1]].x
			<< "," << mesh.vertices[mesh.triangles[9 * triIndex + 3 * 1]].y
			<< "," << mesh.vertices[mesh.triangles[9 * triIndex + 3 * 1]].z << "},{"
			<< mesh.vertices[mesh.triangles[9 * triIndex + 3 * 2]].x
			<< "," << mesh.vertices[mesh.triangles[9 * triIndex + 3 * 2]].y
			<< "," << mesh.vertices[mesh.triangles[9 * triIndex + 3 * 2]].z << "}}";
		sep = ",";
	}
	out << "}]]]\"," << std::endl;
	out << std::string(indent + 1, '\t') << "\"children\": {";
	if (mesh.octree[octreeIndex].children[0] != -1) {
		sep = "";
		for (int i = 0; i < 8; i++) {
			out << sep << std::endl;
			out << std::string(indent + 2, '\t') << "\"" << i << "\": ";
			json(mesh, mesh.octree[octreeIndex].children[i], out, indent + 2);
			sep = ",";
		}
		out << std::endl << std::string(indent + 1, '\t');
	}
	out << "}" << std::endl;
	out << std::string(indent, '\t') << "}";
}

void pickPlatform(cl::Platform& platform, const std::vector<cl::Platform>& platforms) {

	if (platforms.size() == 1) platform = platforms[0];
	else{
		int input = 0;
		std::cout << "\nChoose an OpenCL platform: ";
		std::cin >> input;

		// handle incorrect user input
		while (input < 1 || input > platforms.size()){
			std::cin.clear(); //clear errors/bad flags on cin
			std::cin.ignore(std::cin.rdbuf()->in_avail(), '\n'); // ignores exact number of chars in cin buffer
			std::cout << "No such option. Choose an OpenCL platform: ";
			std::cin >> input;
		}
		platform = platforms[input - 1];
	}
}

void pickDevice(cl::Device& device, const std::vector<cl::Device>& devices){

	if (devices.size() == 1) device = devices[0];
	else{
		int input = 0;
		std::cout << "\nChoose an OpenCL device: ";
		std::cin >> input;

		// handle incorrect user input
		while (input < 1 || input > devices.size()){
			std::cin.clear(); //clear errors/bad flags on cin
			std::cin.ignore(std::cin.rdbuf()->in_avail(), '\n'); // ignores exact number of chars in cin buffer
			std::cout << "No such option. Choose an OpenCL device: ";
			std::cin >> input;
		}
		device = devices[input - 1];
	}
}

void initOpenCL()
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
	pickPlatform(platform, platforms);
	std::cout << "\nUsing OpenCL platform: \t" << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

	// Get available OpenCL devices on platform
	std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

	std::cout << "Available OpenCL devices on this platform: " << std::endl << std::endl;
	for (int i = 0; i < devices.size(); i++){
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

	context = cl::Context(device, properties);

	// Create a command queue
	queue = cl::CommandQueue(context, device);

	
	// Convert the OpenCL source code to a string// Convert the OpenCL source code to a string
	
	std::ifstream file("opencl_kernel.cl");
	if (!file){
		std::cout << "\nNo OpenCL file found!" << std::endl << "Exiting..." << std::endl;
		system("PAUSE");
		exit(1);
	}
	std::string source{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};

	const char* kernel_source = source.c_str();

	// Create an OpenCL program with source
	program = cl::Program(context, kernel_source);

	// Build the program for the selected device 
	cl_int result = program.build({ device }); // "-cl-fast-relaxed-math"
	if (result) std::cout << "Error during compilation OpenCL code!!!\n (" << result << ")" << std::endl;
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

cl_double3 operator+(const cl_double3 &v1, const cl_double3 &v2) {
	return double3(
		v1.x + v2.x,
		v1.y + v2.y,
		v1.z + v2.z
	);
}

cl_double3 &operator+=(cl_double3 &v1, const cl_double3 &v2) {
	v1 = v1 + v2;
	return v1;
}

cl_double3 operator-(const cl_double3 &v1, const cl_double3 &v2) {
	return double3(
		v1.x - v2.x,
		v1.y - v2.y,
		v1.z - v2.z
	);
}

cl_double3 operator*(const cl_double3 &v, const double &c) {
	return double3(
		v.x * c,
		v.y * c,
		v.z * c
	);
}
cl_double3 operator*(const double &c, const cl_double3 &v) {
	return v * c;
}

cl_double3 operator/(const cl_double3 &v, const double &c) {
	return double3(
		v.x / c,
		v.y / c,
		v.z / c
	);
}

double dot(const cl_double3 &a, const cl_double3 &b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

cl_double3 cross(const cl_double3 &a, const cl_double3 &b) {
	return double3(
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	);
}

cl_double3 elementwise_min(const cl_double3 &a, const cl_double3 &b) {
	return double3(
		min(a.x, b.x),
		min(a.y, b.y),
		min(a.z, b.z)
	);
}

cl_double3 elementwise_max(const cl_double3 &a, const cl_double3 &b) {
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
	cl_double3 extents = (max - min) / 2;

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
	if (depth <= 0 || mesh.octree[octreeIndex].trisCount <= minTris) return;
	cl_double3 extents = mesh.octree[octreeIndex].max - mesh.octree[octreeIndex].min;
	cl_double3 half_extents = extents / 2;
	cl_double3 ex = double3(half_extents.x, 0, 0);
	cl_double3 ey = double3(0, half_extents.y, 0);
	cl_double3 ez = double3(0, 0, half_extents.z);
	int trisStart = mesh.octree[octreeIndex].trisIndex;
	int trisCount = mesh.octree[octreeIndex].trisCount;
	std::map<int, int> trisPerVertex;
	int maxTrisPerVertex = 0;
	for (int tri = trisStart; tri < trisStart + trisCount; tri++) {
		int triIndex = mesh.octreeTris[tri];
		trisPerVertex[mesh.triangles[9 * triIndex + 3 * 0]]++;
		trisPerVertex[mesh.triangles[9 * triIndex + 3 * 1]]++;
		trisPerVertex[mesh.triangles[9 * triIndex + 3 * 2]]++;
		maxTrisPerVertex = max(maxTrisPerVertex, trisPerVertex[mesh.triangles[9 * triIndex + 3 * 0]]);
		maxTrisPerVertex = max(maxTrisPerVertex, trisPerVertex[mesh.triangles[9 * triIndex + 3 * 1]]);
		maxTrisPerVertex = max(maxTrisPerVertex, trisPerVertex[mesh.triangles[9 * triIndex + 3 * 2]]);
	}
	for (int x = 0; x < 2; x++) {
		for (int y = 0; y < 2; y++) {
			for (int z = 0; z < 2; z++) {
				Octree child;
				child.min = mesh.octree[octreeIndex].min + ex * x + ey * y + ez * z;
				child.max = child.min + half_extents;
				
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
	for (int x = 0; x < 2; x++) {
		for (int y = 0; y < 2; y++) {
			for (int z = 0; z < 2; z++) {
				int childIndex = 4 * x + 2 * y + z;
				int childOctreeIndex = mesh.octree[octreeIndex].children[childIndex];
				if (z == 0) {
					mesh.octree[childOctreeIndex].neighbors[0] = mesh.octree[octreeIndex].neighbors[0];
					mesh.octree[childOctreeIndex].neighbors[1] = mesh.octree[octreeIndex].children[childIndex + 1];
				}
				else {
					mesh.octree[childOctreeIndex].neighbors[0] = mesh.octree[octreeIndex].children[childIndex - 1];
					mesh.octree[childOctreeIndex].neighbors[1] = mesh.octree[octreeIndex].neighbors[1];
				}
				if (x == 0) {
					mesh.octree[childOctreeIndex].neighbors[2] = mesh.octree[octreeIndex].neighbors[2];
					mesh.octree[childOctreeIndex].neighbors[3] = mesh.octree[octreeIndex].children[childIndex + 4];
				}
				else {
					mesh.octree[childOctreeIndex].neighbors[2] = mesh.octree[octreeIndex].children[childIndex - 4];
					mesh.octree[childOctreeIndex].neighbors[3] = mesh.octree[octreeIndex].neighbors[3];
				}
				if (y == 0) {
					mesh.octree[childOctreeIndex].neighbors[4] = mesh.octree[octreeIndex].neighbors[4];
					mesh.octree[childOctreeIndex].neighbors[5] = mesh.octree[octreeIndex].children[childIndex + 2];
				}
				else {
					mesh.octree[childOctreeIndex].neighbors[4] = mesh.octree[octreeIndex].children[childIndex - 2];
					mesh.octree[childOctreeIndex].neighbors[5] = mesh.octree[octreeIndex].neighbors[5];
				}
			}
		}
	}
	for (int i = 0; i < 8; i++) {
		Subdivide(mesh, mesh.octree[octreeIndex].children[i], maxTrisPerVertex, depth - 1);
	}
}

void Mesh::GenerateOctree(int firstTriIndex) {
	Octree newOctree;
	newOctree.trisCount = 0;
	newOctree.trisIndex = octreeTris.size();
	newOctree.min = vertices[triangles[firstTriIndex]];
	newOctree.max = vertices[triangles[firstTriIndex]];
	for (int i = firstTriIndex/3; i < triangles.size() / 3; i++) {
		cl_double3 vert = vertices[triangles[3 * i]];
		newOctree.min = elementwise_min(newOctree.min, vert);
		newOctree.max = elementwise_max(newOctree.max, vert);
	}
	for (int i = 0; i < triangles.size() / 9; i++) {
		octreeTris.push_back(i);
		newOctree.trisCount++;
	}
	int octreeIndex = octree.size();
	octree.push_back(newOctree);
	Subdivide(*this, octreeIndex, 0, 15);
	/*
	ofstream f("octree.json");
	json(*this, 0, f, 0);
	f.close();
	*/
}

bool ReadOBJ(std::string path, Mesh &mesh) {
	if (path.substr(path.size() - 4, 4) != ".obj") return false;
	std::ifstream file(path);
	if (!file) {
		perror("Error opening OBJ file");
		return false;
	}
	std::string line;
	std::map<int, std::vector<int>> vertToTrisMap;
	int lineno = 0;
	int firstTriIndex = mesh.triangles.size();
	int firstVertIndex = mesh.vertices.size();
	int firstNormIndex = mesh.normals.size();
	int firstUVIndex = mesh.uvs.size();
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
			int triIndex = mesh.triangles.size() / 9;
			for (int i = 0; i < 3; i++) {
				stream >> tri;
				std::istringstream vertstream(tri);
				std::string vert, uv, norm;
				std::getline(vertstream, vert, '/');
				int vertIndex = stoul(vert) - 1 + firstVertIndex;
				if (!std::getline(vertstream, uv, '/')) {
					uv = "1";
				}
				if (!std::getline(vertstream, norm, '/')) {
					norm = "1";
					vertToTrisMap[vertIndex].push_back(triIndex);
				}				
				mesh.triangles.push_back(vertIndex);
				mesh.triangles.push_back(stoul(uv) - 1 + firstUVIndex);
				mesh.triangles.push_back(stol(norm) - 1 + firstNormIndex);
			}
		}
		lineno++;
	}
	for (const auto& kv : vertToTrisMap) {
		int vertIndex = kv.first;
		auto triList = kv.second;
		cl_double3 N = double3(0, 0, 0);
		for (int triIndex : triList) {
			int AIndex = mesh.triangles[9 * triIndex + 3 * 0];
			int BIndex = mesh.triangles[9 * triIndex + 3 * 1];
			int CIndex = mesh.triangles[9 * triIndex + 3 * 2];
			cl_double3 A = mesh.vertices[AIndex];
			cl_double3 B = mesh.vertices[BIndex];
			cl_double3 C = mesh.vertices[CIndex];
			// Don't normalize: the cross product is proportional to the area of the triangle,
			// and we want the normal contribution to be proportional to the area as well.
			N += cross(B - A, C - A);
			if (AIndex == vertIndex) {
				mesh.triangles[2 + 9 * triIndex + 3 * 0] = mesh.normals.size();
			} else if (BIndex == vertIndex) {
				mesh.triangles[2 + 9 * triIndex + 3 * 1] = mesh.normals.size();
			}else if (CIndex == vertIndex) {
				mesh.triangles[2 + 9 * triIndex + 3 * 2] = mesh.normals.size();
			}
		}
		mesh.normals.push_back(normalize(N));
	}
	int newOctreeIndex = mesh.octree.size();
	mesh.meshIndices.push_back(newOctreeIndex);
	mesh.GenerateOctree(firstTriIndex);
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

void setLorentzBoost(Object *object, cl_double3 v) {
	double gamma = 1.0 / sqrt(1.0 - dot(v, v));
	double vSqr = dot(v, v);
	object->Lorentz[0]    = double4(gamma,        -v.x * gamma,                           -v.y * gamma,                           -v.z * gamma);
	object->Lorentz[1]    = double4(-v.x * gamma, (gamma - 1.0) * v.x * v.x / vSqr + 1.0, (gamma - 1.0) * v.x * v.y / vSqr,       (gamma - 1.0) * v.x * v.z / vSqr);
	object->Lorentz[2]    = double4(-v.y * gamma, (gamma - 1.0) * v.y * v.x / vSqr,       (gamma - 1.0) * v.y * v.y / vSqr + 1.0, (gamma - 1.0) * v.y * v.z / vSqr);
	object->Lorentz[3]    = double4(-v.z * gamma, (gamma - 1.0) * v.z * v.x / vSqr,       (gamma - 1.0) * v.z * v.y / vSqr,       (gamma - 1.0) * v.z * v.z / vSqr + 1.0); 
	
	object->InvLorentz[0] = double4(gamma,        v.x * gamma,                            v.y * gamma,                            v.z * gamma);
	object->InvLorentz[1] = object->Lorentz[1];
	object->InvLorentz[1].x *= -1;
	object->InvLorentz[2] = object->Lorentz[2];
	object->InvLorentz[2].x *= -1;
	object->InvLorentz[3] = object->Lorentz[3];
	object->InvLorentz[3].x *= -1;
};

void initScene(Object* cpu_objects) {
	if (!ReadTexture("textures/earth.jpg")) {
		exit(EXIT_FAILURE);
	}
	if (!ReadTexture("textures/StanfordBunnyTerracotta.jpg")) {
		exit(EXIT_FAILURE);
	}
	if (!ReadTexture("textures/bricks.jpg")) {
		exit(EXIT_FAILURE);
	}
	/*
	if (!ReadOBJ("models/pear.obj", theMesh)) {
		exit(EXIT_FAILURE);
	}
	/*
	if (!ReadOBJ("models/StanfordBunny.obj", theMesh)) {
		exit(EXIT_FAILURE);
	}
	*/
	queue.enqueueWriteBuffer(cl_objects, CL_TRUE, 0, object_count * sizeof(Object), cpu_objects);

	white_point = double3(1, 1, 1);
	ambient = 1;
	/*
	cpu_objects[0].textureIndex = textureValues[0];
	cpu_objects[0].textureWidth = textureValues[1];
	cpu_objects[0].textureHeight = textureValues[2];
	*/
	//cpu_objects[0].meshIndex = theMesh.meshIndices[0];



	cl_double3 p0 = double3(2 * sqrt(2.0) - 10.0 + 3, -4, 2 * sqrt(2.0) + 5);
	cl_double3 dir = double3(1, 0, 1);
	dir = normalize(dir);

	double cosC = dot(-1 * dir, normalize(p0));
	std::cout << cosC << std::endl;
	double b = magnitude(p0);

	

	cl_double3 offset = double3(0, 2, 0);
	cpu_objects[0].color = double3(0.2, 0.2, 0.2);
	cpu_objects[0].textureIndex = textureValues[0];
	cpu_objects[0].textureWidth = textureValues[1];
	cpu_objects[0].textureHeight = textureValues[2];
	cpu_objects[0].type = SPHERE;
	TRS(&cpu_objects[0], p0 + 10 * dir + offset, 0, double3(0, 1, 0), double3(1, 1, 1));
	setLorentzBoost(&cpu_objects[0], 0.9 * dir);

	cpu_objects[object_count - 2].color = double3(1, 1, 1);
	cpu_objects[object_count - 2].type = SPHERE;
	cpu_objects[object_count - 2].light = true;
	TRS(&cpu_objects[object_count - 2], p0 + offset + 10*dir, 0, double3(0, 1, 0), double3(0.1, 0.1, 0.1));
	setLorentzBoost(&cpu_objects[object_count - 2], -0.9 * dir);

	for (int i = 1; i < object_count - 2; i++) {
		cpu_objects[i].color = double3(i%3==0 ? 1.0:0.0, i%3==1?1.0:0.0, i%3==2?1.0:0.0);
		cpu_objects[i].type = SPHERE;
		double c = magnitude(p0) + 2.0 * (i-1);
		double a = b * cosC + sqrt(-b * b + c * c + b * b*cosC*cosC);
		TRS(&cpu_objects[i], p0 + dir*a, 0, double3(0, 1, 0), double3(1, 1, 1));
		std::cout << (p0 + dir * a).x << ", " << (double)(p0 + dir * a).y << ", " << (p0 + dir * a).z << std::endl;
	}
	cpu_objects[object_count-1].color = float3(0.9f, 0.8f, 0.7f);
	cpu_objects[object_count-1].type = CUBE;
	cpu_objects[object_count-1].textureIndex = textureValues[6];
	cpu_objects[object_count-1].textureWidth = textureValues[7];
	cpu_objects[object_count-1].textureHeight = textureValues[8];
	TRS(&cpu_objects[object_count-1], double3(0, -4, 20), 0.0001, double3(0, 1, 0), double3(40, 0.1, 40));


	/*

	cpu_objects[object_count/2].color = double3(169 / 255.0, 168 / 255.0, 54 / 255.0);
	cpu_objects[object_count/2].type = SPHERE;
	//cpu_objects[object_count/2].meshIndex = theMesh.meshIndices[0];
	TRS(&cpu_objects[object_count/2], double3(-5 + 3, 2, 10), 3.1415926 / 4, double3(0, 1, 0), double3(1, 1, 1));
	setLorentzBoost(&cpu_objects[object_count/2], double3(-0.999 / sqrt(2.0), 0, -0.999 / sqrt(2.0)));
	for (int i = object_count / 2 + 1; i < object_count; i++) {
		cpu_objects[i].color = double3(i % 3 == 1 ? 1.0 : 0.0, i % 3 == 2 ? 1.0 : 0.0, i % 3 == 0 ? 1.0 : 0.0);
		cpu_objects[i].type = SPHERE;
		TRS(&cpu_objects[i], double3(2 * sqrt(2.0)*(i-object_count/2) - 10.0 + 3, 2, 2 * sqrt(2.0)*(i - object_count / 2)+5), 3.1415926 / 4, double3(0, 1, 0), double3(1, 1, 1));
	}



	/*
	// left wall
	cpu_objects[0].color = float3(0.75f, 0.25f, 0.25f);
	cpu_objects[0].type = CUBE;
	TRS(&cpu_objects[0], double3(60, 0, 10), 0, double3(0, 1, 0), double3(0.1f, 10, 10));

	// right wall
	cpu_objects[1].color = float3(0.25f, 0.25f, 0.75f);
	cpu_objects[1].type = CUBE;
	TRS(&cpu_objects[1], double3(60, 0, 10), 0, double3(0, 1, 0), double3(0.1f, 10, 10));

	// floor
	cpu_objects[2].color = float3(0.25f, 0.75f, 0.25f);
	cpu_objects[2].type = CUBE;
	TRS(&cpu_objects[2], double3(0, -6, 10), 0, double3(0, 1, 0), double3(10, 0.1f, 10));

	// ceiling
	cpu_objects[3].color = float3(0.9, 0.8, 0.7);
	cpu_objects[3].type = CUBE;
	TRS(&cpu_objects[3], double3(0, 6, 10), 0, double3(0, 1, 0), double3(10, 0.1f, 10));

	// cube
	cpu_objects[4].color = float3(0.9f, 0.8f, 0.7f);
	cpu_objects[4].type = CUBE;
	cpu_objects[4].textureIndex = textureValues[6];
	cpu_objects[4].textureWidth = textureValues[7];
	cpu_objects[4].textureHeight = textureValues[8];
	TRS(&cpu_objects[4], double3(-4, 0, 5), 0.001, double3(0, 1, 0), double3(1, 1, 1));
	setLorentzBoost(&cpu_objects[4], double3(0.6, 0, 0));

	// Sphere
	cpu_objects[5].color = float3(1, 1, 1);
	cpu_objects[5].type = SPHERE;
	cpu_objects[5].textureIndex = textureValues[0];
	cpu_objects[5].textureWidth = textureValues[1];
	cpu_objects[5].textureHeight = textureValues[2];
	TRS(&cpu_objects[5], double3(0, -1.5, 11), 0, double3(0, 1, 0), double3(1, 1, 1));
	setLorentzBoost(&cpu_objects[5], double3(0, 0, 0.9));

	// Light
	cpu_objects[6].color = white_point;
	cpu_objects[6].type = SPHERE;
	cpu_objects[6].light = true;
	TRS(&cpu_objects[6], double3(0, 5, 10), 0, double3(0, 1, 0), double3(0.1, 0.1, 0.1));
	/*
	// Pear
	cpu_objects[5].color = float3(169/255.0, 168/255.0, 54/255.0);
	cpu_objects[5].type = MESH;
	cpu_objects[5].meshIndex = theMesh.meshIndices[0];

	// Bunny
	cpu_objects[6].color = float3(1.0f, 0.2f, 0.9f);
	cpu_objects[6].type = MESH;
	cpu_objects[6].meshIndex = theMesh.meshIndices[1];
	cpu_objects[6].textureIndex = textureValues[3];
	cpu_objects[6].textureWidth = textureValues[4];
	cpu_objects[6].textureHeight = textureValues[5];
	*/
	
}

void initCLKernel(){

	// pick a rendermode
	unsigned int rendermode = 1;

	// Create a kernel (entry point in the OpenCL source program)
	kernel = cl::Kernel(program, "render_kernel");

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
	kernel.setArg(11, currTime);
	kernel.setArg(12, window_width);
	kernel.setArg(13, window_height);
	kernel.setArg(14, cl_vbo);
}

void runKernel(){
	// every pixel in the image has its own thread or "work item",
	// so the total amount of work items equals the number of pixels
	std::size_t global_work_size = window_width * window_height;
	std::size_t local_work_size = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);

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
	currTime = ms / 1000.0;
	int frame_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock_end - clock_prev).count();
	std::cout << 1000.0 / frame_ms << " fps\taverage: " << 1000.0*framenumber / ms << std::endl;
	clock_prev = std::chrono::high_resolution_clock::now();

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
		cl_vbo = cl::BufferGL(context, CL_MEM_WRITE_ONLY, vbo);
		cl_vbos[0] = cl_vbo;
		kernel.setArg(12, window_width);
		kernel.setArg(13, window_height);
		kernel.setArg(14, cl_vbo);
	}
	double s = ms / 1000.0;
	//TRS(&cpu_objects[4], double3(-10, 0, 5), 3.1415926/2 * floor(ms / 2000.0), double3(0, 1, 0), double3(1, 1, 1));
	/*

	double x = ms / 400.0;
	TRS(&cpu_objects[5], double3(1, 2*sin(x + 3.14159 / 2) - 3.5, 13), 0, double3(0, 1, 0), double3(0.5, 0.5 * (0.8 * sin(1.0 - pow(sin(x/2), 10))/sin(1) + 0.2), 0.5));

	TRS(&cpu_objects[6], double3(4 * sin(ms / 2000.0), -0.5, 9), -3.1415926 / 2 * ms / 3000.0, double3(0, 1, 0), double3(10, 10, -10));

	TRS(&cpu_objects[7], double3(-1, -1.5, 9 + 2 * sin(ms/1500.0)), ms/500.0, double3(0, 1, 0), double3(1, 1, 1));
	*/

	//TRS(&cpu_objects[object_count - 1], double3(0, -4 + sin(s/10), 20), 0.000, double3(0, 1, 0), double3(40, 0.1, 40));
	//setLorentzBoost(&cpu_objects[1], double3(sin(s/10), 0, 0));
	queue.enqueueWriteBuffer(cl_objects, CL_TRUE, 0, object_count * sizeof(Object), cpu_objects);

	cl_double4 cameraPos = double4(currTime, 0, 0, 0);
	for (int i = 0; i < object_count; i++) {
		cpu_objects[i].stationaryCam = double4(
			dot(cpu_objects[i].Lorentz[0], cameraPos),
			dot(cpu_objects[i].Lorentz[1], cameraPos),
			dot(cpu_objects[i].Lorentz[2], cameraPos),
			dot(cpu_objects[i].Lorentz[3], cameraPos)
		);
	}

	kernel.setArg(0, cl_objects);
	kernel.setArg(11, currTime);

	runKernel();

	drawGL();

}

void cleanUp(){
//	delete cpu_output;
}

void main(int argc, char** argv){
	std::cout << sizeof(Object) << std::endl;
	// initialise OpenGL (GLEW and GLUT window + callback functions)
	initGL(argc, argv);
	std::cout << "OpenGL initialized \n";

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

	cl_objects = cl::Buffer(context, CL_MEM_READ_ONLY, object_count * sizeof(Object));
	queue.enqueueWriteBuffer(cl_objects, CL_TRUE, 0, object_count * sizeof(Object), cpu_objects);

	cl_vertices = cl::Buffer(context, CL_MEM_READ_ONLY, theMesh.vertices.size() * sizeof(cl_double3));
	queue.enqueueWriteBuffer(cl_vertices, CL_TRUE, 0, theMesh.vertices.size() * sizeof(cl_double3), theMesh.vertices.size() > 0 ? &theMesh.vertices[0] : NULL);

	cl_normals = cl::Buffer(context, CL_MEM_READ_ONLY, theMesh.normals.size() * sizeof(cl_double3));
	queue.enqueueWriteBuffer(cl_normals, CL_TRUE, 0, theMesh.normals.size() * sizeof(cl_double3), theMesh.normals.size() > 0 ? &theMesh.normals[0] : NULL);
	
	cl_uvs = cl::Buffer(context, CL_MEM_READ_ONLY, theMesh.uvs.size() * sizeof(cl_double2));
	queue.enqueueWriteBuffer(cl_uvs, CL_TRUE, 0, theMesh.uvs.size() * sizeof(cl_double2), theMesh.uvs.size() > 0 ? &theMesh.uvs[0] : NULL);

	cl_triangles = cl::Buffer(context, CL_MEM_READ_ONLY, theMesh.triangles.size() * sizeof(unsigned int));
	queue.enqueueWriteBuffer(cl_triangles, CL_TRUE, 0, theMesh.triangles.size() * sizeof(unsigned int), theMesh.triangles.size() > 0 ? &theMesh.triangles[0] : NULL);

	cl_octrees = cl::Buffer(context, CL_MEM_READ_ONLY, theMesh.octree.size() * sizeof(Octree));
	queue.enqueueWriteBuffer(cl_octrees, CL_TRUE, 0, theMesh.octree.size() * sizeof(Octree), theMesh.octree.size() > 0 ? &theMesh.octree[0] : NULL);
	
	cl_octreeTris = cl::Buffer(context, CL_MEM_READ_ONLY, theMesh.octreeTris.size() * sizeof(int));
	queue.enqueueWriteBuffer(cl_octreeTris, CL_TRUE, 0, theMesh.octreeTris.size() * sizeof(int), theMesh.octreeTris.size() > 0 ? &theMesh.octreeTris[0] : NULL);

	cl_textures = cl::Buffer(context, CL_MEM_READ_ONLY, textures.size() * sizeof(unsigned char));
	queue.enqueueWriteBuffer(cl_textures, CL_TRUE, 0, textures.size() * sizeof(unsigned char), textures.size() > 0 ? &textures[0] : NULL);

	// create OpenCL buffer from OpenGL vertex buffer object
	cl_vbo = cl::BufferGL(context, CL_MEM_WRITE_ONLY, vbo);
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
