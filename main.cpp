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
float currTime = 0;

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
cl_float3 white_point;
float ambient;

// image buffer (not needed with real-time viewport)
cl_float4* cpu_output;
cl_int err;
unsigned int framenumber = 0;


// padding with dummy variables are required for memory alignment
// float3 is considered as float4 by OpenCL
// alignment can also be enforced by using __attribute__ ((aligned (16)));
// see https://www.khronos.org/registry/cl/sdk/1.0f/docs/man/xhtml/attributes-variables.html

enum objectType{SPHERE, CUBE, MESH};

struct Object
{
	cl_float4 M[4];
	cl_float4 InvM[4];
	cl_float4 Lorentz[4] = { {{1,0,0,0}},{{0,1,0,0}},{{0,0,1,0}},{0,0,0,1} };
	cl_float4 InvLorentz[4] = { {{1,0,0,0}},{{0,1,0,0}},{{0,0,1,0}},{0,0,0,1} };
	cl_float4 stationaryCam;
	cl_float3 color;
	enum objectType type;
	int meshIndex;
	int textureIndex = -1;
	int textureWidth;
	int textureHeight;
	bool light = false;
	char dummy[8];
};

Object cpu_objects[object_count];

struct Octree
{
	cl_float3 min;
	cl_float3 max;
	int trisIndex,
		trisCount;
	int children[8]  = { -1, -1, -1, -1, -1, -1, -1, -1 };
	int neighbors[6] = { -1, -1, -1, -1, -1, -1 };
};

struct Mesh
{
	std::vector<cl_float3> vertices;
	std::vector<unsigned int> triangles;
	std::vector<cl_float2> uvs;
	std::vector<cl_float3> normals;
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
	out << std::string(indent + 1, '\t') << "\"Mathematica\": \"Show[Graphics3D[{Opacity[0.5f],Cuboid[{"
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
#define float3(x, y, z) {{x, y, z}}
#define float4(x, y, z, w) {{x, y, z, w}}

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

cl_float3 operator+(const cl_float3 &v1, const cl_float3 &v2) {
	return float3(
		v1.x + v2.x,
		v1.y + v2.y,
		v1.z + v2.z
	);
}

cl_float3 &operator+=(cl_float3 &v1, const cl_float3 &v2) {
	v1 = v1 + v2;
	return v1;
}

cl_float3 operator-(const cl_float3 &v1, const cl_float3 &v2) {
	return float3(
		v1.x - v2.x,
		v1.y - v2.y,
		v1.z - v2.z
	);
}

cl_float3 operator-(const cl_float3 &v) {
	return float3(
		-v.x,
		-v.y,
		-v.z
	);
}

cl_float3 operator*(const cl_float3 &v, const float &c) {
	return float3(
		v.x * c,
		v.y * c,
		v.z * c
	);
}
cl_float3 operator*(const float &c, const cl_float3 &v) {
	return v * c;
}

cl_float3 operator/(const cl_float3 &v, const float &c) {
	return float3(
		v.x / c,
		v.y / c,
		v.z / c
	);
}

float dot(const cl_float4 &a, const cl_float4 &b) {
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

cl_float3 cross(const cl_float3 &a, const cl_float3 &b) {
	return float3(
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	);
}

cl_float3 elementwise_min(const cl_float3 &a, const cl_float3 &b) {
	return float3(
		min(a.x, b.x),
		min(a.y, b.y),
		min(a.z, b.z)
	);
}

cl_float3 elementwise_max(const cl_float3 &a, const cl_float3 &b) {
	return float3(
		max(a.x, b.x),
		max(a.y, b.y),
		max(a.z, b.z)
	);
}

bool AABBTriangleIntersection(Mesh const& mesh, int octreeIndex, int triIndex) {
	const cl_float3 A = mesh.vertices[mesh.triangles[9 * triIndex + 3 * 0]];
	const cl_float3 B = mesh.vertices[mesh.triangles[9 * triIndex + 3 * 1]];
	const cl_float3 C = mesh.vertices[mesh.triangles[9 * triIndex + 3 * 2]];
	cl_float3 min = mesh.octree[octreeIndex].min;
	cl_float3 max = mesh.octree[octreeIndex].max;
	cl_float3 center = (min + max) / 2;
	cl_float3 extents = (max - min) / 2;

	cl_float3 offsetA = A - center;
	cl_float3 offsetB = B - center;
	cl_float3 offsetC = C - center;

	cl_float3 ba = offsetB - offsetA;
	cl_float3 cb = offsetC - offsetB;

	float x_ba_abs = abs(ba.x);
	float y_ba_abs = abs(ba.y);
	float z_ba_abs = abs(ba.z);
	{
		float min = ba.z * offsetA.y - ba.y * offsetA.z;
		float max = ba.z * offsetC.y - ba.y * offsetC.z;
		if (min > max) {
			float temp = min;
			min = max;
			max = temp;
		}
		float rad = z_ba_abs * extents.y + y_ba_abs * extents.z;
		if (min > rad || max < -rad) return false;
	}
	{
		float min = -ba.z * offsetA.x + ba.x * offsetA.z;
		float max = -ba.z * offsetC.x + ba.x * offsetC.z;
		if (min > max) {
			float temp = min;
			min = max;
			max = temp;
		}
		float rad = z_ba_abs * extents.x + x_ba_abs * extents.z;
		if (min > rad || max < -rad) return false;
	}
	{
		float min = ba.y * offsetB.x - ba.x * offsetB.y;
		float max = ba.y * offsetC.x - ba.x * offsetC.y;
		if (min > max) {
			float temp = min;
			min = max;
			max = temp;
		}
		float rad = y_ba_abs * extents.x + x_ba_abs * extents.y;
		if (min > rad || max < -rad) return false;
	}
	float x_cb_abs = abs(cb.x);
	float y_cb_abs = abs(cb.y);
	float z_cb_abs = abs(cb.z);
	{
		float min = cb.z * offsetA.y - cb.y * offsetA.z,
			max = cb.z * offsetC.y - cb.y * offsetC.z;
		if (min > max) {
			float temp = min;
			min = max;
			max = temp;
		}
		float rad = z_cb_abs * extents.y + y_cb_abs * extents.z;
		if (min > rad || max < -rad) return false;
	}
	{
		float min = -cb.z * offsetA.x + cb.x * offsetA.z,
			max = -cb.z * offsetC.x + cb.x * offsetC.z;
		if (min > max) {
			float temp = min;
			min = max;
			max = temp;
		}
		float rad = z_cb_abs * extents.x + x_cb_abs * extents.z;
		if (min > rad || max < -rad) return false;
	}
	{
		float min = cb.y * offsetA.x - cb.x * offsetA.y,
			max = cb.y * offsetB.x - cb.x * offsetB.y;
		if (min > max) {
			float temp = min;
			min = max;
			max = temp;
		}
		float rad = y_cb_abs * extents.x + x_cb_abs * extents.y;
		if (min > rad || max < -rad) return false;
	}
	cl_float3 ac = offsetA - offsetC;
	float x_ac_abs = abs(ac.x);
	float y_ac_abs = abs(ac.y);
	float z_ac_abs = abs(ac.z);
	{
		float min = ac.z * offsetA.y - ac.y * offsetA.z,
			max = ac.z * offsetB.y - ac.y * offsetB.z;
		if (min > max) {
			float temp = min;
			min = max;
			max = temp;
		}
		float rad = z_ac_abs * extents.y + y_ac_abs * extents.z;
		if (min > rad || max < -rad) return false;
	}
	{
		float min = -ac.z * offsetA.x + ac.x * offsetA.z,
			max = -ac.z * offsetB.x + ac.x * offsetB.z;
		if (min > max) {
			float temp = min;
			min = max;
			max = temp;
		}
		float rad = z_ac_abs * extents.x + x_ac_abs * extents.z;
		if (min > rad || max < -rad) return false;
	}
	{
		float min = ac.y * offsetB.x - ac.x * offsetB.y,
			max = ac.y * offsetC.x - ac.x * offsetC.y;
		if (min > max) {
			float temp = min;
			min = max;
			max = temp;
		}
		float rad = y_ac_abs * extents.x + x_ac_abs * extents.y;
		if (min > rad || max < -rad) return false;
	}
	{
		cl_float3 normal = cross(ba, cb);
		cl_float3 min, max;
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
		cl_float3 min = elementwise_min(elementwise_min(offsetA, offsetB), offsetC);
		cl_float3 max = elementwise_max(elementwise_max(offsetA, offsetB), offsetC);
		if (min.x > extents.x || max.x < -extents.x) return false;
		if (min.y > extents.y || max.y < -extents.y) return false;
		if (min.z > extents.z || max.z < -extents.z) return false;
	}
	return true;
}

void Subdivide(Mesh &mesh, int octreeIndex, int minTris, int depth) {
	if (depth <= 0 || mesh.octree[octreeIndex].trisCount <= minTris) return;
	cl_float3 extents = mesh.octree[octreeIndex].max - mesh.octree[octreeIndex].min;
	cl_float3 half_extents = extents / 2;
	cl_float3 ex = float3(half_extents.x, 0, 0);
	cl_float3 ey = float3(0, half_extents.y, 0);
	cl_float3 ez = float3(0, 0, half_extents.z);
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
		cl_float3 vert = vertices[triangles[3 * i]];
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
			cl_float3 vert;
			stream >> vert.x >> vert.y >> vert.z;
			if (stream.fail()) {
				std::cerr << "Error reading OBJ file \"" << path
					<< "\": Invalid syntax on line " << lineno << std::endl;
				return false;
			}
			mesh.vertices.push_back(vert);
		}
		else if (prefix == "vt") {
			cl_float2 uv;
			stream >> uv.x >> uv.y;
			if (stream.fail()) {
				std::cerr << "Error reading OBJ file \"" << path
					<< "\": Invalid syntax on line " << lineno << std::endl;
				return false;
			}
			mesh.uvs.push_back(uv);
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
		cl_float3 N = float3(0, 0, 0);
		for (int triIndex : triList) {
			int AIndex = mesh.triangles[9 * triIndex + 3 * 0];
			int BIndex = mesh.triangles[9 * triIndex + 3 * 1];
			int CIndex = mesh.triangles[9 * triIndex + 3 * 2];
			cl_float3 A = mesh.vertices[AIndex];
			cl_float3 B = mesh.vertices[BIndex];
			cl_float3 C = mesh.vertices[CIndex];
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
	float A2323 = object->M[2].z * object->M[3].w - object->M[2].w * object->M[3].z;
	float A1323 = object->M[2].y * object->M[3].w - object->M[2].w * object->M[3].y;
	float A1223 = object->M[2].y * object->M[3].z - object->M[2].z * object->M[3].y;
	float A0323 = object->M[2].x * object->M[3].w - object->M[2].w * object->M[3].x;
	float A0223 = object->M[2].x * object->M[3].z - object->M[2].z * object->M[3].x;
	float A0123 = object->M[2].x * object->M[3].y - object->M[2].y * object->M[3].x;
	float A2313 = object->M[1].z * object->M[3].w - object->M[1].w * object->M[3].z;
	float A1313 = object->M[1].y * object->M[3].w - object->M[1].w * object->M[3].y;
	float A1213 = object->M[1].y * object->M[3].z - object->M[1].z * object->M[3].y;
	float A2312 = object->M[1].z * object->M[2].w - object->M[1].w * object->M[2].z;
	float A1312 = object->M[1].y * object->M[2].w - object->M[1].w * object->M[2].y;
	float A1212 = object->M[1].y * object->M[2].z - object->M[1].z * object->M[2].y;
	float A0313 = object->M[1].x * object->M[3].w - object->M[1].w * object->M[3].x;
	float A0213 = object->M[1].x * object->M[3].z - object->M[1].z * object->M[3].x;
	float A0312 = object->M[1].x * object->M[2].w - object->M[1].w * object->M[2].x;
	float A0212 = object->M[1].x * object->M[2].z - object->M[1].z * object->M[2].x;
	float A0113 = object->M[1].x * object->M[3].y - object->M[1].y * object->M[3].x;
	float A0112 = object->M[1].x * object->M[2].y - object->M[1].y * object->M[2].x;

	float det =
		object->M[0].x * (object->M[1].y * A2323 - object->M[1].z * A1323 + object->M[1].w * A1223)
		- object->M[0].y * (object->M[1].x * A2323 - object->M[1].z * A0323 + object->M[1].w * A0223)
		+ object->M[0].z * (object->M[1].x * A1323 - object->M[1].y * A0323 + object->M[1].w * A0123)
		- object->M[0].w * (object->M[1].x * A1223 - object->M[1].y * A0223 + object->M[1].z * A0123);
	if (det == 0.0f) {
		return false;
	}
	det = 1 / det;

	object->InvM[0] = float4(
		det * (object->M[1].y * A2323 - object->M[1].z * A1323 + object->M[1].w * A1223),
		det * -(object->M[0].y * A2323 - object->M[0].z * A1323 + object->M[0].w * A1223),
		det * (object->M[0].y * A2313 - object->M[0].z * A1313 + object->M[0].w * A1213),
		det * -(object->M[0].y * A2312 - object->M[0].z * A1312 + object->M[0].w * A1212)
	);
	object->InvM[1] = float4(
		det * -(object->M[1].x * A2323 - object->M[1].z * A0323 + object->M[1].w * A0223),
		det * (object->M[0].x * A2323 - object->M[0].z * A0323 + object->M[0].w * A0223),
		det * -(object->M[0].x * A2313 - object->M[0].z * A0313 + object->M[0].w * A0213),
		det * (object->M[0].x * A2312 - object->M[0].z * A0312 + object->M[0].w * A0212)
	);
	object->InvM[2] = float4(
		det * (object->M[1].x * A1323 - object->M[1].y * A0323 + object->M[1].w * A0123),
		det * -(object->M[0].x * A1323 - object->M[0].y * A0323 + object->M[0].w * A0123),
		det * (object->M[0].x * A1313 - object->M[0].y * A0313 + object->M[0].w * A0113),
		det * -(object->M[0].x * A1312 - object->M[0].y * A0312 + object->M[0].w * A0112)
	);
	object->InvM[3] = float4(
		det * -(object->M[1].x * A1223 - object->M[1].y * A0223 + object->M[1].z * A0123),
		det * (object->M[0].x * A1223 - object->M[0].y * A0223 + object->M[0].z * A0123),
		det * -(object->M[0].x * A1213 - object->M[0].y * A0213 + object->M[0].z * A0113),
		det * (object->M[0].x * A1212 - object->M[0].y * A0212 + object->M[0].z * A0112)
	);
	return true;
}

void TRS(Object *object, cl_float3 translation, float angle, cl_float3 axis, cl_float3 scale) {
	cl_float3 R[3];
	float c = cos(angle);
	float s = sin(angle);
	cl_float3 u = normalize(axis);
	R[0] = float3(c + u.x*u.x*(1 - c), u.x*u.y*(1 - c) - u.z*s, u.x*u.z*(1 - c) + u.y*s);
	R[1] = float3(u.y*u.x*(1 - c) + u.z*s, c + u.y*u.y*(1 - c), u.y*u.z*(1 - c) - u.x*s);
	R[2] = float3(u.z*u.x*(1 - c) - u.y*s, u.z*u.y*(1 - c) + u.x*s, c + u.z*u.z*(1 - c));
	object->M[0] = float4(R[0].x * scale.x, R[0].y * scale.y, R[0].z * scale.z, translation.x);
	object->M[1] = float4(R[1].x * scale.x, R[1].y * scale.y, R[1].z * scale.z, translation.y);
	object->M[2] = float4(R[2].x * scale.x, R[2].y * scale.y, R[2].z * scale.z, translation.z);
	object->M[3] = float4(0, 0, 0, 1);
	calcInvM(object);
}

void Identity(cl_float4(&M)[4]) {
	M[0] = float4(1, 0, 0, 0);
	M[1] = float4(0, 1, 0, 0);
	M[2] = float4(0, 0, 1, 0);
	M[3] = float4(0, 0, 0, 1);
}

void Lorentz(cl_float4(&M)[4], cl_float3 v) {
	float gamma = 1.0f / sqrt(1.0f - dot(v, v));
	float vSqr = dot(v, v);
	if (vSqr == 0) {
		Identity(M);
	}
	else {
		M[0] = float4(gamma, -v.x * gamma, -v.y * gamma, -v.z * gamma);
		M[1] = float4(-v.x * gamma, (gamma - 1.0f) * v.x * v.x / vSqr + 1.0f, (gamma - 1.0f) * v.x * v.y / vSqr, (gamma - 1.0f) * v.x * v.z / vSqr);
		M[2] = float4(-v.y * gamma, (gamma - 1.0f) * v.y * v.x / vSqr, (gamma - 1.0f) * v.y * v.y / vSqr + 1.0f, (gamma - 1.0f) * v.y * v.z / vSqr);
		M[3] = float4(-v.z * gamma, (gamma - 1.0f) * v.z * v.x / vSqr, (gamma - 1.0f) * v.z * v.y / vSqr, (gamma - 1.0f) * v.z * v.z / vSqr + 1.0f);
	}
}


void MatrixMultiplyLeft(cl_float4(&A)[4], cl_float4 const(&B)[4]) {
	for (int i = 0; i < 4; i++) {
		A[i] = float4(
			dot(A[i], float4(B[0].x, B[1].x, B[2].x, B[3].x)),
			dot(A[i], float4(B[0].y, B[1].y, B[2].y, B[3].y)),
			dot(A[i], float4(B[0].z, B[1].z, B[2].z, B[3].z)),
			dot(A[i], float4(B[0].w, B[1].w, B[2].w, B[3].w))
		);
	}
}

void MatrixMultiplyRight(cl_float4 const (&A)[4], cl_float4 (&B)[4]) {
	cl_float4 out[4];
	for (int i = 0; i < 4; i++) {
		out[i] = float4(
			dot(A[i], float4(B[0].x, B[1].x, B[2].x, B[3].x)),
			dot(A[i], float4(B[0].y, B[1].y, B[2].y, B[3].y)),
			dot(A[i], float4(B[0].z, B[1].z, B[2].z, B[3].z)),
			dot(A[i], float4(B[0].w, B[1].w, B[2].w, B[3].w))
		);
	}
	B[0] = out[0];
	B[1] = out[1];
	B[2] = out[2];
	B[3] = out[3];
}

void setLorentzBoost(Object &object, cl_float3 v) {
	Lorentz(object.Lorentz, v);
	float gamma = 1.0f / sqrt(1.0f - dot(v, v));
	object.InvLorentz[0] = float4(gamma, v.x * gamma, v.y * gamma, v.z * gamma);
	object.InvLorentz[1] = object.Lorentz[1];
	object.InvLorentz[1].x *= -1;
	object.InvLorentz[2] = object.Lorentz[2];
	object.InvLorentz[2].x *= -1;
	object.InvLorentz[3] = object.Lorentz[3];
	object.InvLorentz[3].x *= -1;
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
	*/
	if (!ReadOBJ("models/StanfordBunny.obj", theMesh)) {
		exit(EXIT_FAILURE);
	}
	queue.enqueueWriteBuffer(cl_objects, CL_TRUE, 0, object_count * sizeof(Object), cpu_objects);

	white_point = float3(1, 1, 1);
	ambient = 0.1;
	/*
	cpu_objects[0].textureIndex = textureValues[0];
	cpu_objects[0].textureWidth = textureValues[1];
	cpu_objects[0].textureHeight = textureValues[2];
	*/
	//cpu_objects[0].meshIndex = theMesh.meshIndices[0];



	cl_float3 p0 = float3(2 * sqrt(2.0f) - 20.0f + 3, -4, 2 * sqrt(2.0f) + 2);
	cl_float3 dir = float3(1, 0, 1);
	dir = normalize(dir);

	float cosC = dot(-1 * dir, normalize(p0));
	std::cout << cosC << std::endl;
	float b = magnitude(p0);

	

	
	cpu_objects[0].color = float3(0.2f, 0.2f, 0.2f);
	cpu_objects[0].textureIndex = textureValues[3];
	cpu_objects[0].textureWidth = textureValues[4];
	cpu_objects[0].textureHeight = textureValues[5];
	cpu_objects[0].meshIndex = theMesh.meshIndices[0];
	cpu_objects[0].type = MESH;
	TRS(&cpu_objects[0], p0 + 10 * dir, 0, float3(0, 1, 0), float3(1, 1, 1));
	TRS(&cpu_objects[0], float3(0, -1, 12), 0, float3(0, 1, 0), float3(1, 1, 1));
	

	cpu_objects[object_count - 2].color = white_point;
	cpu_objects[object_count - 2].type = CUBE;
	cpu_objects[object_count - 2].light = true;
	cl_float3 offset = float3(1, 0.5f, -1);
	TRS(&cpu_objects[object_count - 2], float3(0, -3.5f, 16), 0.001f, float3(0, 1, 0), float3(0.5f, 0.5f, 0.5f));

	for (int i = 1; i < object_count - 2; i++) {
		cpu_objects[i].color = float3(i%3==0 ? 1.0f:0.0f, i%3==1?1.0f:0.0f, i%3==2?1.0f:0.0f);
		cpu_objects[i].type = CUBE;
		TRS(&cpu_objects[i], float3(-0.75f*(object_count - 4) + 1.5f*(i-1) + 0.25f, -4.5f, 15), 0.001f, float3(0, 1, 0), float3(0.5f, 0.5f, 0.5f));
		/*
		cpu_objects[i].type = SPHERE;
		float c = magnitude(p0) + 2.0f * (i-1);
		float a = b * cosC + sqrt(-b * b + c * c + b * b*cosC*cosC);
		TRS(&cpu_objects[i], p0 + dir*a, 0, float3(0, 1, 0), float3(1, 1, 1));
		std::cout << (p0 + dir * a).x << ", " << (float)(p0 + dir * a).y << ", " << (p0 + dir * a).z << std::endl;
		*/
	}
	TRS(&cpu_objects[object_count - 3], float3(-0.75f*(object_count - 4) + 1.5f*(object_count - 4) + 0.25f, -4.5f, 15), 0.001f, float3(0, 1, 0), float3(0.001f, 0.5f, 0.5f));

	cpu_objects[object_count-1].color = float3(0.9f, 0.8f, 0.7f);
	cpu_objects[object_count-1].type = CUBE;
	
	TRS(&cpu_objects[object_count-1], float3(0, -5.1f, 20), 0.01f, float3(0, 1, 0), float3(40, 0.1f, 40));
	//offset = float3(1, -5.1f, -1);
	//TRS(&cpu_objects[object_count - 1], p0 + offset + 200 * dir, 0, float3(0, 1, 0), float3(40, 0.1f, 40));

	/*

	cpu_objects[object_count/2].color = float3(169 / 255.0f, 168 / 255.0f, 54 / 255.0f);
	cpu_objects[object_count/2].type = SPHERE;
	//cpu_objects[object_count/2].meshIndex = theMesh.meshIndices[0];
	TRS(&cpu_objects[object_count/2], float3(-5 + 3, 2, 10), 3.1415926f / 4, float3(0, 1, 0), float3(1, 1, 1));
	setLorentzBoost(&cpu_objects[object_count/2], float3(-0.999f / sqrt(2.0f), 0, -0.999f / sqrt(2.0f)));
	for (int i = object_count / 2 + 1; i < object_count; i++) {
		cpu_objects[i].color = float3(i % 3 == 1 ? 1.0f : 0.0f, i % 3 == 2 ? 1.0f : 0.0f, i % 3 == 0 ? 1.0f : 0.0f);
		cpu_objects[i].type = SPHERE;
		TRS(&cpu_objects[i], float3(2 * sqrt(2.0f)*(i-object_count/2) - 10.0f + 3, 2, 2 * sqrt(2.0f)*(i - object_count / 2)+5), 3.1415926f / 4, float3(0, 1, 0), float3(1, 1, 1));
	}



	/*
	// left wall
	cpu_objects[0].color = float3(0.75ff, 0.25ff, 0.25ff);
	cpu_objects[0].type = CUBE;
	TRS(&cpu_objects[0], float3(60, 0, 10), 0, float3(0, 1, 0), float3(0.1ff, 10, 10));

	// right wall
	cpu_objects[1].color = float3(0.25ff, 0.25ff, 0.75ff);
	cpu_objects[1].type = CUBE;
	TRS(&cpu_objects[1], float3(60, 0, 10), 0, float3(0, 1, 0), float3(0.1ff, 10, 10));

	// floor
	cpu_objects[2].color = float3(0.25ff, 0.75ff, 0.25ff);
	cpu_objects[2].type = CUBE;
	TRS(&cpu_objects[2], float3(0, -6, 10), 0, float3(0, 1, 0), float3(10, 0.1ff, 10));

	// ceiling
	cpu_objects[3].color = float3(0.9f, 0.8f, 0.7f);
	cpu_objects[3].type = CUBE;
	TRS(&cpu_objects[3], float3(0, 6, 10), 0, float3(0, 1, 0), float3(10, 0.1ff, 10));

	// cube
	cpu_objects[4].color = float3(0.9ff, 0.8ff, 0.7ff);
	cpu_objects[4].type = CUBE;
	cpu_objects[4].textureIndex = textureValues[6];
	cpu_objects[4].textureWidth = textureValues[7];
	cpu_objects[4].textureHeight = textureValues[8];
	TRS(&cpu_objects[4], float3(-4, 0, 5), 0.001f, float3(0, 1, 0), float3(1, 1, 1));
	setLorentzBoost(&cpu_objects[4], float3(0.6f, 0, 0));

	// Sphere
	cpu_objects[5].color = float3(1, 1, 1);
	cpu_objects[5].type = SPHERE;
	cpu_objects[5].textureIndex = textureValues[0];
	cpu_objects[5].textureWidth = textureValues[1];
	cpu_objects[5].textureHeight = textureValues[2];
	TRS(&cpu_objects[5], float3(0, -1.5f, 11), 0, float3(0, 1, 0), float3(1, 1, 1));
	setLorentzBoost(&cpu_objects[5], float3(0, 0, 0.9f));

	// Light
	cpu_objects[6].color = white_point;
	cpu_objects[6].type = SPHERE;
	cpu_objects[6].light = true;
	TRS(&cpu_objects[6], float3(0, 5, 10), 0, float3(0, 1, 0), float3(0.1f, 0.1f, 0.1f));
	/*
	// Pear
	cpu_objects[5].color = float3(169/255.0f, 168/255.0f, 54/255.0f);
	cpu_objects[5].type = MESH;
	cpu_objects[5].meshIndex = theMesh.meshIndices[0];

	// Bunny
	cpu_objects[6].color = float3(1.0ff, 0.2ff, 0.9ff);
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
	kernel.setArg(11, window_width);
	kernel.setArg(12, window_height);
	kernel.setArg(13, cl_vbo);
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

void render(){
	
	framenumber++;

	clock_end = std::chrono::high_resolution_clock::now();
	int ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock_end - clock_start).count();
	currTime = ms / 1000.0f;
	int frame_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock_end - clock_prev).count();
	std::cout << 1000.0f / frame_ms << " fps\taverage: " << 1000.0f*framenumber / ms << std::endl;
	clock_prev = std::chrono::high_resolution_clock::now();

	int new_window_width = glutGet(GLUT_WINDOW_WIDTH),
	    new_window_height = glutGet(GLUT_WINDOW_HEIGHT);
	if (new_window_width != window_width || new_window_height != window_height) {
		window_width = new_window_width;
		window_height = new_window_height;

		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluOrtho2D(0.0f, window_width, 0.0f, window_height);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		unsigned int size = window_width * window_height * sizeof(cl_float3);
		glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
		cl_vbo = cl::BufferGL(context, CL_MEM_WRITE_ONLY, vbo);
		cl_vbos[0] = cl_vbo;
		kernel.setArg(11, window_width);
		kernel.setArg(12, window_height);
		kernel.setArg(13, cl_vbo);
	}
	float s = ms / 1000.0f;

	cl_float4 cameraLorentz[4];
	cl_float4 cameraInvLorentz[4];
	cl_float3 v = float3((float)sqrt(3) / 2.0f, 0, 0);
	cl_float4 cameraPos = float4(currTime, 0, 0, 0);
	Lorentz(cameraLorentz, v);
	Lorentz(cameraInvLorentz, -v);

	for (int i = 0; i < object_count; i++) {
		Identity(cpu_objects[i].Lorentz);
		Identity(cpu_objects[i].InvLorentz);
	}

	cl_float3 dir = float3(1, 0, 1);
	dir = normalize(dir);
	setLorentzBoost(cpu_objects[0], float3(0.99f, 0, 0));
	setLorentzBoost(cpu_objects[object_count - 2], float3((float)sqrt(3) / 2.0f, 0, 0));

	for (int i = 0; i < object_count; i++) {
		MatrixMultiplyLeft(cpu_objects[i].Lorentz, cameraInvLorentz);
		MatrixMultiplyRight(cameraLorentz, cpu_objects[i].InvLorentz);
		cpu_objects[i].stationaryCam = float4(
			dot(cpu_objects[i].Lorentz[0], cameraPos),
			dot(cpu_objects[i].Lorentz[1], cameraPos),
			dot(cpu_objects[i].Lorentz[2], cameraPos),
			dot(cpu_objects[i].Lorentz[3], cameraPos)
		);
		/*
		std::cout << "{{" << cpu_objects[i].Lorentz[0].x << "," << cpu_objects[i].Lorentz[0].y << "," << cpu_objects[i].Lorentz[0].z << "," << cpu_objects[i].Lorentz[0].w << "},{"
			<< cpu_objects[i].Lorentz[1].x << "," << cpu_objects[i].Lorentz[1].y << "," << cpu_objects[i].Lorentz[1].z << "," << cpu_objects[i].Lorentz[1].w << "},{"
			<< cpu_objects[i].Lorentz[2].x << "," << cpu_objects[i].Lorentz[2].y << "," << cpu_objects[i].Lorentz[2].z << "," << cpu_objects[i].Lorentz[2].w << "},{"
			<< cpu_objects[i].Lorentz[3].x << "," << cpu_objects[i].Lorentz[3].y << "," << cpu_objects[i].Lorentz[3].z << "," << cpu_objects[i].Lorentz[3].w << "}}" << std::endl;
		
		std::cout << "{{" << cpu_objects[i].InvLorentz[0].x << "," << cpu_objects[i].InvLorentz[0].y << "," << cpu_objects[i].InvLorentz[0].z << "," << cpu_objects[i].InvLorentz[0].w << "},{"
			<< cpu_objects[i].InvLorentz[1].x << "," << cpu_objects[i].InvLorentz[1].y << "," << cpu_objects[i].InvLorentz[1].z << "," << cpu_objects[i].InvLorentz[1].w << "},{"
			<< cpu_objects[i].InvLorentz[2].x << "," << cpu_objects[i].InvLorentz[2].y << "," << cpu_objects[i].InvLorentz[2].z << "," << cpu_objects[i].InvLorentz[2].w << "},{"
			<< cpu_objects[i].InvLorentz[3].x << "," << cpu_objects[i].InvLorentz[3].y << "," << cpu_objects[i].InvLorentz[3].z << "," << cpu_objects[i].InvLorentz[3].w << "}}" << std::endl;
		*/
	}

	queue.enqueueWriteBuffer(cl_objects, CL_TRUE, 0, object_count * sizeof(Object), cpu_objects);

	kernel.setArg(0, cl_objects);

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

	cl_vertices = cl::Buffer(context, CL_MEM_READ_ONLY, theMesh.vertices.size() * sizeof(cl_float3));
	queue.enqueueWriteBuffer(cl_vertices, CL_TRUE, 0, theMesh.vertices.size() * sizeof(cl_float3), theMesh.vertices.size() > 0 ? &theMesh.vertices[0] : NULL);

	cl_normals = cl::Buffer(context, CL_MEM_READ_ONLY, theMesh.normals.size() * sizeof(cl_float3));
	queue.enqueueWriteBuffer(cl_normals, CL_TRUE, 0, theMesh.normals.size() * sizeof(cl_float3), theMesh.normals.size() > 0 ? &theMesh.normals[0] : NULL);
	
	cl_uvs = cl::Buffer(context, CL_MEM_READ_ONLY, theMesh.uvs.size() * sizeof(cl_float2));
	queue.enqueueWriteBuffer(cl_uvs, CL_TRUE, 0, theMesh.uvs.size() * sizeof(cl_float2), theMesh.uvs.size() > 0 ? &theMesh.uvs[0] : NULL);

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
