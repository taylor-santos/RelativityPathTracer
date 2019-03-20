#include "Mesh.h"
#include <fstream>
#include <iostream>

void Mesh::GenerateOctree(int firstTriIndex) {
	Octree newOctree;
	newOctree.trisCount = 0;
	newOctree.trisIndex = octreeTris.size();
	newOctree.min = vertices[triangles[firstTriIndex]];
	newOctree.max = vertices[triangles[firstTriIndex]];
	for (int i = firstTriIndex / 3; i < triangles.size() / 3; i++) {
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

	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	cl::Platform platform = platforms[0];

	std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	cl::Device device = devices[0];

	cl::Context context = cl::Context(device);

	std::ifstream file("octree_kernel.cl");
	if (!file) {
		std::cout << "\nNo OpenCL file found!" << std::endl << "Exiting..." << std::endl;
		system("PAUSE");
		exit(1);
	}
	std::string source{ std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>() };

	const char* kernel_source = source.c_str();

	cl::Program program = cl::Program(context, kernel_source);
	program.build({ device }, "");

	cl::Kernel kernel = cl::Kernel(program, "parallel_add");

	cl::Buffer vertBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vertices.size() * sizeof(cl_float3), &vertices[0]);
	cl::Buffer triBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, triangles.size() * sizeof(unsigned int), &triangles[0]);
	kernel.setArg(0, vertBuffer);
	kernel.setArg(1, triBuffer);

	Subdivide(*this, octreeIndex, 0, 1, context, kernel, device);
}