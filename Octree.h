#pragma once
#include <CL/cl.hpp>

struct Octree
{
	cl_float3 min;
	cl_float3 max;
	int trisIndex,
		trisCount;
	int children[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
	int neighbors[6] = { -1, -1, -1, -1, -1, -1 };
};

bool AABBTriangleIntersection(struct Mesh const& mesh, int octreeIndex, int triIndex);
void Subdivide(Mesh &mesh, int octreeIndex, int minTris, int depth, cl::Context &context, cl::Kernel &kernel, cl::Device &device);