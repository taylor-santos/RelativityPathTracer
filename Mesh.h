#pragma once
#include "Octree.h"
#include "Vector.h"

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