#include "Mesh.h"
#include <iostream>
#include <chrono>

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

	auto clock_start = std::chrono::high_resolution_clock::now();
	Subdivide(*this, octreeIndex, 0, 6);
	auto clock_end = std::chrono::high_resolution_clock::now();
	int ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock_end - clock_start).count();
	std::cout << "Octree generated in " << ms << "ms" << std::endl;
}