#include "Octree.h"
#include "Mesh.h"
#include <map>
#include <windows.h>

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
					if (AABBTriangleIntersection(mesh, mesh.octree.size() - 1, triIndex)) {
						mesh.octreeTris.push_back(triIndex);
						mesh.octree[mesh.octree.size() - 1].trisCount++;
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