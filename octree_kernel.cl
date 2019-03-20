unsigned char AABBTriangleIntersection(__const float3 A, __const float3 B, __const float3 C, __const float3 min, __const float3 max) {
	float3 center = (min + max) / 2;
	float3 extents = (max - min) / 2;

	float3 offsetA = A - center;
	float3 offsetB = B - center;
	float3 offsetC = C - center;

	float3 ba = offsetB - offsetA;
	float3 cb = offsetC - offsetB;

	float3 ba_abs = fabs(ba);
	{
		float min = ba.z * offsetA.y - ba.y * offsetA.z;
		float max = ba.z * offsetC.y - ba.y * offsetC.z;
		if (min > max) {
			float temp = min;
			min = max;
			max = temp;
		}
		float rad = ba_abs.z * extents.y + ba_abs.y * extents.z;
		if (min > rad || max < -rad) return 0;
	}
	{
		float min = -ba.z * offsetA.x + ba.x * offsetA.z;
		float max = -ba.z * offsetC.x + ba.x * offsetC.z;
		if (min > max) {
			float temp = min;
			min = max;
			max = temp;
		}
		float rad = ba_abs.z * extents.x + ba_abs.x * extents.z;
		if (min > rad || max < -rad) return 0;
	}
	{
		float min = ba.y * offsetB.x - ba.x * offsetB.y;
		float max = ba.y * offsetC.x - ba.x * offsetC.y;
		if (min > max) {
			float temp = min;
			min = max;
			max = temp;
		}
		float rad = ba_abs.y * extents.x + ba_abs.x * extents.y;
		if (min > rad || max < -rad) return 0;
	}
	float3 cb_abs = fabs(cb);
	{
		float min = cb.z * offsetA.y - cb.y * offsetA.z,
			max = cb.z * offsetC.y - cb.y * offsetC.z;
		if (min > max) {
			float temp = min;
			min = max;
			max = temp;
		}
		float rad = cb_abs.z * extents.y + cb_abs.y * extents.z;
		if (min > rad || max < -rad) return 0;
	}
	{
		float min = -cb.z * offsetA.x + cb.x * offsetA.z,
			max = -cb.z * offsetC.x + cb.x * offsetC.z;
		if (min > max) {
			float temp = min;
			min = max;
			max = temp;
		}
		float rad = cb_abs.z * extents.x + cb_abs.x * extents.z;
		if (min > rad || max < -rad) return 0;
	}
	{
		float min = cb.y * offsetA.x - cb.x * offsetA.y,
			max = cb.y * offsetB.x - cb.x * offsetB.y;
		if (min > max) {
			float temp = min;
			min = max;
			max = temp;
		}
		float rad = cb_abs.y * extents.x + cb_abs.x * extents.y;
		if (min > rad || max < -rad) return 0;
	}
	float3 ac = offsetA - offsetC;
	float3 ac_abs = fabs(ac);
	{
		
		float min = ac.z * offsetA.y - ac.y * offsetA.z,
			max = ac.z * offsetB.y - ac.y * offsetB.z;
		if (min > max) {
			float temp = min;
			min = max;
			max = temp;
		}
		float rad = ac_abs.z * extents.y + ac_abs.y * extents.z;
		if (min > rad || max < -rad) return 0;
	}
	{
		float min = -ac.z * offsetA.x + ac.x * offsetA.z,
			max = -ac.z * offsetB.x + ac.x * offsetB.z;
		if (min > max) {
			float temp = min;
			min = max;
			max = temp;
		}
		float rad = ac_abs.z * extents.x + ac_abs.x * extents.z;
		if (min > rad || max < -rad) return 0;
	}
	{
		float min = ac.y * offsetB.x - ac.x * offsetB.y,
			max = ac.y * offsetC.x - ac.x * offsetC.y;
		if (min > max) {
			float temp = min;
			min = max;
			max = temp;
		}
		float rad = ac_abs.y * extents.x + ac_abs.x * extents.y;
		if (min > rad || max < -rad) return 0;
	}
	{
		float3 normal = cross(ba, cb);
		float3 min, max;
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
		if (dot(normal, min) > 0) return 0;
		if (dot(normal, max) < 0) return 0;
	}
	{
		
		float3 min = fmin(fmin(offsetA, offsetB), offsetC);
		float3 max = fmax(fmax(offsetA, offsetB), offsetC);
		if (min.x > extents.x || max.x < -extents.x) return 0;
		if (min.y > extents.y || max.y < -extents.y) return 0;
		if (min.z > extents.z || max.z < -extents.z) return 0;
	}
	return 1;
}

__kernel void parallel_add(__global float3* vertices, __global unsigned int* triangles, __global unsigned int* triIndices, __const float3 min, __const float3 max, __global unsigned long* output, __const unsigned int num_triangles) {
	const int i = get_global_id(0);
	if (i >= num_triangles) return;
	const int triIndex = 9*triIndices[i];
	const float3 A = vertices[triangles[triIndex + 0]],
		B = vertices[triangles[triIndex + 3]],
		C = vertices[triangles[triIndex + 6]],
		Center = (min + max) / 2,
		Extent = (max - min) / 2,
		extent = (max - min) / 4;

	unsigned long o0 = 0,
		o1 = 0,
		o2 = 0,
		o3 = 0,
		o4 = 0,
		o5 = 0,
		o6 = 0,
		o7 = 0;
	float3 half_min = min,
		half_max = Center;
	float3 center = (half_min + half_max) / 2;
	o0 = AABBTriangleIntersection(A, B, C, half_min, center) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(0, 0, extent.z), center + (float3)(0, 0, extent.z)) * 2) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(0, extent.y, 0), center + (float3)(0, extent.y, 0)) * 4) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(0, extent.y, extent.z), center + (float3)(0, extent.y, extent.z)) * 8) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(extent.x, 0, 0), center + (float3)(extent.x, 0, 0)) * 16) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(extent.x, 0, extent.z), center + (float3)(extent.x, 0, extent.z)) * 32) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(extent.x, extent.y, 0), center + (float3)(extent.x, extent.y, 0)) * 64) +
		(AABBTriangleIntersection(A, B, C, center, half_max) * 128);
	half_min = min + (float3)(0, 0, Extent.z);
	half_max = Center + (float3)(0, 0, Extent.z);
	center = (half_min + half_max) / 2;
	o1 = AABBTriangleIntersection(A, B, C, half_min, center) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(0, 0, extent.z), center + (float3)(0, 0, extent.z)) * 2) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(0, extent.y, 0), center + (float3)(0, extent.y, 0)) * 4) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(0, extent.y, extent.z), center + (float3)(0, extent.y, extent.z)) * 8) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(extent.x, 0, 0), center + (float3)(extent.x, 0, 0)) * 16) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(extent.x, 0, extent.z), center + (float3)(extent.x, 0, extent.z)) * 32) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(extent.x, extent.y, 0), center + (float3)(extent.x, extent.y, 0)) * 64) +
		(AABBTriangleIntersection(A, B, C, center, half_max) * 128);
	half_min = min + (float3)(0, Extent.y, 0);
	half_max = Center + (float3)(0, Extent.y, 0);
	center = (half_min + half_max) / 2;
	o2 = AABBTriangleIntersection(A, B, C, half_min, center) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(0, 0, extent.z), center + (float3)(0, 0, extent.z)) * 2) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(0, extent.y, 0), center + (float3)(0, extent.y, 0)) * 4) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(0, extent.y, extent.z), center + (float3)(0, extent.y, extent.z)) * 8) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(extent.x, 0, 0), center + (float3)(extent.x, 0, 0)) * 16) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(extent.x, 0, extent.z), center + (float3)(extent.x, 0, extent.z)) * 32) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(extent.x, extent.y, 0), center + (float3)(extent.x, extent.y, 0)) * 64) +
		(AABBTriangleIntersection(A, B, C, center, half_max) * 128);
	half_min = min + (float3)(0, Extent.y, Extent.z);
	half_max = Center + (float3)(0, Extent.y, Extent.z);
	center = (half_min + half_max) / 2;
	o3 = AABBTriangleIntersection(A, B, C, half_min, center) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(0, 0, extent.z), center + (float3)(0, 0, extent.z)) * 2) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(0, extent.y, 0), center + (float3)(0, extent.y, 0)) * 4) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(0, extent.y, extent.z), center + (float3)(0, extent.y, extent.z)) * 8) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(extent.x, 0, 0), center + (float3)(extent.x, 0, 0)) * 16) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(extent.x, 0, extent.z), center + (float3)(extent.x, 0, extent.z)) * 32) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(extent.x, extent.y, 0), center + (float3)(extent.x, extent.y, 0)) * 64) +
		(AABBTriangleIntersection(A, B, C, center, half_max) * 128);
	half_min = min + (float3)(Extent.x, 0, 0);
	half_max = Center + (float3)(Extent.x, 0, 0);
	center = (half_min + half_max) / 2;
	o4 = AABBTriangleIntersection(A, B, C, half_min, center) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(0, 0, extent.z), center + (float3)(0, 0, extent.z)) * 2) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(0, extent.y, 0), center + (float3)(0, extent.y, 0)) * 4) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(0, extent.y, extent.z), center + (float3)(0, extent.y, extent.z)) * 8) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(extent.x, 0, 0), center + (float3)(extent.x, 0, 0)) * 16) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(extent.x, 0, extent.z), center + (float3)(extent.x, 0, extent.z)) * 32) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(extent.x, extent.y, 0), center + (float3)(extent.x, extent.y, 0)) * 64) +
		(AABBTriangleIntersection(A, B, C, center, half_max) * 128);
	half_min = min + (float3)(Extent.x, 0, Extent.z);
	half_max = Center + (float3)(Extent.x, 0, Extent.z);
	center = (half_min + half_max) / 2;
	o5 = AABBTriangleIntersection(A, B, C, half_min, center) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(0, 0, extent.z), center + (float3)(0, 0, extent.z)) * 2) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(0, extent.y, 0), center + (float3)(0, extent.y, 0)) * 4) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(0, extent.y, extent.z), center + (float3)(0, extent.y, extent.z)) * 8) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(extent.x, 0, 0), center + (float3)(extent.x, 0, 0)) * 16) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(extent.x, 0, extent.z), center + (float3)(extent.x, 0, extent.z)) * 32) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(extent.x, extent.y, 0), center + (float3)(extent.x, extent.y, 0)) * 64) +
		(AABBTriangleIntersection(A, B, C, center, half_max) * 128);
	half_min = min + (float3)(Extent.x, Extent.y, 0);
	half_max = Center + (float3)(Extent.x, Extent.y, 0);
	center = (half_min + half_max) / 2;
	o6 = AABBTriangleIntersection(A, B, C, half_min, center) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(0, 0, extent.z), center + (float3)(0, 0, extent.z)) * 2) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(0, extent.y, 0), center + (float3)(0, extent.y, 0)) * 4) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(0, extent.y, extent.z), center + (float3)(0, extent.y, extent.z)) * 8) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(extent.x, 0, 0), center + (float3)(extent.x, 0, 0)) * 16) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(extent.x, 0, extent.z), center + (float3)(extent.x, 0, extent.z)) * 32) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(extent.x, extent.y, 0), center + (float3)(extent.x, extent.y, 0)) * 64) +
		(AABBTriangleIntersection(A, B, C, center, half_max) * 128);
	half_min = Center;
	half_max = max;
	center = (half_min + half_max) / 2;
	o7 = AABBTriangleIntersection(A, B, C, half_min, center) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(0, 0, extent.z), center + (float3)(0, 0, extent.z)) * 2) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(0, extent.y, 0), center + (float3)(0, extent.y, 0)) * 4) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(0, extent.y, extent.z), center + (float3)(0, extent.y, extent.z)) * 8) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(extent.x, 0, 0), center + (float3)(extent.x, 0, 0)) * 16) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(extent.x, 0, extent.z), center + (float3)(extent.x, 0, extent.z)) * 32) +
		(AABBTriangleIntersection(A, B, C, half_min + (float3)(extent.x, extent.y, 0), center + (float3)(extent.x, extent.y, 0)) * 64) +
		(AABBTriangleIntersection(A, B, C, center, half_max) * 128);

	output[i] = o0 + 256 * (o1 + 256 * (o2 + 256 * (o3 + 256 * (o4 + 256 * (o5 + 256 * (o6 + 256 * o7))))));
}