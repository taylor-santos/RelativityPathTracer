/* OpenCL based simple sphere path tracer by Sam Lapere, 2016*/
/* based on smallpt by Kevin Beason */
/* http://raytracey.blogspot.com */


__constant double EPSILON = 0.0000000001; /* required to compensate for limited float precision */
__constant int MSAASAMPLES = 1;
#define MAX_OCTREE_DEPTH 8;

typedef struct Ray{
	double3 origin;
	double3 dir;
} Ray;

enum objectType { SPHERE, CUBE };

typedef struct Object {
	double4 M[4];
	double4 InvM[4];
	float3 color;
	enum objectType type;
} Object;

typedef struct Hit {
	double dist;
	double3 normal;
	double2 uv;
	float3 color;
} Hit;

typedef struct Octree {
	double3 min;
	double3 max;
	int trisIndex,
		trisCount;
	int children[8];
	int neighbors[6];
} Octree;

Ray createCamRay(const double x_coord, const double y_coord, const int width, const int height){

	double fx = (double)x_coord / (double)width;  /* convert int in range [0 - width] to float in range [0-1] */
	double fy = (double)y_coord / (double)height; /* convert int in range [0 - height] to float in range [0-1] */

	/* calculate aspect ratio */
	double aspect_ratio = (double)(width) / (double)(height);
	double fx2 = (fx - 0.5) * aspect_ratio;
	double fy2 = fy - 0.5;

	/* determine position of pixel on screen */
	double3 pixel_pos = (double3)(fx2, fy2, 1.0);

	/* create camera ray*/
	Ray ray;
	ray.origin = (double3)(0, 0, 0); /* fixed camera position */
	ray.dir = normalize(pixel_pos); /* vector from camera to pixel on screen */
	return ray;
}

double3 transformPoint(const double4 M[4], const double3 v) {
	double4 V = (double4)(v, 1.0);
	return (double3)(
		dot(M[0], V),
		dot(M[1], V),
		dot(M[2], V)
	);
}

double3 transformDirection(const double4 M[4], const double3 v) {
	return (double3)(
		dot(M[0].xyz, v),
		dot(M[1].xyz, v),
		dot(M[2].xyz, v)
	);
}


double3 applyTranspose(const double4 M[4], const double3 v) {
	return M[0].xyz * v.xxx + M[1].xyz * v.yyy + M[2].xyz * v.zzz;
}


bool intersect_triangle(const double3 A, const double3 B, const double3 C, const Ray *ray, Hit *hit) {
	/*
	https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
    */
	double3 v0v1 = B - A;
	double3 v0v2 = C - A;
	double3 pvec = cross(ray->dir, v0v2);
	double det = dot(v0v1, pvec);
	if (det < EPSILON) return false;

	double invDet = 1 / det;

	double3 tvec = ray->origin - A;
	double u = dot(tvec, pvec) * invDet;
	if (u < 0 || u > 1) return false;

	double3 qvec = cross(tvec, v0v1);
	double v = dot(ray->dir, qvec) * invDet;
	if (v < 0 || u + v > 1) return false;

	double t = dot(v0v2, qvec) * invDet;
	hit->dist = t;
	hit->uv = (double2)(u, v);
	return true;
}

bool intersect_AABB(const double3 bounds[2], const Ray *ray, double2 *d, int *closeSide, int *farSide) {
	double3 origin = ray->origin;
	double3 dir = ray->dir;
	double3 inv_dir = 1.0 / dir;
	int sign[3] = {
		inv_dir.x < 0 ? 1 : 0,
		inv_dir.y < 0 ? 1 : 0,
		inv_dir.z < 0 ? 1 : 0
	};
	d->s0 = (bounds[sign[0]].x - origin.x) * inv_dir.x;
	d->s1 = (bounds[1 - sign[0]].x - origin.x) * inv_dir.x;
	*closeSide = 2 + sign[0];
	*farSide = 3 - sign[0];
	double tymin = (bounds[sign[1]].y - origin.y) * inv_dir.y;
	double tymax = (bounds[1 - sign[1]].y - origin.y) * inv_dir.y;
	if ((d->s0 > tymax) || (tymin > d->s1)) {
		return false;
	}
	if (tymin > d->s0) {
		d->s0 = tymin;
		*closeSide = 4 + sign[1];
	}
	if (tymax < d->s1) {
		d->s1 = tymax;
		*farSide = 5 - sign[1];
	}
	double tzmin = (bounds[sign[2]].z - origin.z) * inv_dir.z;
	double tzmax = (bounds[1 - sign[2]].z - origin.z) * inv_dir.z;
	if ((d->s0 > tzmax) || (tzmin > d->s1)) {
		return false;
	}
	if (tzmin > d->s0) {
		d->s0 = tzmin;
		*closeSide = sign[2];
	}
	if (tzmax < d->s1) {
		d->s1 = tzmax;
		*farSide = 1 - sign[2];
	}

	return d->s1 > 0;
}

bool intersect_Octree(global const Octree *octrees, const Ray *ray, Hit *hit) {
	int octreeStack[8];
	int childStack[8];
	int stackIndex = 0;
	octreeStack[stackIndex] = 0;
	childStack[stackIndex] = 0;
	Octree curr = octrees[0];
	while (1) {
		if (curr.children[0] != -1) {
			stackIndex++;
			octreeStack[stackIndex] = curr.children[0];
			childStack[stackIndex] = 0;
			curr = octrees[octreeStack[stackIndex]];
		}
		else {
			childStack[stackIndex]++;
			while (childStack[stackIndex] >= 8) {
				stackIndex--;
				if (stackIndex < 0) break;
				childStack[stackIndex]++;
			}
			if (stackIndex < 0) break;
			curr = octrees[octrees[octreeStack[stackIndex]].children[childStack[stackIndex]]];
		}
	}
	return false;
}

bool intersect_cube(global const Object *objects, int index, const Ray *ray, Hit *hit) {
	double3 origin = transformPoint(objects[index].InvM, ray->origin);
	double3 dir = normalize(transformDirection(objects[index].InvM, ray->dir));
	double3 dir_inv = 1.0 / dir;
	double t1 = (-1.0 - origin.x) * dir_inv.x;
	double t2 = ( 1.0 - origin.x) * dir_inv.x;
	double tmin = min(t1, t2);
	double tmax = max(t1, t2);
	t1 = (-1.0 - origin.y) * dir_inv.y;
	t2 = ( 1.0 - origin.y) * dir_inv.y;
	tmin = max(tmin, min(min(t1, t2), tmax));
	tmax = min(tmax, max(max(t1, t2), tmin));
	t1 = (-1.0 - origin.z) * dir_inv.z;
	t2 = ( 1.0 - origin.z) * dir_inv.z;
	tmin = max(tmin, min(min(t1, t2), tmax));
	tmax = min(tmax, max(max(t1, t2), tmin));
	if (tmin > 0.0 && tmax > tmin) {
		double3 pt = origin + dir * tmin;
		double3 normal = (double3)(0, 0, 0);
		if (pt.x*pt.x - 1 < EPSILON && pt.x*pt.x - 1 > -EPSILON) {
			normal.x = round(pt.x);
		}else if (pt.y*pt.y - 1 < EPSILON && pt.y*pt.y - 1 > -EPSILON) {
			normal.y = round(pt.y);
		}
		else if (pt.z*pt.z - 1 < EPSILON && pt.z*pt.z - 1 > -EPSILON) {
			normal.z = round(pt.z);
		}
		double3 worldPt = transformPoint(objects[index].M, pt);
		hit->dist = length(worldPt);
		hit->normal = normalize(applyTranspose(objects[index].InvM, normal));
		hit->color = objects[index].color;
		return true;
	}
	return false;
}

bool intersect_sphere(global const Object *objects, const int index, const Ray *ray, Hit *hit) {
	double3 rayToSphere = -transformPoint(objects[index].InvM, ray->origin);
	double3 dir = normalize(transformDirection(objects[index].InvM, ray->dir));
	double b = dot(rayToSphere, dir);
	double c = dot(rayToSphere, rayToSphere) - 1.0;
	double disc = b * b - c;
	if (disc < 0.0) return false;
	else disc = sqrt(disc);
	double dist;
	if ((b - disc) > EPSILON) {
		dist = b - disc;
	} else if ((b + disc) > EPSILON) {
		dist = b + disc;
	} else {
		dist = 0.0;
	}
	double3 objPt = -rayToSphere + dir * dist;
	double3 worldPt = transformPoint(objects[index].M, objPt);
	hit->dist = length(worldPt);
	hit->normal = normalize(applyTranspose(objects[index].InvM, objPt));
	hit->color = objects[index].color;
	return true;
}

bool intersect_mesh(
	global const Object* objects,
	const int index,
	global const double3 *vertices,
	global const double3 *normals,
	global const unsigned int *triangles,
	const unsigned int face_count,
	global const Octree *octrees,
	global const int *octreeTris,
	const Ray *ray,
	Hit *hit
) {
	Ray newRay;
	newRay.origin = transformPoint(objects[index].InvM, ray->origin);
	newRay.dir = normalize(transformDirection(objects[index].InvM, ray->dir));

	return intersect_Octree(octrees, &newRay, hit);

	double3 bounds[2] = { octrees[0].min, octrees[0].max };
	double2 d;
	int closeSide, farSide;

	if (intersect_AABB(bounds, &newRay, &d, &closeSide, &farSide)) {
		double3 extents = octrees[0].max - octrees[0].min;
		double3 u1 = (newRay.origin + d.s0 * newRay.dir - bounds[0]) / extents;
		double3 u2 = (newRay.origin + d.s1 * newRay.dir - bounds[0]) / extents;
		
		


		
		Hit newHit;
		newHit.color = objects[index].color;
		bool didHit = false;
		for (int i = octrees[0].trisIndex; i < octrees[0].trisIndex + octrees[0].trisCount; i++) {
			int tri = octreeTris[i];
			double3 A = vertices[triangles[9 * i + 3 * 0]];
			double3 B = vertices[triangles[9 * i + 3 * 1]];
			double3 C = vertices[triangles[9 * i + 3 * 2]];

			if (intersect_triangle(A, B, C, &newRay, &newHit)) {
				double3 objPt = newRay.origin + newRay.dir * newHit.dist;
				double3 worldPt = transformPoint(objects[index].M, objPt);
				newHit.dist = length(worldPt);
				if (newHit.dist > 0.0f && newHit.dist < hit->dist) {
					double3 normA = normals[triangles[2 + 9 * i + 3 * 0]];
					double3 normB = normals[triangles[2 + 9 * i + 3 * 1]];
					double3 normC = normals[triangles[2 + 9 * i + 3 * 2]];
					double u = newHit.uv.s0;
					double v = newHit.uv.s1;
					newHit.normal = (1.0 - u - v)*normA + u * normB + v * normC;
					*hit = newHit;
					didHit = true;
				}
			}
		}
		if (didHit) {
			hit->normal = normalize(applyTranspose(objects[index].InvM, hit->normal));
			return true;
		}
	}
	return false;
}

bool intersect_scene(
	global const Object* objects,
	const int object_count,
	global const double3 *vertices,
	global const double3 *normals,
	global const unsigned int *triangles,
	const unsigned int face_count,
	global const Octree *octrees,
	global const int *octreeTris,
	const Ray *ray,
	Hit *hit
) {
	/* initialise t to a very large number,
	so t will be guaranteed to be smaller
	when a hit with the scene occurs */

	float inf = 1e20;
	hit->dist = inf;

	/* check if the ray intersects each object in the scene */
	for (int i = 0; i < object_count; i++) {
		Hit newHit;
		newHit.dist = inf;
		if (i == 6) {
			if (intersect_mesh(objects, i, vertices, normals, triangles, face_count, octrees, octreeTris, ray, &newHit)) {
				if (newHit.dist > 0.0f && newHit.dist < hit->dist) {
					*hit = newHit;
					break;
				}
			}
		}
		else {
			switch (objects[i].type) {
			case SPHERE:
				if (intersect_sphere(objects, i, ray, &newHit)) {
					if (newHit.dist > 0.0f && newHit.dist < hit->dist) {
						*hit = newHit;
					}
				}
				break;
			case CUBE:
				if (intersect_cube(objects, i, ray, &newHit)) {
					if (newHit.dist > 0.0f && newHit.dist < hit->dist) {
						*hit = newHit;
					}
				}
				break;
			}
		}
	}
	return hit->dist < inf; /* true when ray interesects the scene */
}


/* the path tracing function */
/* computes a path (starting from the camera) with a defined number of bounces, accumulates light/color at each bounce */
/* each ray hitting a surface will be reflected in a random direction (by randomly sampling the hemisphere above the hitpoint) */
/* small optimisation: diffuse ray directions are calculated using cosine weighted importance sampling */

float3 trace(
	global const Object* objects,
	const int object_count,
	global const double3 *vertices,
	global const double3 *normals,
	global const unsigned int *triangles,
	const unsigned int face_count,
	global const Octree *octrees,
	global const int *octreeTris,
	const Ray* camray
) {
	float3 accum_color = (float3)(0.0f, 0.0f, 0.0f);
	float3 mask = (float3)(1.0f, 1.0f, 1.0f);

	float t;   /* distance to intersection */

	/* if ray misses scene, return background colour */
	Hit hit;
	if (!intersect_scene(objects, object_count, vertices, normals, triangles, face_count, octrees, octreeTris, camray, &hit))
		return accum_color += mask * (float3)(0.15f, 0.15f, 0.25f);

	/* compute the hitpoint using the ray equation */
	double3 hitpoint = camray->origin + camray->dir * hit.dist;

	/* compute the surface normal and flip it if necessary to face the incoming ray */
	double3 normal = hit.normal;

	float3 color = convert_float3(normal);
	color = (color + (float3)(1, 1, 1)) / 2;
	return color * hit.color;
	
}

union Colour { float c; uchar4 components; };

__kernel void render_kernel(
	global const Object* objects,
	const int object_count,
	global const double3 *vertices,
	global const double3 *normals,
	global const int *triangles,
	const int face_count,
	global const Octree *octrees,
	global const int *octreeTris,
	const int width,
	const int height,
	__global float3* output
) {
	unsigned int work_item_id = get_global_id(0);	/* the unique global id of the work item for the current pixel */
	unsigned int x_coord = work_item_id % width;			/* x-coordinate of the pixel */
	unsigned int y_coord = work_item_id / width;			/* y-coordinate of the pixel */
	
	float3 finalcolor = (float3)(0, 0, 0);

	for (int y = 0; y < MSAASAMPLES; y++) {
		for (int x = 0; x < MSAASAMPLES; x++) {
			Ray camray = createCamRay((double)x_coord + (double)x/MSAASAMPLES, (double)y_coord + (double)y/ MSAASAMPLES, width, height);

			finalcolor += trace(objects, object_count, vertices, normals, triangles, face_count, octrees, octreeTris, &camray);
		}
	}
	finalcolor = finalcolor / (MSAASAMPLES*MSAASAMPLES);


	union Colour fcolour;
	fcolour.components = (uchar4)(	
		(unsigned char)(finalcolor.x * 255), 
		(unsigned char)(finalcolor.y * 255),
		(unsigned char)(finalcolor.z * 255),
		1);

	/* store the pixelcolour in the output buffer */
	output[work_item_id] = (float3)(x_coord, y_coord, fcolour.c);
}
