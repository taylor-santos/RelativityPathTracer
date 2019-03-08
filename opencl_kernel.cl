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

private int getChildIndex(int side, double3 *uv) {
	switch (side) {
	case 0:
		if (uv->x < 0.5) {
			uv->x *= 2;
			if (uv->y < 0.5) {
				uv->y *= 2;
				return 0;
			}
			else {
				uv->y = 2 * uv->y - 1;
				return 2;
			}
		}
		else {
			uv->x = 2 * uv->x - 1;
			if (uv->y < 0.5) {
				uv->y *= 2;
				return 4;
			}
			else {
				uv->y = 2 * uv->y - 1;
				return 6;
			}
		}
	case 1:
		if (uv->x < 0.5) {
			uv->x *= 2;
			if (uv->y < 0.5) {
				uv->y *= 2;
				return 1;
			}
			else {
				uv->y = 2 * uv->y - 1;
				return 3;
			}
		}
		else {
			uv->x = 2 * uv->x - 1;
			if (uv->y < 0.5) {
				uv->y *= 2;
				return 5;
			}
			else {
				uv->y = 2 * uv->y - 1;
				return 7;
			}
		}
	case 2:
		if (uv->y < 0.5) {
			uv->y *= 2;
			if (uv->z < 0.5) {
				uv->z *= 2;
				return 0;
			}
			else {
				uv->z = 2 * uv->z - 1;
				return 1;
			}
		}
		else {
			uv->y = 2 * uv->y - 1;
			if (uv->z < 0.5) {
				uv->z *= 2;
				return 2;
			}
			else {
				uv->z = 2 * uv->z - 1;
				return 3;
			}
		}
	case 3:
		if (uv->y < 0.5) {
			uv->y *= 2;
			if (uv->z < 0.5) {
				uv->z *= 2;
				return 4;
			}
			else {
				uv->z = 2 * uv->z - 1;
				return 5;
			}
		}
		else {
			uv->y = 2 * uv->y - 1;
			if (uv->z < 0.5) {
				uv->z *= 2;
				return 6;
			}
			else {
				uv->z = 2 * uv->z - 1;
				return 7;
			}
		}
	case 4:
		if (uv->x < 0.5) {
			uv->x *= 2;
			if (uv->z < 0.5) {
				uv->z *= 2;
				return 0;
			}
			else {
				uv->z = 2 * uv->z - 1;
				return 1;
			}
		}
		else {
			uv->x = 2 * uv->x - 1;
			if (uv->z < 0.5) {
				uv->z *= 2;
				return 4;
			}
			else {
				uv->z = 2 * uv->z - 1;
				return 5;
			}
		}
	case 5:
		if (uv->x < 0.5) {
			uv->x *= 2;
			if (uv->z < 0.5) {
				uv->z *= 2;
				return 2;
			}
			else {
				uv->z = 2 * uv->z - 1;
				return 3;
			}
		}
		else {
			uv->x = 2 * uv->x - 1;
			if (uv->z < 0.5) {
				uv->z *= 2;
				return 6;
			}
			else {
				uv->z = 2 * uv->z - 1;
				return 7;
			}
		}
	default:
		return -1;
	}
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

int getOppositeBoxSide(const double3 scaledDir, const int closeSide, double3 *uv) {
	double3 inv_dir = 1.0 / scaledDir;
	int sign[3] = { inv_dir.x < 0, inv_dir.y < 0, inv_dir.z < 0 };
	double dx = (1 - sign[0] - uv->x) * inv_dir.x;
	double dy = (1 - sign[1] - uv->y) * inv_dir.y;
	double dz = (1 - sign[2] - uv->z) * inv_dir.z;
	if (dx < dy) {
		if (dx < dz) { /* dx < dz && dx < dy */
			*uv += scaledDir * dx;
			return 3 - sign[0];
		}
		else { /* dz <= dx < dy */
			*uv += scaledDir * dz;
			return 1 - sign[2];
		}
	}
	else { /* dy <= dx */
		if (dy < dz) {
			*uv += scaledDir * dy;
			return 5 - sign[1];
		}
		else { /* dz <= dy <= dx */
			*uv += scaledDir * dz;
			return 1 - sign[2];
		}
	}
}

bool intersect_Octree(
	global const double3 *vertices,
	global const double3 *normals,
	global const unsigned int *triangles,
	global const Octree *octrees,
	global const int *octreeTris,
	const Ray *ray,
	Hit *hit
) {
	int currOctreeIndex = 0;
	double2 d;
	int closeSide, farSide;
	double3 bounds[2] = {
		octrees[currOctreeIndex].min,
		octrees[currOctreeIndex].max
	};
	bool didHit = false;
	if (!intersect_AABB(bounds, ray, &d, &closeSide, &farSide)) {
		return false;
	}
	double3 uv = ray->origin + ray->dir * d.s0;
	double3 scaledDir = normalize(ray->dir / (octrees[currOctreeIndex].max - octrees[currOctreeIndex].min));
	while(currOctreeIndex != -1) {
		double3 extents = octrees[currOctreeIndex].max - octrees[currOctreeIndex].min;
		uv = (uv - octrees[currOctreeIndex].min) / extents;
		while (octrees[currOctreeIndex].children[0] != -1) {
			int childIndex = getChildIndex(closeSide, &uv);
			currOctreeIndex = octrees[currOctreeIndex].children[childIndex];
		}
		for (
			int i = octrees[currOctreeIndex].trisIndex;
			i < octrees[currOctreeIndex].trisIndex + octrees[currOctreeIndex].trisCount;
			i++
		) {
			int tri = octreeTris[i];
			double3 A = vertices[triangles[9 * tri + 3 * 0]];
			double3 B = vertices[triangles[9 * tri + 3 * 1]];
			double3 C = vertices[triangles[9 * tri + 3 * 2]];
			Hit newHit;
			if (intersect_triangle(A, B, C, ray, &newHit)) {
				if (newHit.dist > 0.0f && newHit.dist < hit->dist) {
					double3 normA = normals[triangles[2 + 9 * tri + 3 * 0]];
					double3 normB = normals[triangles[2 + 9 * tri + 3 * 1]];
					double3 normC = normals[triangles[2 + 9 * tri + 3 * 2]];
					double u = newHit.uv.s0;
					double v = newHit.uv.s1;
					newHit.normal = (1.0 - u - v)*normA + u * normB + v * normC;
					newHit.color = convert_float3((newHit.normal + 1) / 2);
					*hit = newHit;
					didHit = true;
				}
			}
		}
		extents = octrees[currOctreeIndex].max - octrees[currOctreeIndex].min;
		farSide = getOppositeBoxSide(scaledDir, closeSide, &uv);
		closeSide = farSide - 2 * (farSide % 2) + 1;
		uv = octrees[currOctreeIndex].min + uv * extents;
		currOctreeIndex = octrees[currOctreeIndex].neighbors[farSide];
		if (length(uv - ray->origin) > hit->dist) {
			break;
		}
	}

	return didHit;
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
	if (intersect_Octree(vertices, normals, triangles, octrees, octreeTris, &newRay, hit)) {
		double3 objPoint = newRay.origin + hit->dist * newRay.dir;
		double3 worldPoint = transformPoint(objects[index].M, objPoint);
		hit->dist = length(worldPoint - ray->origin);
		hit->normal = normalize(applyTranspose(objects[index].InvM, hit->normal));
		return true;
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
				if (newHit.dist < hit->dist) {
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
	Hit hit;
	if (!intersect_scene(objects, object_count, vertices, normals, triangles, face_count, octrees, octreeTris, camray, &hit))
		return (float3)(0.15f, 0.15f, 0.25f);

	double3 hitpoint = camray->origin + camray->dir * hit.dist;
	double3 normal = hit.normal;

	float3 color = hit.color * 0.2f;
	double3 light_pos = (double3)(0, 0.25, 0);
	double3 light_dir = light_pos - hitpoint;
	float light_intensity = 50;
	if (dot(light_dir, normal) > 0) {
		color += (float)(dot(normalize(light_dir), normal) / (dot(light_dir, light_dir))) * hit.color * light_intensity;
	}
	return color;
	
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
		(unsigned char)(min(finalcolor.x, 1.0f) * 255), 
		(unsigned char)(min(finalcolor.y, 1.0f) * 255),
		(unsigned char)(min(finalcolor.z, 1.0f) * 255),
		1);

	/* store the pixelcolour in the output buffer */
	output[work_item_id] = (float3)(x_coord, y_coord, fcolour.c);
}
