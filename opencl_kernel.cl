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

enum objectType { SPHERE, CUBE, MESH };

typedef struct Object {
	double4 M[4];
	double4 InvM[4];
	float3 color;
	enum objectType type;
	int meshIndex;
	int textureIndex;
	int textureWidth;
	int textureHeight;
} Object;

typedef struct Hit {
	double dist;
	double3 normal;
	double2 uv;
	float3 color;
	int object;
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

bool intersect_triangle(const double3 A, const double3 B, const double3 C, const Ray *ray, double *dist, double2 *uv) {
	/*
	https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
    */
	double3 v0v1 = B - A;
	double3 v0v2 = C - A;
	double3 pvec = cross(ray->dir, v0v2);
	double det = dot(v0v1, pvec);
	if (det < EPSILON && -EPSILON < det) return false;

	double invDet = 1 / det;

	double3 tvec = ray->origin - A;
	uv->s0 = dot(tvec, pvec) * invDet;
	if (uv->s0 < 0 || uv->s0 > 1) return false;

	double3 qvec = cross(tvec, v0v1);
	uv->s1 = dot(ray->dir, qvec) * invDet;
	if (uv->s1 < 0 || uv->s0 + uv->s1 > 1) return false;

	*dist = dot(v0v2, qvec) * invDet;
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

bool intersect_octree(
	global const Object* objects,
	const int index,
	global const double3 *vertices,
	global const double3 *normals,
	global const double2 *uvs,
	global const unsigned int *triangles,
	global const Octree *octrees,
	global const int *octreeTris,
	const Ray *ray,
	Hit *hit
) {
	Ray newRay;
	newRay.origin = transformPoint(objects[index].InvM, ray->origin);
	newRay.dir = normalize(transformDirection(objects[index].InvM, ray->dir));

	int currOctreeIndex = objects[index].meshIndex;
	double2 d;
	int closeSide, farSide;
	double3 bounds[2] = {
		octrees[currOctreeIndex].min,
		octrees[currOctreeIndex].max
	};
	bool didHit = false;
	int hitTri;
	if (!intersect_AABB(bounds, &newRay, &d, &closeSide, &farSide)) {
		return false;
	}
	double3 uv = newRay.origin + newRay.dir * d.s0;

	if (d.s0 < 0) {
		Octree curr = octrees[currOctreeIndex];
		uv = (newRay.origin - curr.min) / (curr.max - curr.min);
		while (curr.children[0] != -1) {
			int childIndex = round(uv.z) + 2 * round(uv.y) + 4 * round(uv.x);
			uv = 2.0 * fmod(min(uv, 1.0 - EPSILON), 0.5);
			currOctreeIndex = curr.children[childIndex];
			curr = octrees[currOctreeIndex];
		}
		bounds[0] = curr.min;
		bounds[1] = curr.max;
		if (!intersect_AABB(bounds, &newRay, &d, &closeSide, &farSide)) {
			return false;
		}
		uv = newRay.origin + newRay.dir * d.s0;
	}

	double3 scaledDir = normalize(newRay.dir / (octrees[currOctreeIndex].max - octrees[currOctreeIndex].min));
	while(currOctreeIndex != -1) {
		Octree curr = octrees[currOctreeIndex];
		double3 extents = curr.max - curr.min;
		uv = (uv - curr.min) / extents;
		while (curr.children[0] != -1) {
			int childIndex = round(uv.z) + 2 * round(uv.y) + 4 * round(uv.x);
			uv = 2.0 * fmod(min(uv, 1.0 - EPSILON), 0.5);
			currOctreeIndex = curr.children[childIndex];
			curr = octrees[currOctreeIndex];
		}
		for (
			int i = curr.trisIndex;
			i < curr.trisIndex + curr.trisCount;
			i++
			) {
			int tri = octreeTris[i];
			double3 A = vertices[triangles[9 * tri + 3 * 0]];
			double3 B = vertices[triangles[9 * tri + 3 * 1]];
			double3 C = vertices[triangles[9 * tri + 3 * 2]];
			double dist;
			double2 triUV;
			if (intersect_triangle(A, B, C, &newRay, &dist, &triUV)) {
				if (0 <= dist && dist < hit->dist) {
					hitTri = tri;
					hit->dist = dist;
					hit->uv = triUV;
					didHit = true;
				}
			}
		}
		extents = curr.max - curr.min;
		farSide = getOppositeBoxSide(scaledDir, closeSide, &uv);
		closeSide = farSide - 2 * (farSide % 2) + 1;
		uv = curr.min + uv * extents;
		currOctreeIndex = curr.neighbors[farSide];
		if (length(uv - newRay.origin) > hit->dist) {
			break;
		}
	}
	if (didHit) {
		double u = hit->uv.s0;
		double v = hit->uv.s1;

		double3 normA = normals[triangles[2 + 9 * hitTri + 3 * 0]];
		double3 normB = normals[triangles[2 + 9 * hitTri + 3 * 1]];
		double3 normC = normals[triangles[2 + 9 * hitTri + 3 * 2]];
		hit->normal = normalize(applyTranspose(objects[index].InvM, (1.0 - u - v)*normA + u * normB + v * normC));

		double2 uvA = uvs[triangles[1 + 9 * hitTri + 3 * 0]];
		double2 uvB = uvs[triangles[1 + 9 * hitTri + 3 * 1]];
		double2 uvC = uvs[triangles[1 + 9 * hitTri + 3 * 2]];
		hit->uv = (1.0 - u - v)*uvA + u * uvB + v * uvC;

		double3 objPoint = newRay.origin + hit->dist * newRay.dir;
		double3 worldPoint = transformPoint(objects[index].M, objPoint);
		hit->dist = length(worldPoint - ray->origin);
		
		return true;
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
		double2 uv = (double2)(0, 0);
		if (pt.x*pt.x - 1 < EPSILON && pt.x*pt.x - 1 > -EPSILON) {
			normal.x = round(pt.x);
			uv.s0 = (pt.y + 1) / 2.0;
			uv.s1 = (pt.z + 1) / 2.0;
		}
		else if (pt.y*pt.y - 1 < EPSILON && pt.y*pt.y - 1 > -EPSILON) {
			normal.y = round(pt.y);
			uv.s0 = (pt.x + 1) / 2.0;
			uv.s1 = (pt.z + 1) / 2.0;
		}
		else if (pt.z*pt.z - 1 < EPSILON && pt.z*pt.z - 1 > -EPSILON) {
			normal.z = round(pt.z);
			uv.s0 = (pt.x + 1) / 2.0;
			uv.s1 = (pt.y + 1) / 2.0;
		}
		double3 worldPt = transformPoint(objects[index].M, pt);
		hit->dist = length(worldPt - ray->origin);
		hit->normal = normalize(applyTranspose(objects[index].InvM, normal));
		hit->uv = uv;
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
		return false;
	}
	double3 objPt = -rayToSphere + dir * dist;
	double3 worldPt = transformPoint(objects[index].M, objPt);
	hit->dist = length(worldPt - ray->origin);
	hit->normal = normalize(applyTranspose(objects[index].InvM, objPt));
	hit->uv.s0 = 0.5 + atan2(objPt.z, objPt.x) / (2 * M_PI);
	hit->uv.s1 = asin(objPt.y) / M_PI + 0.5;
	return true;
}

bool sample_light(
	global const Object* objects,
	const int object_count,
	global const double3 *vertices,
	global const double3 *normals,
	global const double2 *uvs,
	global const unsigned int *triangles,
	global const Octree *octrees,
	global const int *octreeTris,
	global const unsigned char *textures,
	const Ray *ray,
	int lightIndex,
	double lightDist
) {
	/* check if the ray intersects each object in the scene */
	for (int i = 0; i < object_count; i++) {
		if (i != lightIndex) {
			Hit newHit;
			newHit.dist = 1e20;
			switch (objects[i].type) {
			case SPHERE:
				if (intersect_sphere(objects, i, ray, &newHit)) {
					if (newHit.dist < lightDist) {
						return false;
					}
				}
				break;
			case CUBE:
				if (intersect_cube(objects, i, ray, &newHit)) {
					if (newHit.dist < lightDist) {
						return false;
					}
				}
				break;
			case MESH:
				if (intersect_octree(objects, i, vertices, normals, uvs, triangles, octrees, octreeTris, ray, &newHit)) {
					if (newHit.dist < lightDist) {
						return false;
					}
				}
				break;
			}
		}
	}
	
	return true;
}

bool intersect_scene(
	global const Object* objects,
	const int object_count,
	global const double3 *vertices,
	global const double3 *normals,
	global const double2 *uvs,
	global const unsigned int *triangles,
	global const Octree *octrees,
	global const int *octreeTris,
	global const unsigned char *textures,
	const Ray *ray,
	Hit *hit
) {
	/* initialise t to a very large number,
	so t will be guaranteed to be smaller
	when a hit with the scene occurs */

	float inf = 1e20;
	hit->dist = inf;
	bool didHit = false;
	/* check if the ray intersects each object in the scene */
	for (int i = 0; i < object_count; i++) {
		Hit newHit;
		newHit.dist = inf;
		switch (objects[i].type) {
		case SPHERE:
			if (intersect_sphere(objects, i, ray, &newHit)) {
				if (newHit.dist < hit->dist) {
					*hit = newHit;
					hit->object = i;
					didHit = true;
				}
			}
			break;
		case CUBE:
			if (intersect_cube(objects, i, ray, &newHit)) {
				if (newHit.dist < hit->dist) {
					*hit = newHit;
					hit->object = i;
					didHit = true;
				}
			}
			break;
		case MESH:
			if (intersect_octree(objects, i, vertices, normals, uvs, triangles, octrees, octreeTris, ray, &newHit)) {
				if (newHit.dist < hit->dist) {
					*hit = newHit;
					hit->object = i;
					didHit = true;
				}
			}
			break;
		}
	}
	if (didHit) {
		if (dot(hit->normal, ray->dir) > 0) {
			hit->normal *= -1;
		}
		if (objects[hit->object].textureIndex != -1) {
			int width = objects[hit->object].textureWidth;
			int x = width * hit->uv.s0;
			int y = objects[hit->object].textureHeight * (1.0 - hit->uv.s1);
			int offset = objects[hit->object].textureIndex;
			hit->color = (float3)(
				textures[offset + 3 * (width * y + x) + 0] / 255.0,
				textures[offset + 3 * (width * y + x) + 1] / 255.0,
				textures[offset + 3 * (width * y + x) + 2] / 255.0
			);
		}
		else {
			hit->color = objects[hit->object].color;
		}
		return true;
	}
	return false;
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
	global const double2 *uvs,
	global const unsigned int *triangles,
	global const Octree *octrees,
	global const int *octreeTris,
	global const unsigned char *textures,
	const Ray* camray
) {
	Hit hit;
	if (!intersect_scene(objects, object_count, vertices, normals, uvs, triangles, octrees, octreeTris, textures, camray, &hit))
		return (float3)(0.15f, 0.15f, 0.25f);

	double3 hitpoint = camray->origin + camray->dir * hit.dist;
	double3 normal = hit.normal;

	float3 color = hit.color * 0.2f;

	/*
	if (objects[hit.object].type == MESH) {
		Ray reflectRay;
		reflectRay.dir = camray->dir - 2 * normal * dot(normal, camray->dir);
		reflectRay.origin = hitpoint + reflectRay.dir * 0.001;

		if (!intersect_scene(objects, object_count, vertices, normals, uvs, triangles, octrees, octreeTris, textures, &reflectRay, &hit))
			return (float3)(0.15f, 0.15f, 0.25f);

		hitpoint = reflectRay.origin + reflectRay.dir * hit.dist;
		normal = hit.normal;

		color += hit.color * 0.2f;
	}
	*/
	
	double3 light_pos = transformPoint(objects[7].M, (double3)(0,0,0)) + (double3)(0, 1, 0);
	double3 light_dir = light_pos - hitpoint;
	
	Ray lightRay;
	lightRay.dir = normalize(light_dir);
	lightRay.origin = hitpoint + hit.normal * 0.001;

	Hit newHit;
	if (dot(normal, light_dir) > 0) {
		if (sample_light(objects, object_count, vertices, normals, uvs, triangles, octrees, octreeTris, textures, &lightRay, 7, length(light_dir))) {
			color += (float)(0.8*dot(normal, lightRay.dir)) * hit.color;
		}
	}
	return color;
}

union Colour { float c; uchar4 components; };

__kernel void render_kernel(
	global const Object* objects,
	const int object_count,
	global const double3 *vertices,
	global const double3 *normals,
	global const double2 *uvs,
	global const int *triangles,
	global const Octree *octrees,
	global const int *octreeTris,
	global const unsigned char *textures,
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

			finalcolor += trace(objects, object_count, vertices, normals, uvs, triangles, octrees, octreeTris, textures, &camray);
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
