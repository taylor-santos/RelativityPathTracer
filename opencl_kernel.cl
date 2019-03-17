/* OpenCL based simple sphere path tracer by Sam Lapere, 2016*/
/* based on smallpt by Kevin Beason */
/* http://raytracey.blogspot.com */


__constant double EPSILON = 0.0000000001; /* required to compensate for limited float precision */
__constant int MSAASAMPLES = 1;

#define SPACELIKE 0
#define LIGHTLIKE -1
#define INTERVAL SPACELIKE

typedef struct Ray{
	double3 origin;
	double3 dir;
} Ray;

typedef struct Ray4D {
	double4 origin;
	double4 dir;
} Ray4D;

enum objectType { SPHERE, CUBE, MESH };

typedef struct Object {
	double4 M[4];
	double4 InvM[4];
	double4 Lorentz[4];
	double4 InvLorentz[4];
	double4 stationaryCam;
	double3 color;
	enum objectType type;
	int meshIndex;
	int textureIndex;
	int textureWidth;
	int textureHeight;
	bool light;
} Object;

typedef struct Hit {
	double dist;
	double3 normal;
	double2 uv;
	double3 color;
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

double4 transformPoint4D(const double4 M[4], const double4 v) {
	return (double4)(
		dot(M[0], v),
		dot(M[1], v),
		dot(M[2], v),
		dot(M[3], v)
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
	newRay.dir = transformDirection(objects[index].InvM, ray->dir);
	double scale = length(newRay.dir);
	newRay.dir /= scale;


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
		hit->dist = length(worldPoint - ray->origin) / scale;
		
		return true;
	}
	return false;
}

double max3(double3 v) { return max(max(v.x, v.y), v.z); }

bool intersect_cube(global const Object *objects, int index, const Ray4D *ray, Hit *hit) {
	/* http://www.jcgt.org/published/0007/03/04/paper-lowres.pdf */
	Ray newRay;
	double3 origin = transformPoint(objects[index].InvM, ray->origin.yzw);
	double3 dir = transformDirection(objects[index].InvM, ray->dir.yzw);
	double scale = length(dir);
	dir /= scale;
	double winding = max3(fabs(origin)) < 1.0 ? -1.0 : 1.0;
	double3 sgn = -sign(dir);
	double3 d = (winding * sgn - origin) / dir;
	const double2 one = (double2)(1, 1);
# define TEST(U, VW) (d.U >= 0.0) && all(isless(fabs(origin.VW + dir.VW * d.U), one))
	sgn = TEST(x, yz) ? (double3)(sgn.x, 0, 0) : (TEST(y, zx) ? (double3)(0, sgn.y, 0) :
		(double3)(0, 0, TEST(z, xy) ? sgn.z : 0));
# undef TEST
	double dist = (sgn.x != 0) ? d.x : ((sgn.y != 0) ? d.y : d.z);
	double3 objPt = origin + dir * dist;
	hit->dist = dist / scale;
	hit->normal = normalize(applyTranspose(objects[index].InvM, sgn));
	hit->uv = (sgn.x != 0) ? (objPt.yz + 1) / 2 : ((sgn.y != 0) ? (objPt.xz + 1) / 2 : (objPt.xy + 1) / 2);
	return (sgn.x != 0) || (sgn.y != 0) || (sgn.z != 0);

}

bool intersect_sphere(global const Object *objects, const int index, const Ray4D *ray, Hit *hit) {
	double3 rayToSphere = -transformPoint(objects[index].InvM, ray->origin.yzw);
	double3 dir = transformDirection(objects[index].InvM, ray->dir.yzw);
	double scale = length(dir);
	dir /= scale;
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
	hit->dist = dist / scale;
	hit->normal = normalize(applyTranspose(objects[index].InvM, objPt));
	hit->uv.s0 = 0.5 + atan2(objPt.z, objPt.x) / (2 * M_PI);
	hit->uv.s1 = asin(objPt.y) / M_PI + 0.5;
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
	const double time,
	Hit *hit
) {
	/* initialise t to a very large number,
	so t will be guaranteed to be smaller
	when a hit with the scene occurs */

	float inf = 1e20;
	hit->dist = inf;
	bool didHit = false;
	double4 event = (double)(0, 0, 0, 0);
	/* check if the ray intersects each object in the scene */
	for (int i = 0; i < object_count; i++) {
		Hit newHit;
		newHit.dist = inf;
		Ray4D newRay;
		double4 newEvent0 = objects[i].stationaryCam;
		double4 lightDir = (double4)(INTERVAL, normalize(ray->dir)); //Set t=0 for spacelike, set t=-1 for lightlike
		lightDir = transformPoint4D(objects[i].Lorentz, lightDir);
		//lightDir /= length(lightDir.yzw);
		newRay.origin = newEvent0;
		newRay.dir = lightDir;

		switch (objects[i].type) {
		case SPHERE:
			if (intersect_sphere(objects, i, &newRay, &newHit)) {
				if (newHit.dist < hit->dist) {
					event = newEvent0 + lightDir * newHit.dist;
					//double4 localEvent = transformPoint4D(objects[i].InvLorentz, event);
					//newHit.dist = length(localEvent.yzw - ray->origin);
					*hit = newHit;
					hit->object = i;
					didHit = true;
				}
			}
			break;
		case CUBE:
			if (intersect_cube(objects, i, &newRay, &newHit)) {
				if (newHit.dist < hit->dist) {
					event = newEvent0 + lightDir * newHit.dist;
					//double4 localEvent = transformPoint4D(objects[i].InvLorentz, event);
					//newHit.dist = length(localEvent.yzw - ray->origin);
					*hit = newHit;
					hit->object = i;
					didHit = true;
				}
			}
			break;
			/*
		case MESH:
			if (intersect_octree(objects, i, vertices, normals, uvs, triangles, octrees, octreeTris, &newRay, &newHit)) {
				if (newHit.dist < hit->dist) {
					*hit = newHit;
					hit->object = i;
					didHit = true;
				}
			}
			break;
			*/
		}
		
	}
	if (didHit) {
		if (objects[hit->object].textureIndex != -1) {
			int width = objects[hit->object].textureWidth;
			int height = objects[hit->object].textureHeight;
			double u = width * hit->uv.s0;
			double v = height * (1.0 - hit->uv.s1);
			int x = min((int)floor(u), width - 1);
			int y = min((int)floor(v), height - 1);
			double u_ratio = u - x;
			double v_ratio = v - y;
			double u_opp = 1 - u_ratio;
			double v_opp = 1 - v_ratio;

			int offset = objects[hit->object].textureIndex;
			double3 result = (double3)(
				textures[offset + 3 * (width * y + x) + 0] / 255.0,
				textures[offset + 3 * (width * y + x) + 1] / 255.0,
				textures[offset + 3 * (width * y + x) + 2] / 255.0
			) * u_opp;
			x = clamp(x + 1, 0, width - 1);
			result += (double3)(
				textures[offset + 3 * (width * y + x) + 0] / 255.0,
				textures[offset + 3 * (width * y + x) + 1] / 255.0,
				textures[offset + 3 * (width * y + x) + 2] / 255.0
			) * u_ratio;
			result *= v_opp;
			y = clamp(y + 1, 0, height - 1);
			double3 result2 = (double3)(
				textures[offset + 3 * (width * y + x) + 0] / 255.0,
				textures[offset + 3 * (width * y + x) + 1] / 255.0,
				textures[offset + 3 * (width * y + x) + 2] / 255.0
			) * u_ratio;
			x = clamp(x - 1, 0, width - 1);
			result2 += (double3)(
				textures[offset + 3 * (width * y + x) + 0] / 255.0,
				textures[offset + 3 * (width * y + x) + 1] / 255.0,
				textures[offset + 3 * (width * y + x) + 2] / 255.0
			) * u_opp;
			result2 *= v_ratio;

			hit->color = result + result2;
		}
		else {
			hit->color = objects[hit->object].color;
		}
		/* // Periodic Flash
		double period = 2;
		double duration = 1;
		if (event.x - period * floor(event.x / period) < duration) {
			hit->color += (double3)(0.5, 0.5, 0.5);
		}
		*/
		/*
		
		hit->color *= (event.x - k * floor(event.x / k))/k;
		*/
		//hit->color *= 1.0 - hit->dist / (hit->dist + 1.0);
		/*
		double d = hit->dist / 1000.0;
		int r = d * 255.0;
		int g = fmod(d * 255.0, 1.0) * 255.0;
		int b = fmod(fmod(d * 255.0, 1.0) * 255.0, 1.0) * 255.0;

		hit->color = (double3)(r / 255.0, g / 255.0, b / 255.0);
		*/
		return true;
	}
	return false;
}

int sample_light(
	global const Object* objects,
	const int object_count,
	global const double3 *vertices,
	global const double3 *normals,
	global const double2 *uvs,
	global const unsigned int *triangles,
	global const Octree *octrees,
	global const int *octreeTris,
	global const unsigned char *textures,
	const Ray4D *ray,
	const double time,
	double lightDist,
	const int lightIndex
) {
	/* initialise t to a very large number,
	so t will be guaranteed to be smaller
	when a hit with the scene occurs */

	float inf = 1e20;
	bool didHit = false;
	double4 event = (double)(0, 0, 0, 0);
	/* check if the ray intersects each object in the scene */
	for (int i = 0; i < object_count; i++) {
		if (i != lightIndex) {

			Hit newHit;
			newHit.dist = inf;
			Ray4D newRay;
			double4 newEvent0 = transformPoint4D(objects[i].Lorentz, ray->origin);
			double4 lightDir = (double4)(INTERVAL, normalize(ray->dir.yzw));
			lightDir = transformPoint4D(objects[i].Lorentz, lightDir);
			//lightDir /= length(lightDir.yzw);
			newRay.origin = newEvent0;
			newRay.dir = lightDir;

			switch (objects[i].type) {
			case SPHERE:
				if (intersect_sphere(objects, i, &newRay, &newHit)) {
					if (newHit.dist < lightDist) {
						return i;
					}
				}
				break;
			case CUBE:
				if (intersect_cube(objects, i, &newRay, &newHit)) {
					if (newHit.dist < lightDist) {
						return i;
					}
				}
				break;
				/*
			case MESH:
				if (intersect_octree(objects, i, vertices, normals, uvs, triangles, octrees, octreeTris, &newRay, &newHit)) {
					if (newHit.dist < hit->dist) {
						*hit = newHit;
						hit->object = i;
						didHit = true;
					}
				}
				break;
				*/
			}
		}
	}
	return -1;
}

double3 trace(
	global const Object* objects,
	const int object_count,
	global const double3 *vertices,
	global const double3 *normals,
	global const double2 *uvs,
	global const unsigned int *triangles,
	global const Octree *octrees,
	global const int *octreeTris,
	global const unsigned char *textures,
	const double ambient,
	const double time,
	const Ray* camray
) {
	Hit hit;
	if (!intersect_scene(objects, object_count, vertices, normals, uvs, triangles, octrees, octreeTris, textures, camray, time, &hit))
		return (double3)(0.15, 0.15, 0.25);

	double3 color = hit.color * ambient;// *(normal.y > 0 ? normal.y + 0.2 : 0.2);

	if (objects[hit.object].light) {
		color += hit.color;
	}
	for (int i = 0; i < object_count; i++) {
		if (i != hit.object && objects[i].light) {

			double4 cameraPos_ObjFrame = objects[hit.object].stationaryCam;
			double4 rayDir = (double4)(INTERVAL, normalize(camray->dir));
			double4 rayDir_ObjFrame = transformPoint4D(objects[hit.object].Lorentz, rayDir);
			double4 hitPos_ObjFrame = cameraPos_ObjFrame + rayDir_ObjFrame * hit.dist;
			hitPos_ObjFrame += (double4)(0, hit.normal * 0.001);
			double4 hitPos = transformPoint4D(objects[hit.object].InvLorentz, hitPos_ObjFrame);
			double4 hitPos_LightFrame = transformPoint4D(objects[i].Lorentz, hitPos);
			double4 lightDir;
			if (INTERVAL) {
				double3 hitPos3_LightFrame = hitPos_LightFrame.yzw;
				double3 lightPos3_LightFrame = (double3)(objects[i].M[0].w, objects[i].M[1].w, objects[i].M[2].w);
				double3 lightDir3_LightFrame = lightPos3_LightFrame - hitPos3_LightFrame;
				double4 lightDir_LightFrame = (double4)(INTERVAL * length(lightDir3_LightFrame), lightDir3_LightFrame);
				lightDir = transformPoint4D(objects[i].InvLorentz, lightDir_LightFrame);
			}
			else {
				double4 hitPos_OffsetLightFrame = hitPos_LightFrame - (double4)(0, objects[i].M[0].w, objects[i].M[1].w, objects[i].M[2].w);
				double4 norm = normalize(transformPoint4D(objects[i].InvLorentz, (double4)(1, 0, 0, 0)));
				double4 lightPos_OffsetLightFrame = (double4)(dot(norm, hitPos_OffsetLightFrame) / norm.x, 0, 0, 0);
				double4 lightPos_LightFrame = lightPos_OffsetLightFrame + (double4)(0, objects[i].M[0].w, objects[i].M[1].w, objects[i].M[2].w);
				double4 lightPos = transformPoint4D(objects[i].InvLorentz, lightPos_LightFrame);
				lightDir = lightPos - hitPos;
			}
			double4 lightDir_ObjFrame = transformPoint4D(objects[hit.object].Lorentz, lightDir);
			double3 lightDir3_ObjFrame = lightDir_ObjFrame.yzw;
			double3 unitLightDir3 = normalize(lightDir3_ObjFrame);

			if (dot(hit.normal, unitLightDir3) > 0) {
				Ray4D newRay;
				newRay.dir = (double4)(INTERVAL, normalize(lightDir.yzw));
				newRay.origin = hitPos;
				int shadowIndex = sample_light(objects, object_count, vertices, normals, uvs, triangles, octrees, octreeTris, textures, &newRay, time, length(lightDir.yzw), i);
				if (shadowIndex == -1) {
					color += dot(hit.normal, unitLightDir3) / (1.0 + 0.1 * length(lightDir3_ObjFrame) + 0.01*dot(lightDir3_ObjFrame, lightDir3_ObjFrame)) * hit.color * objects[i].color;
				}
			}
		}
	}
	return color;
}


double3 hable(const double3 x) {
	double A = 0.15;
	double B = 0.50;
	double C = 0.10;
	double D = 0.20;
	double E = 0.02;
	double F = 0.30;

	return ((x*(A*x + C * B) + D * E) / (x*(A*x + B) + D * F)) - E / F;
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
	const double3 white_point,
	const double ambient,
	const double time,
	const int width,
	const int height,
	__global float3* output
) {
	unsigned int work_item_id = get_global_id(0);	/* the unique global id of the work item for the current pixel */
	unsigned int x_coord = work_item_id % width;			/* x-coordinate of the pixel */
	unsigned int y_coord = work_item_id / width;			/* y-coordinate of the pixel */
	
	double3 finalcolor = (double3)(0, 0, 0);

	for (int y = 0; y < MSAASAMPLES; y++) {
		for (int x = 0; x < MSAASAMPLES; x++) {
			Ray camray = createCamRay((double)x_coord + (double)x/MSAASAMPLES, (double)y_coord + (double)y/ MSAASAMPLES, width, height);

			finalcolor += trace(objects, object_count, vertices, normals, uvs, triangles, octrees, octreeTris, textures, ambient, time, &camray);
		}
	}
	finalcolor = finalcolor / (MSAASAMPLES*MSAASAMPLES);
	finalcolor = hable(finalcolor) / hable(white_point);
	finalcolor = min(finalcolor, 1.0);

	union Colour fcolour;
	fcolour.components = (uchar4)(	
		(unsigned char)(finalcolor.x * 255), 
		(unsigned char)(finalcolor.y * 255),
		(unsigned char)(finalcolor.z * 255),
		1);
	/* store the pixelcolour in the output buffer */
	output[work_item_id] = (float3)(x_coord, y_coord, fcolour.c);
}
