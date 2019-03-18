/* OpenCL based simple sphere path tracer by Sam Lapere, 2016*/
/* based on smallpt by Kevin Beason */
/* http://raytracey.blogspot.com */


__constant float EPSILON = 0.0000001f;
__constant int MSAASAMPLES = 1;

typedef struct Ray{
	float3 origin;
	float3 dir;
} Ray;

typedef struct Ray4D {
	float4 origin;
	float4 dir;
} Ray4D;

enum objectType { SPHERE, CUBE, MESH };

typedef struct Object {
	float4 M[4];
	float4 InvM[4];
	float4 Lorentz[4];
	float4 InvLorentz[4];
	float4 stationaryCam;
	float3 color;
	enum objectType type;
	int meshIndex;
	int textureIndex;
	int textureWidth;
	int textureHeight;
	bool light;
} Object;

typedef struct Hit {
	float dist;
	float3 normal;
	float2 uv;
	float3 color;
	int object;
} Hit;

typedef struct Octree {
	float3 min;
	float3 max;
	int trisIndex,
		trisCount;
	int children[8];
	int neighbors[6];
} Octree;

Ray createCamRay(const float x_coord, const float y_coord, const int width, const int height){

	float fx = (float)x_coord / (float)width;  /* convert int in range [0 - width] to float in range [0-1] */
	float fy = (float)y_coord / (float)height; /* convert int in range [0 - height] to float in range [0-1] */

	/* calculate aspect ratio */
	float aspect_ratio = (float)(width) / (float)(height);
	float fx2 = (fx - 0.5f) * aspect_ratio;
	float fy2 = fy - 0.5f;

	/* determine position of pixel on screen */
	float3 pixel_pos = (float3)(fx2, fy2, 0.5f);

	/* create camera ray*/
	Ray ray;
	ray.origin = (float3)(0, 0, 0); /* fixed camera position */
	ray.dir = normalize(pixel_pos); /* vector from camera to pixel on screen */
	return ray;
}

float3 transformPoint(const float4 M[4], const float3 v) {
	float4 V = (float4)(v, 1.0f);
	return (float3)(
		dot(M[0], V),
		dot(M[1], V),
		dot(M[2], V)
	);
}

float4 transformPoint4D(const float4 M[4], const float4 v) {
	return (float4)(
		dot(M[0], v),
		dot(M[1], v),
		dot(M[2], v),
		dot(M[3], v)
	);
}

float3 transformDirection(const float4 M[4], const float3 v) {
	return (float3)(
		dot(M[0].xyz, v),
		dot(M[1].xyz, v),
		dot(M[2].xyz, v)
	);
}


float3 applyTranspose(const float4 M[4], const float3 v) {
	return M[0].xyz * v.xxx + M[1].xyz * v.yyy + M[2].xyz * v.zzz;
}

bool intersect_triangle(const float3 A, const float3 B, const float3 C, const Ray *ray, float *dist, float2 *uv) {
	/*
	https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
    */
	float3 v0v1 = B - A;
	float3 v0v2 = C - A;
	float3 pvec = cross(ray->dir, v0v2);
	float det = dot(v0v1, pvec);
	if (det < EPSILON && -EPSILON < det) return false;

	float invDet = 1 / det;

	float3 tvec = ray->origin - A;
	uv->s0 = dot(tvec, pvec) * invDet;
	if (uv->s0 < 0 || uv->s0 > 1) return false;

	float3 qvec = cross(tvec, v0v1);
	uv->s1 = dot(ray->dir, qvec) * invDet;
	if (uv->s1 < 0 || uv->s0 + uv->s1 > 1) return false;

	*dist = dot(v0v2, qvec) * invDet;
	return true;
}

bool intersect_AABB(const float3 bounds[2], const Ray *ray, float2 *d, int *closeSide, int *farSide) {
	float3 origin = ray->origin;
	float3 dir = ray->dir;
	float3 inv_dir = 1.0f / dir;
	int sign[3] = {
		inv_dir.x < 0 ? 1 : 0,
		inv_dir.y < 0 ? 1 : 0,
		inv_dir.z < 0 ? 1 : 0
	};
	d->s0 = (bounds[sign[0]].x - origin.x) * inv_dir.x;
	d->s1 = (bounds[1 - sign[0]].x - origin.x) * inv_dir.x;
	*closeSide = 2 + sign[0];
	*farSide = 3 - sign[0];
	float tymin = (bounds[sign[1]].y - origin.y) * inv_dir.y;
	float tymax = (bounds[1 - sign[1]].y - origin.y) * inv_dir.y;
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
	float tzmin = (bounds[sign[2]].z - origin.z) * inv_dir.z;
	float tzmax = (bounds[1 - sign[2]].z - origin.z) * inv_dir.z;
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

int getOppositeBoxSide(const float3 scaledDir, const int closeSide, float3 *uv) {
	float3 inv_dir = 1.0f / scaledDir;
	int sign[3] = { inv_dir.x < 0, inv_dir.y < 0, inv_dir.z < 0 };
	float dx = (1 - sign[0] - uv->x) * inv_dir.x;
	float dy = (1 - sign[1] - uv->y) * inv_dir.y;
	float dz = (1 - sign[2] - uv->z) * inv_dir.z;
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
	global const float3 *vertices,
	global const float3 *normals,
	global const float2 *uvs,
	global const unsigned int *triangles,
	global const Octree *octrees,
	global const int *octreeTris,
	const Ray4D *ray,
	Hit *hit
) {
	Ray newRay;
	newRay.origin = transformPoint(objects[index].InvM, ray->origin.yzw);
	newRay.dir = transformDirection(objects[index].InvM, ray->dir.yzw);
	float scale = length(newRay.dir);
	newRay.dir /= scale;


	int currOctreeIndex = objects[index].meshIndex;
	float2 d;
	int closeSide, farSide;
	float3 bounds[2] = {
		octrees[currOctreeIndex].min,
		octrees[currOctreeIndex].max
	};
	bool didHit = false;
	int hitTri;
	if (!intersect_AABB(bounds, &newRay, &d, &closeSide, &farSide)) {
		return false;
	}
	float3 uv = newRay.origin + newRay.dir * d.s0;

	if (d.s0 < 0) {
		Octree curr = octrees[currOctreeIndex];
		uv = (newRay.origin - curr.min) / (curr.max - curr.min);
		while (curr.children[0] != -1) {
			int childIndex = round(uv.z) + 2 * round(uv.y) + 4 * round(uv.x);
			uv = 2.0f * fmod(min(uv, 1.0f - EPSILON), 0.5f);
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

	float3 scaledDir = newRay.dir / (octrees[currOctreeIndex].max - octrees[currOctreeIndex].min);
	scaledDir = normalize(scaledDir);
	while(currOctreeIndex != -1) {
		Octree curr = octrees[currOctreeIndex];
		float3 extents = curr.max - curr.min;
		uv = (uv - curr.min) / extents;
		while (curr.children[0] != -1) {
			int childIndex = round(uv.z) + 2 * round(uv.y) + 4 * round(uv.x);
			uv = 2.0f * fmod(min(uv, 1.0f - EPSILON), 0.5f);
			currOctreeIndex = curr.children[childIndex];
			curr = octrees[currOctreeIndex];
		}
		for (int i = curr.trisIndex; i < curr.trisIndex + curr.trisCount; i++) {
			int tri = octreeTris[i];
			float3 A = vertices[triangles[9 * tri + 3 * 0]];
			float3 B = vertices[triangles[9 * tri + 3 * 1]];
			float3 C = vertices[triangles[9 * tri + 3 * 2]];
			float dist;
			float2 triUV;
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
		float u = hit->uv.s0;
		float v = hit->uv.s1;

		float3 normA = normals[triangles[2 + 9 * hitTri + 3 * 0]];
		float3 normB = normals[triangles[2 + 9 * hitTri + 3 * 1]];
		float3 normC = normals[triangles[2 + 9 * hitTri + 3 * 2]];
		hit->normal = normalize(applyTranspose(objects[index].InvM, (1.0f - u - v)*normA + u * normB + v * normC));

		float2 uvA = uvs[triangles[1 + 9 * hitTri + 3 * 0]];
		float2 uvB = uvs[triangles[1 + 9 * hitTri + 3 * 1]];
		float2 uvC = uvs[triangles[1 + 9 * hitTri + 3 * 2]];
		hit->uv = (1.0f - u - v)*uvA + u * uvB + v * uvC;

		float3 objPoint = newRay.origin + hit->dist * newRay.dir;
		float3 worldPoint = transformPoint(objects[index].M, objPoint);
		hit->dist = length(worldPoint - ray->origin.yzw) / length(ray->dir.yzw);
		
		return true;
	}
	return false;
}

float max3(float3 v) { return max(max(v.x, v.y), v.z); }

bool intersect_cube(global const Object *objects, int index, const Ray4D *ray, Hit *hit) {
	/* http://www.jcgt.org/published/0007/03/04/paper-lowres.pdf */
	float3 origin = transformPoint(objects[index].InvM, ray->origin.yzw);
	float3 dir = transformDirection(objects[index].InvM, ray->dir.yzw);
	float scale = length(dir);
	dir /= scale;
	float winding = max3(fabs(origin)) < 1.0f ? -1.0f : 1.0f;
	float3 sgn = -sign(dir);
	float3 d = (winding * sgn - origin) / dir;
	const float2 one = (float2)(1, 1);
# define TEST(U, VW) (d.U >= 0.0f) && all(isless(fabs(origin.VW + dir.VW * d.U), one))
	sgn = TEST(x, yz) ? (float3)(sgn.x, 0, 0) : (TEST(y, zx) ? (float3)(0, sgn.y, 0) :
		(float3)(0, 0, TEST(z, xy) ? sgn.z : 0));
# undef TEST
	float dist = (sgn.x != 0) ? d.x : ((sgn.y != 0) ? d.y : d.z);
	float3 objPt = origin + dir * dist;
	hit->dist = dist / scale;
	hit->normal = normalize(applyTranspose(objects[index].InvM, sgn));
	hit->uv = (sgn.x != 0) ? (objPt.yz + 1) / 2 : ((sgn.y != 0) ? (objPt.xz + 1) / 2 : (objPt.xy + 1) / 2);
	return (sgn.x != 0) || (sgn.y != 0) || (sgn.z != 0);

}

bool intersect_sphere(global const Object *objects, const int index, const Ray4D *ray, Hit *hit) {
	float3 rayToSphere = -transformPoint(objects[index].InvM, ray->origin.yzw);
	float3 dir = transformDirection(objects[index].InvM, ray->dir.yzw);
	float scale = length(dir);
	dir /= scale;
	float b = dot(rayToSphere, dir);
	float c = dot(rayToSphere, rayToSphere) - 1.0f;
	float disc = b * b - c;
	if (disc < 0.0f) return false;
	else disc = sqrt(disc);
	float dist;
	if ((b - disc) > EPSILON) {
		dist = b - disc;
	} else if ((b + disc) > EPSILON) {
		dist = b + disc;
	} else {
		return false;
	}
	float3 objPt = -rayToSphere + dir * dist;
	hit->dist = dist / scale;
	hit->normal = normalize(applyTranspose(objects[index].InvM, objPt));
	hit->uv.s0 = 0.5f + atan2(objPt.z, objPt.x) / (2 * M_PI);
	hit->uv.s1 = asin(objPt.y) / M_PI + 0.5f;
	return true;
}

float modulo(float a, float b) {
	return (a - b * floor(a / b));
}


float3 EncodeFloatRGB(float v) {
	float val = 1.0f / (1 + pow(1.1f, -v));
	float3 enc = (float3)(1.0f, 255.0f, 65025.0f)*val;
	enc -= floor(enc);
	enc -= enc.yzz * (float3)(1 / 255.0f, 1 / 255.0f, 0);
	return enc;
}

bool intersect_scene(
	global const Object* objects,
	const int object_count,
	const int interval,
	global const float3 *vertices,
	global const float3 *normals,
	global const float2 *uvs,
	global const unsigned int *triangles,
	global const Octree *octrees,
	global const int *octreeTris,
	global const unsigned char *textures,
	const Ray *ray,
	const float time,
	Hit *hit
) {
	/* initialise t to a very large number,
	so t will be guaranteed to be smaller
	when a hit with the scene occurs */

	float inf = 1e20;
	hit->dist = inf;
	bool didHit = false;
	float4 event = (float)(0, 0, 0, 0);
	/* check if the ray intersects each object in the scene */
	for (int i = 0; i < object_count; i++) {
		Hit newHit;
		newHit.dist = inf;
		Ray4D newRay;
		float4 newEvent0 = objects[i].stationaryCam;
		float4 lightDir = (float4)(interval, normalize(ray->dir)); //Set t=0 for spacelike, set t=-1 for lightlike
		lightDir = transformPoint4D(objects[i].Lorentz, lightDir);
		//lightDir /= length(lightDir.yzw);
		newRay.origin = newEvent0;
		newRay.dir = lightDir;

		switch (objects[i].type) {
		case SPHERE:
			if (intersect_sphere(objects, i, &newRay, &newHit)) {
				if (newHit.dist < hit->dist) {
					event = newEvent0 + lightDir * newHit.dist;
					//float4 localEvent = transformPoint4D(objects[i].InvLorentz, event);
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
					//float4 localEvent = transformPoint4D(objects[i].InvLorentz, event);
					//newHit.dist = length(localEvent.yzw - ray->origin);
					*hit = newHit;
					hit->object = i;
					didHit = true;
				}
			}
			break;
		case MESH:
			if (intersect_octree(objects, i, vertices, normals, uvs, triangles, octrees, octreeTris, &newRay, &newHit)) {
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
		if (objects[hit->object].textureIndex != -1) {
			int width = objects[hit->object].textureWidth;
			int height = objects[hit->object].textureHeight;
			float u = width * hit->uv.s0;
			float v = height * (1.0f - hit->uv.s1);
			int x = min((int)floor(u), width - 1);
			int y = min((int)floor(v), height - 1);
			float u_ratio = u - x;
			float v_ratio = v - y;
			float u_opp = 1 - u_ratio;
			float v_opp = 1 - v_ratio;

			int offset = objects[hit->object].textureIndex;
			float3 result = (float3)(
				textures[offset + 3 * (width * y + x) + 0] / 255.0f,
				textures[offset + 3 * (width * y + x) + 1] / 255.0f,
				textures[offset + 3 * (width * y + x) + 2] / 255.0f

			) * u_opp;
			x = clamp(x + 1, 0, width - 1);
			result += (float3)(
				textures[offset + 3 * (width * y + x) + 0] / 255.0f,
				textures[offset + 3 * (width * y + x) + 1] / 255.0f,
				textures[offset + 3 * (width * y + x) + 2] / 255.0f

			) * u_ratio;
			result *= v_opp;
			y = clamp(y + 1, 0, height - 1);
			float3 result2 = (float3)(
				textures[offset + 3 * (width * y + x) + 0] / 255.0f,
				textures[offset + 3 * (width * y + x) + 1] / 255.0f,
				textures[offset + 3 * (width * y + x) + 2] / 255.0f

			) * u_ratio;
			x = clamp(x - 1, 0, width - 1);
			result2 += (float3)(
				textures[offset + 3 * (width * y + x) + 0] / 255.0f,
				textures[offset + 3 * (width * y + x) + 1] / 255.0f,
				textures[offset + 3 * (width * y + x) + 2] / 255.0f

			) * u_opp;
			result2 *= v_ratio;

			hit->color = result + result2;
		}
		else {
			hit->color = objects[hit->object].color;
		}
		// Periodic Flash
		float period = 2;
		float duration = 0.5f;
		if (event.x - period * floor(event.x / period) < duration) {
		//	hit->color += (float3)(0.5f, 0.5f, 0.5f);
		}
		//hit->color *= 0.95f + 0.05f * (event.x - k * floor(event.x / k))/k;
		//hit->color *= 1.0f - hit->dist / (hit->dist + 1.0f);
		/*
		float d = hit->dist / 1000.0f;
		int r = d * 255.0f;
		int g = fmod(d * 255.0f, 1.0f) * 255.0f;
		int b = fmod(fmod(d * 255.0f, 1.0f) * 255.0f, 1.0f) * 255.0f;

		hit->color = (float3)(r / 255.0f, g / 255.0f, b / 255.0f);
		*/
		//hit->color = EncodeFloatRGB(hit->dist);
		return true;
	}
	return false;
}

int sample_light(
	global const Object* objects,
	const int object_count,
	const int interval,
	global const float3 *vertices,
	global const float3 *normals,
	global const float2 *uvs,
	global const unsigned int *triangles,
	global const Octree *octrees,
	global const int *octreeTris,
	global const unsigned char *textures,
	const Ray4D *ray,
	const float time,
	float lightDist,
	const int lightIndex
) {
	/* initialise t to a very large number,
	so t will be guaranteed to be smaller
	when a hit with the scene occurs */

	float inf = 1e20;
	bool didHit = false;
	float4 event = (float)(0, 0, 0, 0);
	/* check if the ray intersects each object in the scene */
	for (int i = 0; i < object_count; i++) {
		if (i != lightIndex) {

			Hit newHit;
			newHit.dist = inf;
			Ray4D newRay;
			float4 newEvent0 = transformPoint4D(objects[i].Lorentz, ray->origin);
			float4 lightDir = (float4)(interval, normalize(ray->dir.yzw));
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
			case MESH:
				if (intersect_octree(objects, i, vertices, normals, uvs, triangles, octrees, octreeTris, &newRay, &newHit)) {
					if (newHit.dist < lightDist) {
						return i;
					}
				}
				break;
			}
		}
	}
	return -1;
}


float3 trace(
	global const Object* objects,
	const int object_count,
	const int interval,
	global const float3 *vertices,
	global const float3 *normals,
	global const float2 *uvs,
	global const unsigned int *triangles,
	global const Octree *octrees,
	global const int *octreeTris,
	global const unsigned char *textures,
	const float ambient,
	const float time,
	const Ray* camray
) {
	Hit hit;
	if (!intersect_scene(objects, object_count, interval, vertices, normals, uvs, triangles, octrees, octreeTris, textures, camray, time, &hit))
		return (float3)(0.15f, 0.15f, 0.25f);

	float3 color = hit.color * (interval != 0 ? ambient : 1.0f);

	if (objects[hit.object].light) {
		color += hit.color;
	}
	if (interval != 0) {
		for (int i = 0; i < object_count; i++) {
			if (i != hit.object && objects[i].light) {
				float4 cameraPos_ObjFrame = objects[hit.object].stationaryCam;
				float4 rayDir = (float4)(interval, normalize(camray->dir));
				float4 rayDir_ObjFrame = transformPoint4D(objects[hit.object].Lorentz, rayDir);
				float4 hitPos_ObjFrame = cameraPos_ObjFrame + rayDir_ObjFrame * hit.dist;
				hitPos_ObjFrame += (float4)(0, hit.normal * 0.001f);
				float4 hitPos = transformPoint4D(objects[hit.object].InvLorentz, hitPos_ObjFrame);				
				float4 hitPos_LightFrame = transformPoint4D(objects[i].Lorentz, hitPos);
				float3 hitPos3_LightFrame = hitPos_LightFrame.yzw;
				float3 lightPos3_LightFrame = (float3)(objects[i].M[0].w, objects[i].M[1].w, objects[i].M[2].w);
				float3 lightDir3_LightFrame = lightPos3_LightFrame - hitPos3_LightFrame;
				float4 lightDir_LightFrame = (float4)(interval * length(lightDir3_LightFrame), lightDir3_LightFrame);
				float4 lightDir = transformPoint4D(objects[i].InvLorentz, lightDir_LightFrame);
				float4 lightDir_ObjFrame = transformPoint4D(objects[hit.object].Lorentz, lightDir);
				float3 lightDir3_ObjFrame = lightDir_ObjFrame.yzw;
				float3 unitLightDir3 = normalize(lightDir3_ObjFrame);

				if (dot(hit.normal, unitLightDir3) > 0) {
					Ray4D newRay;
					newRay.dir = (float4)(interval, normalize(lightDir.yzw));
					newRay.origin = hitPos;
					int shadowIndex = sample_light(objects, object_count, interval, vertices, normals, uvs, triangles, octrees, octreeTris, textures, &newRay, time, length(lightDir.yzw), i);
					if (shadowIndex == -1) {
						color += dot(hit.normal, unitLightDir3) / (1.0f + 0.1f * length(lightDir3_ObjFrame) + 0.01f*dot(lightDir3_ObjFrame, lightDir3_ObjFrame)) * hit.color * objects[i].color;
					}
					else {
						//	color += 0.5f * objects[shadowIndex].color;
					}
				}
			}			
		}
	}
	return color;
}


float3 hable(const float3 x) {
	float A = 0.15f;
	float B = 0.50f;
	float C = 0.10f;
	float D = 0.20f;
	float E = 0.02f;
	float F = 0.30f;

	return ((x*(A*x + C * B) + D * E) / (x*(A*x + B) + D * F)) - E / F;
}

union Colour { float c; uchar4 components; };

__kernel void render_kernel(
	global const Object* objects,
	const int object_count,
	global const float3 *vertices,
	global const float3 *normals,
	global const float2 *uvs,
	global const int *triangles,
	global const Octree *octrees,
	global const int *octreeTris,
	global const unsigned char *textures,
	const float3 white_point,
	const float ambient,
	const int width,
	const int height,
	const int interval,
	__global float3* output
) {
	unsigned int work_item_id = get_global_id(0);	/* the unique global id of the work item for the current pixel */
	unsigned int x_coord = work_item_id % width;			/* x-coordinate of the pixel */
	unsigned int y_coord = work_item_id / width;			/* y-coordinate of the pixel */

	float3 finalcolor = (float3)(0, 0, 0);
	for (int y = 0; y < MSAASAMPLES; y++) {
		for (int x = 0; x < MSAASAMPLES; x++) {
			Ray camray = createCamRay((float)x_coord + (float)x/MSAASAMPLES, (float)y_coord + (float)y/ MSAASAMPLES, width, height);
			finalcolor += trace(objects, object_count, interval, vertices, normals, uvs, triangles, octrees, octreeTris, textures, ambient, 0, &camray);
		}
	}
	finalcolor = finalcolor / (MSAASAMPLES*MSAASAMPLES);
	finalcolor = hable(finalcolor) / hable(white_point);
	finalcolor = min(finalcolor, 1.0f);

	union Colour fcolour;
	fcolour.components = (uchar4)(	
		(unsigned char)(finalcolor.x * 255), 
		(unsigned char)(finalcolor.y * 255),
		(unsigned char)(finalcolor.z * 255),
		1);
	/* store the pixelcolour in the output buffer */
	output[work_item_id] = (float3)(x_coord, y_coord, fcolour.c);
}
