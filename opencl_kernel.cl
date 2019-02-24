/* OpenCL based simple sphere path tracer by Sam Lapere, 2016*/
/* based on smallpt by Kevin Beason */
/* http://raytracey.blogspot.com */


__constant float EPSILON = 0.00003f; /* required to compensate for limited float precision */
__constant float PI = 3.14159265359f;

typedef struct Ray{
	float3 origin;
	float3 dir;
} Ray;

typedef struct Sphere{
	float4 M[4];
	float4 InvM[4];
	float3 color;
	float3 emission;
} Sphere;

typedef struct Hit {
	float dist;
	float3 normal;
} Hit;

static float get_random(unsigned int *seed0, unsigned int *seed1) {

	/* hash the seeds using bitwise AND operations and bitshifts */
	*seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16);
	*seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);

	unsigned int ires = ((*seed0) << 16) + (*seed1);

	/* use union struct to convert int to float */
	union {
		float f;
		unsigned int ui;
	} res;

	res.ui = (ires & 0x007fffff) | 0x40000000;  /* bitwise AND, bitwise OR */
	return (res.f - 2.0f) / 2.0f;
}

Ray createCamRay(const int x_coord, const int y_coord, const int width, const int height){

	float fx = (float)x_coord / (float)width;  /* convert int in range [0 - width] to float in range [0-1] */
	float fy = (float)y_coord / (float)height; /* convert int in range [0 - height] to float in range [0-1] */

	/* calculate aspect ratio */
	float aspect_ratio = (float)(width) / (float)(height);
	float fx2 = (fx - 0.5f) * aspect_ratio;
	float fy2 = fy - 0.5f;

	/* determine position of pixel on screen */
	float3 pixel_pos = (float3)(fx2, fy2, 1.0f);

	/* create camera ray*/
	Ray ray;
	ray.origin = (float3)(0, 0, 0); /* fixed camera position */
	ray.dir = normalize(pixel_pos); /* vector from camera to pixel on screen */

	return ray;
}

float3 transformPoint(const float4 M[4], const float3 v) {
	float4 V = (float4)(v, 1.0);
	return (float3)(
		dot(M[0], V),
		dot(M[1], V),
		dot(M[2], V)
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

bool intersect_sphere(const Sphere* sphere, const Ray* ray, Hit *hit)
{
	float3 rayToSphere = -transformPoint(sphere->InvM, ray->origin);
	float3 dir = normalize(transformDirection(sphere->InvM, ray->dir));
	float b = dot(rayToSphere, dir);
	float c = dot(rayToSphere, rayToSphere) - 1.0;
	float disc = b * b - c;
	if (disc < 0.0f) return false;
	else disc = sqrt(disc);
	float dist;
	if ((b - disc) > EPSILON) {
		dist = b - disc;
	} else if ((b + disc) > EPSILON) {
		dist = b + disc;
	} else {
		dist = 0.0f;
	}
	float3 objPt = -rayToSphere + dir * dist;
	float3 worldPt = transformPoint(sphere->M, objPt);
	hit->dist = length(worldPt);
	hit->normal = normalize(applyTranspose(sphere->InvM, objPt));

	return true;
}

bool intersect_scene(__constant Sphere* spheres, const Ray *ray, Hit *hit, const int sphere_count)
{
	/* initialise t to a very large number,
	so t will be guaranteed to be smaller
	when a hit with the scene occurs */

	float inf = 1e20f;
	hit->dist = inf;

	/* check if the ray intersects each sphere in the scene */
	for (int i = 0; i < sphere_count; i++) {
		Sphere sphere = spheres[i]; /* create local copy of sphere */

		Hit newHit;
		if (intersect_sphere(&sphere, ray, &newHit)) {
			if (newHit.dist > 0.0f && newHit.dist < hit->dist) {
				hit->dist = newHit.dist;
				hit->normal = newHit.normal;
			}
		}
	}
	return hit->dist < inf; /* true when ray interesects the scene */
}


/* the path tracing function */
/* computes a path (starting from the camera) with a defined number of bounces, accumulates light/color at each bounce */
/* each ray hitting a surface will be reflected in a random direction (by randomly sampling the hemisphere above the hitpoint) */
/* small optimisation: diffuse ray directions are calculated using cosine weighted importance sampling */

float3 trace(__constant Sphere* spheres, const Ray* camray, const int sphere_count, const int* seed0, const int* seed1){

	Ray ray = *camray;

	float3 accum_color = (float3)(0.0f, 0.0f, 0.0f);
	float3 mask = (float3)(1.0f, 1.0f, 1.0f);

	float t;   /* distance to intersection */
	int hitsphere_id = 0; /* index of intersected sphere */

	/* if ray misses scene, return background colour */
	Hit hit;
	if (!intersect_scene(spheres, &ray, &hit, sphere_count))
		return accum_color += mask * (float3)(0.15f, 0.15f, 0.25f);

		/* compute the hitpoint using the ray equation */
	float3 hitpoint = ray.origin + ray.dir * hit.dist;

	/* compute the surface normal and flip it if necessary to face the incoming ray */
	float3 normal = hit.normal;
	float3 normal_facing = dot(normal, ray.dir) < 0.0f ? normal : normal * (-1.0f);

	return (normal + (float3)(1,1,1))/2;
}

union Colour { float c; uchar4 components; };

__kernel void render_kernel(__constant Sphere* spheres, const int width, const int height, const int sphere_count, __global float3* output, const int hashedframenumber)
{
	unsigned int work_item_id = get_global_id(0);	/* the unique global id of the work item for the current pixel */
	unsigned int x_coord = work_item_id % width;			/* x-coordinate of the pixel */
	unsigned int y_coord = work_item_id / width;			/* y-coordinate of the pixel */

	/* seeds for random number generator */
	unsigned int seed0 = x_coord + hashedframenumber;
	unsigned int seed1 = y_coord + hashedframenumber;

	Ray camray = createCamRay(x_coord, y_coord, width, height);

	/* add the light contribution of each sample and average over all samples*/
	float3 finalcolor = trace(spheres, &camray, sphere_count, &seed0, &seed1);

	union Colour fcolour;
	fcolour.components = (uchar4)(	
		(unsigned char)(finalcolor.x * 255), 
		(unsigned char)(finalcolor.y * 255),
		(unsigned char)(finalcolor.z * 255),
		1);

	/* store the pixelcolour in the output buffer */
	output[work_item_id] = (float3)(x_coord, y_coord, fcolour.c);
}
