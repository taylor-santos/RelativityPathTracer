/* OpenCL based simple sphere path tracer by Sam Lapere, 2016*/
/* based on smallpt by Kevin Beason */
/* http://raytracey.blogspot.com */


__constant double EPSILON = 0.0000001; /* required to compensate for limited float precision */
__constant int MSAASAMPLES = 2;

typedef struct Ray{
	double3 origin;
	double3 dir;
} Ray;

typedef struct Sphere{
	double4 M[4];
	double4 InvM[4];
	float3 color;
	float3 emission;
} Sphere;

typedef struct Hit {
	double dist;
	double3 normal;
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

bool intersect_sphere(const Sphere* sphere, const Ray* ray, Hit *hit)
{
	double3 rayToSphere = -transformPoint(sphere->InvM, ray->origin);
	double3 dir = normalize(transformDirection(sphere->InvM, ray->dir));
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
	double3 worldPt = transformPoint(sphere->M, objPt);
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

float3 trace(__constant Sphere* spheres, const Ray* camray, const int sphere_count){

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
	double3 hitpoint = ray.origin + ray.dir * hit.dist;

	/* compute the surface normal and flip it if necessary to face the incoming ray */
	double3 normal = hit.normal;
	double3 normal_facing = dot(normal, ray.dir) < 0.0f ? normal : normal * (-1.0f);

	float3 color = (float3)((float)normal_facing.x, (float)normal_facing.y, (float)normal_facing.z);
	color = (color + (float3)(1, 1, 1)) / 2;
	return color;
}

union Colour { float c; uchar4 components; };

__kernel void render_kernel(__constant Sphere* spheres, const int width, const int height, const int sphere_count, __global float3* output, const int hashedframenumber)
{
	unsigned int work_item_id = get_global_id(0);	/* the unique global id of the work item for the current pixel */
	unsigned int x_coord = work_item_id % width;			/* x-coordinate of the pixel */
	unsigned int y_coord = work_item_id / width;			/* y-coordinate of the pixel */
	
	float3 finalcolor = (float3)(0, 0, 0);

	for (int y = 0; y < MSAASAMPLES; y++) {
		for (int x = 0; x < MSAASAMPLES; x++) {
			Ray camray = createCamRay((double)x_coord + (double)x/MSAASAMPLES, (double)y_coord + (double)y/ MSAASAMPLES, width, height);

			finalcolor += trace(spheres, &camray, sphere_count);
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
