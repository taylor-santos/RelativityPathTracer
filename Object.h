#pragma once
#include <CL/cl.hpp>

enum objectType { SPHERE, CUBE, MESH };

struct Object
{
	cl_float4 M[4];
	cl_float4 InvM[4];
	cl_float4 Lorentz[4] = { {{1,0,0,0}},{{0,1,0,0}},{{0,0,1,0}},{0,0,0,1} };
	cl_float4 InvLorentz[4] = { {{1,0,0,0}},{{0,1,0,0}},{{0,0,1,0}},{0,0,0,1} };
	cl_float4 stationaryCam;
	cl_float3 color;
	enum objectType type;
	int meshIndex;
	int textureIndex = -1;
	int textureWidth;
	int textureHeight;
	bool light = false;
	float flashPeriod = 0;
	float flashDuration = 0;
};

extern std::vector<Object> cpu_objects;
extern std::vector<cl_float3> velocities;