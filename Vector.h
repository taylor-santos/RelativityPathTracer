#pragma once
#include "Object.h"
#include <CL\cl.hpp>
#include <windows.h>


#define float3(x, y, z) {{x, y, z}}  // macro to replace ugly initializer braces
#define float4(x, y, z, w) {{x, y, z, w}}

float sqr_magnitude(const cl_float3 v);
float magnitude(const cl_float3 v);
cl_float3 normalize(const cl_float3 v);
cl_float3 operator+(const cl_float3 &v1, const cl_float3 &v2);
cl_float3 &operator+=(cl_float3 &v1, const cl_float3 &v2);
cl_float3 operator-(const cl_float3 &v1, const cl_float3 &v2);
cl_float3 operator-(const cl_float3 &v);
cl_float3 operator*(const cl_float3 &v, const float &c);
cl_float3 operator*(const float &c, const cl_float3 &v);
cl_float3 operator/(const cl_float3 &v, const float &c);
float dot(const cl_float4 &a, const cl_float4 &b);
cl_float3 cross(const cl_float3 &a, const cl_float3 &b);
cl_float3 elementwise_min(const cl_float3 &a, const cl_float3 &b);
cl_float3 elementwise_max(const cl_float3 &a, const cl_float3 &b);
bool calcInvM(Object &object);
void TRS(Object &object, cl_float3 translation, float angle, cl_float3 axis, cl_float3 scale);
void Identity(cl_float4(&M)[4]);
void Lorentz(cl_float4(&M)[4], cl_float3 v);
cl_float3 AddVelocity(cl_float3 const& v1, cl_float3 const& v2);
void MatrixMultiplyLeft(cl_float4(&A)[4], cl_float4 const(&B)[4]);
void MatrixMultiplyRight(cl_float4 const (&A)[4], cl_float4(&B)[4]);
void setLorentzBoost(Object &object, cl_float3 v);
