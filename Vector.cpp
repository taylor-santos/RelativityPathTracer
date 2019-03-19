#include "Vector.h"
#include "Object.h"

float sqr_magnitude(const cl_float3 v) {
	return v.x*v.x + v.y*v.y + v.z*v.z;
}

float magnitude(const cl_float3 v) {
	return sqrt(sqr_magnitude(v));
}

cl_float3 normalize(const cl_float3 v) {
	float m = magnitude(v);
	return float3(v.x / m, v.y / m, v.z / m);
}

cl_float3 operator+(const cl_float3 &v1, const cl_float3 &v2) {
	return float3(
		v1.x + v2.x,
		v1.y + v2.y,
		v1.z + v2.z
	);
}

cl_float3 &operator+=(cl_float3 &v1, const cl_float3 &v2) {
	v1 = v1 + v2;
	return v1;
}

cl_float3 operator-(const cl_float3 &v1, const cl_float3 &v2) {
	return float3(
		v1.x - v2.x,
		v1.y - v2.y,
		v1.z - v2.z
	);
}

cl_float3 operator-(const cl_float3 &v) {
	return float3(
		-v.x,
		-v.y,
		-v.z
	);
}

cl_float3 operator*(const cl_float3 &v, const float &c) {
	return float3(
		v.x * c,
		v.y * c,
		v.z * c
	);
}
cl_float3 operator*(const float &c, const cl_float3 &v) {
	return v * c;
}

cl_float3 operator/(const cl_float3 &v, const float &c) {
	return float3(
		v.x / c,
		v.y / c,
		v.z / c
	);
}

float dot(const cl_float4 &a, const cl_float4 &b) {
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

cl_float3 cross(const cl_float3 &a, const cl_float3 &b) {
	return float3(
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	);
}

cl_float3 elementwise_min(const cl_float3 &a, const cl_float3 &b) {
	return float3(
		min(a.x, b.x),
		min(a.y, b.y),
		min(a.z, b.z)
	);
}

cl_float3 elementwise_max(const cl_float3 &a, const cl_float3 &b) {
	return float3(
		max(a.x, b.x),
		max(a.y, b.y),
		max(a.z, b.z)
	);
}


bool calcInvM(Object &object) {
	float A2323 = object.M[2].z * object.M[3].w - object.M[2].w * object.M[3].z;
	float A1323 = object.M[2].y * object.M[3].w - object.M[2].w * object.M[3].y;
	float A1223 = object.M[2].y * object.M[3].z - object.M[2].z * object.M[3].y;
	float A0323 = object.M[2].x * object.M[3].w - object.M[2].w * object.M[3].x;
	float A0223 = object.M[2].x * object.M[3].z - object.M[2].z * object.M[3].x;
	float A0123 = object.M[2].x * object.M[3].y - object.M[2].y * object.M[3].x;
	float A2313 = object.M[1].z * object.M[3].w - object.M[1].w * object.M[3].z;
	float A1313 = object.M[1].y * object.M[3].w - object.M[1].w * object.M[3].y;
	float A1213 = object.M[1].y * object.M[3].z - object.M[1].z * object.M[3].y;
	float A2312 = object.M[1].z * object.M[2].w - object.M[1].w * object.M[2].z;
	float A1312 = object.M[1].y * object.M[2].w - object.M[1].w * object.M[2].y;
	float A1212 = object.M[1].y * object.M[2].z - object.M[1].z * object.M[2].y;
	float A0313 = object.M[1].x * object.M[3].w - object.M[1].w * object.M[3].x;
	float A0213 = object.M[1].x * object.M[3].z - object.M[1].z * object.M[3].x;
	float A0312 = object.M[1].x * object.M[2].w - object.M[1].w * object.M[2].x;
	float A0212 = object.M[1].x * object.M[2].z - object.M[1].z * object.M[2].x;
	float A0113 = object.M[1].x * object.M[3].y - object.M[1].y * object.M[3].x;
	float A0112 = object.M[1].x * object.M[2].y - object.M[1].y * object.M[2].x;

	float det =
		object.M[0].x * (object.M[1].y * A2323 - object.M[1].z * A1323 + object.M[1].w * A1223)
		- object.M[0].y * (object.M[1].x * A2323 - object.M[1].z * A0323 + object.M[1].w * A0223)
		+ object.M[0].z * (object.M[1].x * A1323 - object.M[1].y * A0323 + object.M[1].w * A0123)
		- object.M[0].w * (object.M[1].x * A1223 - object.M[1].y * A0223 + object.M[1].z * A0123);
	if (det == 0.0f) {
		return false;
	}
	det = 1 / det;

	object.InvM[0] = float4(
		det * (object.M[1].y * A2323 - object.M[1].z * A1323 + object.M[1].w * A1223),
		det * -(object.M[0].y * A2323 - object.M[0].z * A1323 + object.M[0].w * A1223),
		det * (object.M[0].y * A2313 - object.M[0].z * A1313 + object.M[0].w * A1213),
		det * -(object.M[0].y * A2312 - object.M[0].z * A1312 + object.M[0].w * A1212)
	);
	object.InvM[1] = float4(
		det * -(object.M[1].x * A2323 - object.M[1].z * A0323 + object.M[1].w * A0223),
		det * (object.M[0].x * A2323 - object.M[0].z * A0323 + object.M[0].w * A0223),
		det * -(object.M[0].x * A2313 - object.M[0].z * A0313 + object.M[0].w * A0213),
		det * (object.M[0].x * A2312 - object.M[0].z * A0312 + object.M[0].w * A0212)
	);
	object.InvM[2] = float4(
		det * (object.M[1].x * A1323 - object.M[1].y * A0323 + object.M[1].w * A0123),
		det * -(object.M[0].x * A1323 - object.M[0].y * A0323 + object.M[0].w * A0123),
		det * (object.M[0].x * A1313 - object.M[0].y * A0313 + object.M[0].w * A0113),
		det * -(object.M[0].x * A1312 - object.M[0].y * A0312 + object.M[0].w * A0112)
	);
	object.InvM[3] = float4(
		det * -(object.M[1].x * A1223 - object.M[1].y * A0223 + object.M[1].z * A0123),
		det * (object.M[0].x * A1223 - object.M[0].y * A0223 + object.M[0].z * A0123),
		det * -(object.M[0].x * A1213 - object.M[0].y * A0213 + object.M[0].z * A0113),
		det * (object.M[0].x * A1212 - object.M[0].y * A0212 + object.M[0].z * A0112)
	);
	return true;
}

void TRS(Object &object, cl_float3 translation, float angle, cl_float3 axis, cl_float3 scale) {
	cl_float3 R[3] = { float3(1,0,0), float3(0,1,0), float3(0,0,1) };
	if (angle != 0) {
		float c = cos(angle);
		float s = sin(angle);
		cl_float3 u = normalize(axis);
		R[0] = float3(c + u.x*u.x*(1 - c), u.x*u.y*(1 - c) - u.z*s, u.x*u.z*(1 - c) + u.y*s);
		R[1] = float3(u.y*u.x*(1 - c) + u.z*s, c + u.y*u.y*(1 - c), u.y*u.z*(1 - c) - u.x*s);
		R[2] = float3(u.z*u.x*(1 - c) - u.y*s, u.z*u.y*(1 - c) + u.x*s, c + u.z*u.z*(1 - c));
	}
	object.M[0] = float4(R[0].x * scale.x, R[0].y * scale.y, R[0].z * scale.z, translation.x);
	object.M[1] = float4(R[1].x * scale.x, R[1].y * scale.y, R[1].z * scale.z, translation.y);
	object.M[2] = float4(R[2].x * scale.x, R[2].y * scale.y, R[2].z * scale.z, translation.z);
	object.M[3] = float4(0, 0, 0, 1);
	calcInvM(object);
}

void Identity(cl_float4(&M)[4]) {
	M[0] = float4(1, 0, 0, 0);
	M[1] = float4(0, 1, 0, 0);
	M[2] = float4(0, 0, 1, 0);
	M[3] = float4(0, 0, 0, 1);
}

void Lorentz(cl_float4(&M)[4], cl_float3 v) {
	float gamma = 1.0f / sqrt(1.0f - dot(v, v));
	float vSqr = dot(v, v);
	if (vSqr == 0) {
		Identity(M);
	}
	else {
		M[0] = float4(gamma, -v.x * gamma, -v.y * gamma, -v.z * gamma);
		M[1] = float4(-v.x * gamma, (gamma - 1.0f) * v.x * v.x / vSqr + 1.0f, (gamma - 1.0f) * v.x * v.y / vSqr, (gamma - 1.0f) * v.x * v.z / vSqr);
		M[2] = float4(-v.y * gamma, (gamma - 1.0f) * v.y * v.x / vSqr, (gamma - 1.0f) * v.y * v.y / vSqr + 1.0f, (gamma - 1.0f) * v.y * v.z / vSqr);
		M[3] = float4(-v.z * gamma, (gamma - 1.0f) * v.z * v.x / vSqr, (gamma - 1.0f) * v.z * v.y / vSqr, (gamma - 1.0f) * v.z * v.z / vSqr + 1.0f);
	}
}

cl_float3 AddVelocity(cl_float3 const& v1, cl_float3 const& v2) {
	float gamma_v = 1.0f / sqrt(1 - dot(v1, v1));
	float a_v = sqrt(1 - dot(v1, v1));
	return 1.0f / (1.0f + dot(v2, v1))*(v1 + v2 + gamma_v / (1.0 + gamma_v)*cross(v1, cross(v1, v2)));
}

void MatrixMultiplyLeft(cl_float4(&A)[4], cl_float4 const(&B)[4]) {
	for (int i = 0; i < 4; i++) {
		A[i] = float4(
			dot(A[i], float4(B[0].x, B[1].x, B[2].x, B[3].x)),
			dot(A[i], float4(B[0].y, B[1].y, B[2].y, B[3].y)),
			dot(A[i], float4(B[0].z, B[1].z, B[2].z, B[3].z)),
			dot(A[i], float4(B[0].w, B[1].w, B[2].w, B[3].w))
		);
	}
}

void MatrixMultiplyRight(cl_float4 const (&A)[4], cl_float4(&B)[4]) {
	cl_float4 out[4];
	for (int i = 0; i < 4; i++) {
		out[i] = float4(
			dot(A[i], float4(B[0].x, B[1].x, B[2].x, B[3].x)),
			dot(A[i], float4(B[0].y, B[1].y, B[2].y, B[3].y)),
			dot(A[i], float4(B[0].z, B[1].z, B[2].z, B[3].z)),
			dot(A[i], float4(B[0].w, B[1].w, B[2].w, B[3].w))
		);
	}
	B[0] = out[0];
	B[1] = out[1];
	B[2] = out[2];
	B[3] = out[3];
}

void setLorentzBoost(Object &object, cl_float3 v) {
	Lorentz(object.Lorentz, v);
	float gamma = 1.0f / sqrt(1.0f - dot(v, v));
	object.InvLorentz[0] = float4(gamma, v.x * gamma, v.y * gamma, v.z * gamma);
	object.InvLorentz[1] = object.Lorentz[1];
	object.InvLorentz[1].x *= -1;
	object.InvLorentz[2] = object.Lorentz[2];
	object.InvLorentz[2].x *= -1;
	object.InvLorentz[3] = object.Lorentz[3];
	object.InvLorentz[3].x *= -1;
}
