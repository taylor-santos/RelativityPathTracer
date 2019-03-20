#define __CL_ENABLE_EXCEPTIONS
#include <iostream>
#include <ostream>
#include <queue>

#include "gl_interop.h"
#include "Vector.h"
#include "Mesh.h"
#include "CLSetup.h"
#include "Object.h"
#include "Render.h"

// image buffer (not needed with real-time viewport)
cl_float4* cpu_output;
cl_int err;

void json(Mesh const& mesh, const unsigned int octreeIndex, std::ostream &out, const unsigned int indent) {
	out << "{" << std::endl;
	out << std::string(indent + 1, '\t') << "\"bounds\": {" << std::endl;
	out << std::string(indent + 2, '\t') << "\"min\": \"(" << mesh.octree[octreeIndex].min.x << ", "
		<< mesh.octree[octreeIndex].min.y << ", " << mesh.octree[octreeIndex].min.z << ")\"," << std::endl;
	out << std::string(indent + 2, '\t') << "\"max\": \"(" << mesh.octree[octreeIndex].max.x << ", "
		<< mesh.octree[octreeIndex].max.y << ", " << mesh.octree[octreeIndex].max.z << ")\"" << std::endl;
	out << std::string(indent + 1, '\t') << "}," << std::endl;
	out << std::string(indent + 1, '\t') << "\"trisCount\": " << mesh.octree[octreeIndex].trisCount << "," << std::endl;
	out << std::string(indent + 1, '\t') << "\"tris\": [";
	std::string sep = "";
	for (int i = 0; i < mesh.octree[octreeIndex].trisCount; i++) {
		out << sep << mesh.octreeTris[mesh.octree[octreeIndex].trisIndex + i];
		sep = ", ";
	}
	out << "]," << std::endl;
	out << std::string(indent + 1, '\t') << "\"Mathematica\": \"Show[Graphics3D[{Opacity[0.5],Cuboid[{"
		<< mesh.octree[octreeIndex].min.x << "," << mesh.octree[octreeIndex].min.y << "," << mesh.octree[octreeIndex].min.z << "},{"
		<< mesh.octree[octreeIndex].max.x << "," << mesh.octree[octreeIndex].max.y << "," << mesh.octree[octreeIndex].max.z
		<< "}]}],Graphics3D[Triangle[{";
	sep = "";
	for (int i = 0; i < mesh.octree[octreeIndex].trisCount; i++) {
		int triIndex = mesh.octreeTris[mesh.octree[octreeIndex].trisIndex + i];
		out << sep << "{{" << mesh.vertices[mesh.triangles[9 * triIndex + 3 * 0]].x
			<< "," << mesh.vertices[mesh.triangles[9 * triIndex + 3 * 0]].y
			<< "," << mesh.vertices[mesh.triangles[9 * triIndex + 3 * 0]].z << "},{"
			<< mesh.vertices[mesh.triangles[9 * triIndex + 3 * 1]].x
			<< "," << mesh.vertices[mesh.triangles[9 * triIndex + 3 * 1]].y
			<< "," << mesh.vertices[mesh.triangles[9 * triIndex + 3 * 1]].z << "},{"
			<< mesh.vertices[mesh.triangles[9 * triIndex + 3 * 2]].x
			<< "," << mesh.vertices[mesh.triangles[9 * triIndex + 3 * 2]].y
			<< "," << mesh.vertices[mesh.triangles[9 * triIndex + 3 * 2]].z << "}}";
		sep = ",";
	}
	out << "}]]]\"," << std::endl;
	out << std::string(indent + 1, '\t') << "\"children\": {";
	if (mesh.octree[octreeIndex].children[0] != -1) {
		sep = "";
		for (int i = 0; i < 8; i++) {
			out << sep << std::endl;
			out << std::string(indent + 2, '\t') << "\"" << i << "\": ";
			json(mesh, mesh.octree[octreeIndex].children[i], out, indent + 2);
			sep = ",";
		}
		out << std::endl << std::string(indent + 1, '\t');
	}
	out << "}" << std::endl;
	out << std::string(indent, '\t') << "}";
}

void main(int argc, char** argv) {

	inputScene();
	std::ofstream file("parallel.json");
	json(theMesh, 0, file, 0);
	file.close();
	// initialise OpenGL (GLEW and GLUT window + callback functions)
	initGL(argc, argv);
	
	// initialise OpenCL
	initOpenCL("relativity_kernel.cl");
	/*
	const int n = 1000000;
	cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * n);
	cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * n);
	cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * n);
	int *A = new int[n];
	int *B = new int[n];
	int *C = new int[n];
	std::fill(A, A + n, 1);
	std::fill(B, B + n, 2);
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int)*n, A);
	queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int)*n, B);
	cl::Kernel add;
	try {
		add = cl::Kernel(program, "add2");
	}
	catch (cl::Error &e) {
		std::cerr << e.what() << ": " << e.err() << std::endl;
		throw e;
	}
	std::size_t global_work_size = n;
	std::size_t local_work_size;
	try {
		//std::size_t local_work_size = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
		kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
	}
	catch (cl::Error &e) {
		std::cerr << e.what() << ": " << e.err() << std::endl;
		throw e;
	}
	*/
	// create vertex buffer object
	createVBO(&vbo);

	// call Timer():
	Timer(0);

	//make sure OpenGL is finished before we proceed
	glFinish();


	try {
		if (cpu_objects.size() > 0) {
			cl_objects = cl::Buffer(context, CL_MEM_READ_ONLY, cpu_objects.size() * sizeof(Object));
			queue.enqueueWriteBuffer(cl_objects, CL_TRUE, 0, cpu_objects.size() * sizeof(Object), &cpu_objects[0]);
		}

		if (theMesh.vertices.size() > 0) {
			cl_vertices = cl::Buffer(context, CL_MEM_READ_ONLY, theMesh.vertices.size() * sizeof(cl_float3));
			queue.enqueueWriteBuffer(cl_vertices, CL_TRUE, 0, theMesh.vertices.size() * sizeof(cl_float3), &theMesh.vertices[0]);
		}
		if (theMesh.normals.size() > 0) {
			cl_normals = cl::Buffer(context, CL_MEM_READ_ONLY, theMesh.normals.size() * sizeof(cl_float3));
			queue.enqueueWriteBuffer(cl_normals, CL_TRUE, 0, theMesh.normals.size() * sizeof(cl_float3), &theMesh.normals[0]);
		}
		if (theMesh.uvs.size() > 0) {
			cl_uvs = cl::Buffer(context, CL_MEM_READ_ONLY, theMesh.uvs.size() * sizeof(cl_float2));
			queue.enqueueWriteBuffer(cl_uvs, CL_TRUE, 0, theMesh.uvs.size() * sizeof(cl_float2), &theMesh.uvs[0]);
		}
		if (theMesh.triangles.size() > 0) {
			cl_triangles = cl::Buffer(context, CL_MEM_READ_ONLY, theMesh.triangles.size() * sizeof(unsigned int));
			queue.enqueueWriteBuffer(cl_triangles, CL_TRUE, 0, theMesh.triangles.size() * sizeof(unsigned int), &theMesh.triangles[0]);
		}
		if (theMesh.octree.size() > 0) {
			cl_octrees = cl::Buffer(context, CL_MEM_READ_ONLY, theMesh.octree.size() * sizeof(Octree));
			queue.enqueueWriteBuffer(cl_octrees, CL_TRUE, 0, theMesh.octree.size() * sizeof(Octree), &theMesh.octree[0]);
		}
		if (theMesh.octreeTris.size() > 0) {
			cl_octreeTris = cl::Buffer(context, CL_MEM_READ_ONLY, theMesh.octreeTris.size() * sizeof(int));
			queue.enqueueWriteBuffer(cl_octreeTris, CL_TRUE, 0, theMesh.octreeTris.size() * sizeof(int), &theMesh.octreeTris[0]);
		}
		if (textures.size() > 0) {
			cl_textures = cl::Buffer(context, CL_MEM_READ_ONLY, textures.size() * sizeof(unsigned char));
			queue.enqueueWriteBuffer(cl_textures, CL_TRUE, 0, textures.size() * sizeof(unsigned char), &textures[0]);
		}
		// create OpenCL buffer from OpenGL vertex buffer object
		cl_vbo = cl::BufferGL(context, CL_MEM_WRITE_ONLY, vbo);
		cl_vbos.push_back(cl_vbo);
	}
	catch (cl::Error &e) {
		std::cerr << e.what() << ": " << e.err() << std::endl;
		throw e;
	}

	// intitialise the kernel
	initRelativityKernel();

	clock_start = std::chrono::high_resolution_clock::now();
	clock_prev = std::chrono::high_resolution_clock::now();

	// start rendering continuously
	glutMainLoop();

	// release memory
	cleanUp();

	system("PAUSE");
}
