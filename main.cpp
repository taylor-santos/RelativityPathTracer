#include <iostream>

#include "gl_interop.h"
#include "Vector.h"
#include "Mesh.h"
#include "CLSetup.h"
#include "Object.h"
#include "Render.h"

// image buffer (not needed with real-time viewport)
cl_float4* cpu_output;
cl_int err;

void main(int argc, char** argv){
	// initialise OpenGL (GLEW and GLUT window + callback functions)
	initGL(argc, argv);
	std::cout << "OpenGL initialized \n";

	// initialise OpenCL
	initOpenCL();

	// create vertex buffer object
	createVBO(&vbo);

	// call Timer():
	Timer(0);
	
	//make sure OpenGL is finished before we proceed
	glFinish();

	inputScene();

	cl_objects = cl::Buffer(context, CL_MEM_READ_ONLY, cpu_objects.size() * sizeof(Object));
	queue.enqueueWriteBuffer(cl_objects, CL_TRUE, 0, cpu_objects.size() * sizeof(Object), cpu_objects.size() > 0 ? &cpu_objects[0] : NULL);

	cl_vertices = cl::Buffer(context, CL_MEM_READ_ONLY, theMesh.vertices.size() * sizeof(cl_float3));
	queue.enqueueWriteBuffer(cl_vertices, CL_TRUE, 0, theMesh.vertices.size() * sizeof(cl_float3), theMesh.vertices.size() > 0 ? &theMesh.vertices[0] : NULL);

	cl_normals = cl::Buffer(context, CL_MEM_READ_ONLY, theMesh.normals.size() * sizeof(cl_float3));
	queue.enqueueWriteBuffer(cl_normals, CL_TRUE, 0, theMesh.normals.size() * sizeof(cl_float3), theMesh.normals.size() > 0 ? &theMesh.normals[0] : NULL);
	
	cl_uvs = cl::Buffer(context, CL_MEM_READ_ONLY, theMesh.uvs.size() * sizeof(cl_float2));
	queue.enqueueWriteBuffer(cl_uvs, CL_TRUE, 0, theMesh.uvs.size() * sizeof(cl_float2), theMesh.uvs.size() > 0 ? &theMesh.uvs[0] : NULL);

	cl_triangles = cl::Buffer(context, CL_MEM_READ_ONLY, theMesh.triangles.size() * sizeof(unsigned int));
	queue.enqueueWriteBuffer(cl_triangles, CL_TRUE, 0, theMesh.triangles.size() * sizeof(unsigned int), theMesh.triangles.size() > 0 ? &theMesh.triangles[0] : NULL);

	cl_octrees = cl::Buffer(context, CL_MEM_READ_ONLY, theMesh.octree.size() * sizeof(Octree));
	queue.enqueueWriteBuffer(cl_octrees, CL_TRUE, 0, theMesh.octree.size() * sizeof(Octree), theMesh.octree.size() > 0 ? &theMesh.octree[0] : NULL);
	
	cl_octreeTris = cl::Buffer(context, CL_MEM_READ_ONLY, theMesh.octreeTris.size() * sizeof(int));
	queue.enqueueWriteBuffer(cl_octreeTris, CL_TRUE, 0, theMesh.octreeTris.size() * sizeof(int), theMesh.octreeTris.size() > 0 ? &theMesh.octreeTris[0] : NULL);

	cl_textures = cl::Buffer(context, CL_MEM_READ_ONLY, textures.size() * sizeof(unsigned char));
	queue.enqueueWriteBuffer(cl_textures, CL_TRUE, 0, textures.size() * sizeof(unsigned char), textures.size() > 0 ? &textures[0] : NULL);

	// create OpenCL buffer from OpenGL vertex buffer object
	cl_vbo = cl::BufferGL(context, CL_MEM_WRITE_ONLY, vbo);
	cl_vbos.push_back(cl_vbo);

	// intitialise the kernel
	initCLKernel();

	clock_start = std::chrono::high_resolution_clock::now();
	clock_prev = std::chrono::high_resolution_clock::now();

	// start rendering continuously
	glutMainLoop();

	// release memory
	cleanUp();

	system("PAUSE");
}
