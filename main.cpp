#define __CL_ENABLE_EXCEPTIONS
#include <iostream>
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

void main(int argc, char** argv) {

	inputScene();
	for (const int meshIndex : theMesh.meshIndices) {
		using namespace std;
		using namespace cl;

		vector<cl_float3> vertices = theMesh.vertices;
		vector<unsigned int> triangles = theMesh.triangles;

		std::queue<vector<unsigned int>> trisQueue;
		std::queue<vector<unsigned int>> trisIndexQueue;
		std::queue<unsigned int> octreeIndexQueue;

		vector<unsigned int> triVerts(triangles.size() / 3);
		vector<unsigned int> triIndex(triangles.size() / 9);
		int count = 0, n = 3;
		copy_if(triangles.begin(), triangles.end(), triVerts.begin(),
			[&count, &n](int i)->bool { return count++ % n == 0; });
		for (int i = 0; i < triIndex.size(); i++) {
			triIndex[i] = 9 * i;
		}
		trisQueue.push(triVerts);
		trisIndexQueue.push(triIndex);
		octreeIndexQueue.push(0);

		vector<Platform> platforms;
		Platform::get(&platforms);
		Platform platform = platforms[0];

		vector<Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
		Device device = devices[0];

		Context context = Context(device);

		std::ifstream file("octree_kernel.cl");
		if (!file) {
			std::cout << "\nNo OpenCL file found!" << std::endl << "Exiting..." << std::endl;
			system("PAUSE");
			exit(1);
		}
		std::string source{ std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>() };

		const char* kernel_source = source.c_str();

		Program program = Program(context, kernel_source);
		program.build({ device }, "");

		Kernel kernel = Kernel(program, "parallel_add");

		while (!trisQueue.empty()) {
			vector<unsigned int> newTriangles = trisQueue.front();
			trisQueue.pop();
			unsigned int octreeIndex = octreeIndexQueue.front();
			octreeIndexQueue.pop();
			int numTriangles = newTriangles.size() / 3;
			unsigned char *cpuOutput = new unsigned char[numTriangles];

			Buffer vertBuffer = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vertices.size() * sizeof(cl_float3), &vertices[0]);
			Buffer triBuffer = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, newTriangles.size() * sizeof(unsigned int), &newTriangles[0]);
			Buffer clOutput = Buffer(context, CL_MEM_WRITE_ONLY, numTriangles * sizeof(unsigned char), NULL);

			cl_float3 min = theMesh.octree[0].min;
			cl_float3 max = theMesh.octree[0].max;
			cl_float3 extents = max - min;
			cl_float3 half_extents = extents / 2;
			cl_float3 ex = float3(half_extents.x, 0, 0);
			cl_float3 ey = float3(0, half_extents.y, 0);
			cl_float3 ez = float3(0, 0, half_extents.z);

			kernel.setArg(0, vertBuffer);
			kernel.setArg(1, triBuffer);
			kernel.setArg(2, min);
			kernel.setArg(3, max);
			kernel.setArg(4, clOutput);
			kernel.setArg(5, numTriangles);
			CommandQueue queue = CommandQueue(context, device);

			std::size_t global_work_size = numTriangles;
			std::size_t local_work_size = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
			if (global_work_size % local_work_size != 0)
				global_work_size = (global_work_size / local_work_size + 1) * local_work_size;

			queue.enqueueNDRangeKernel(kernel, NULL, global_work_size, local_work_size);

			queue.enqueueReadBuffer(clOutput, CL_TRUE, 0, numTriangles * sizeof(unsigned char), cpuOutput);
			for (int x = 0; x < 2; x++) {
				for (int y = 0; y < 2; y++) {
					for (int z = 0; z < 2; z++) {
						Octree child;
						child.min = theMesh.octree[octreeIndex].min + ex * x + ey * y + ez * z;
						child.max = child.min + half_extents;
						child.trisIndex = theMesh.octreeTris.size();
						child.trisCount = 0;
						theMesh.octree[octreeIndex].children[z + 2 * y + 4 * x] = theMesh.octree.size();
						theMesh.octree.push_back(child);
						for (int tri = 0; tri < numTriangles; tri++) {
							if (cpuOutput[tri] & 1 << z + 2 * y + 4 * x) {
								theMesh.octreeTris.push_back(tri);
								theMesh.octree[theMesh.octree.size() - 1].trisCount++;
							}
						}
					}
				}
			}
			delete[] cpuOutput;
		}
	}

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
