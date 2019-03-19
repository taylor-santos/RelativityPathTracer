#include "Render.h"
#include "gl_interop.h"
#include "CLSetup.h"
#include "Vector.h"
#include <iostream>
#include <map>

unsigned int framenumber = 0;
bool downKeys[9] = { false, false, false, false, false, false, false, false, false }; // w a s d q e r space i
cl_float3 cameraVelocity = { {0,0,0} };
cl_float4 cameraPos = { {0,0,0,0} };
bool stopTime = true;
int interval = -1;
bool changedTime = false;
bool changedInterval = false;
float currTime = 0;
cl_float3 white_point;
float ambient;
std::chrono::time_point<std::chrono::high_resolution_clock> clock_start, clock_end, clock_prev;
Mesh theMesh;
std::vector<unsigned char> textures;
std::vector<int> textureValues; // {index, width, height}


void keyDown(unsigned char key, int x, int y) {
	switch (key) {
	case 'w':
		downKeys[0] = true;
		break;
	case 'a':
		downKeys[1] = true;
		break;
	case 's':
		downKeys[2] = true;
		break;
	case 'd':
		downKeys[3] = true;
		break;
	case 'q':
		downKeys[4] = true;
		break;
	case 'e':
		downKeys[5] = true;
		break;
	case 'r':
		downKeys[6] = true;
		break;
	case ' ':
		downKeys[7] = true;
		break;
	case 'i':
		downKeys[8] = true;
		break;
	}
}

void keyUp(unsigned char key, int x, int y) {
	switch (key) {
	case 'w':
		downKeys[0] = false;
		break;
	case 'a':
		downKeys[1] = false;
		break;
	case 's':
		downKeys[2] = false;
		break;
	case 'd':
		downKeys[3] = false;
		break;
	case 'q':
		downKeys[4] = false;
		break;
	case 'e':
		downKeys[5] = false;
		break;
	case 'r':
		downKeys[6] = false;
		break;
	case ' ':
		downKeys[7] = false;
		break;
	case 'i':
		downKeys[8] = false;
		break;
	}
}

void render() {

	framenumber++;

	clock_end = std::chrono::high_resolution_clock::now();
	int ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock_end - clock_start).count();
	currTime = ms / 1000.0f;
	int frame_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock_end - clock_prev).count();
	//std::cout << 1000.0f / frame_ms << " fps\taverage: " << 1000.0f*framenumber / ms << std::endl;
	clock_prev = std::chrono::high_resolution_clock::now();

	int new_window_width = glutGet(GLUT_WINDOW_WIDTH),
		new_window_height = glutGet(GLUT_WINDOW_HEIGHT);
	if (new_window_width != window_width || new_window_height != window_height) {
		window_width = new_window_width;
		window_height = new_window_height;

		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluOrtho2D(0.0f, window_width, 0.0f, window_height);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		unsigned int size = window_width * window_height * sizeof(cl_float3);
		glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
		cl_vbo = cl::BufferGL(context, CL_MEM_WRITE_ONLY, vbo);
		cl_vbos[0] = cl_vbo;
		kernel.setArg(11, window_width);
		kernel.setArg(12, window_height);
		kernel.setArg(14, cl_vbo);
	}
	float s = ms / 1000.0f;

	cl_float4 cameraLorentz[4];
	cl_float4 cameraInvLorentz[4];

	if (downKeys[7]) {
		if (!changedTime) {
			changedTime = true;
			stopTime = !stopTime;
			if (stopTime) std::cout << "PAUSED" << std::endl;
			else std::cout << "UNPAUSED" << std::endl;
		}
	}
	else {
		changedTime = false;
	}

	if (downKeys[8]) {
		if (!changedInterval) {
			changedInterval = true;
			interval = -!interval;
			kernel.setArg(13, interval);
			std::cout << "Interval: " << interval << std::endl;
		}
	}
	else {
		changedInterval = false;
	}

	if (downKeys[6]) {
		cameraVelocity = float3(0, 0, 0);
		std::cout.precision(5);
		std::cout << "V: (" << std::fixed
			<< cameraVelocity.x << ", "
			<< cameraVelocity.y << ", "
			<< cameraVelocity.z << ")" << std::endl;
	}
	else {
		cl_float3 dV = float3(0, 0, 0);
		if (downKeys[0]) dV += float3(0, 0, 1);  // W
		if (downKeys[1]) dV += float3(-1, 0, 0); // A
		if (downKeys[2]) dV += float3(0, 0, -1); // S
		if (downKeys[3]) dV += float3(1, 0, 0);  // D
		if (downKeys[4]) dV += float3(0, -1, 0); // Q
		if (downKeys[5]) dV += float3(0, 1, 0); // E

		if (magnitude(dV) != 0) {
			dV = tanh(frame_ms / 5000.0f) *  normalize(dV);
			cameraVelocity = AddVelocity(cameraVelocity, dV);
			std::cout.precision(5);
			std::cout << "V: (" << std::fixed
				<< cameraVelocity.x << ", "
				<< cameraVelocity.y << ", "
				<< cameraVelocity.z << ")" << std::endl;
		}

	}
	if (!stopTime) cameraPos += float4(frame_ms / 1000.0f, 0, 0, 0);

	Lorentz(cameraLorentz, cameraVelocity);
	Lorentz(cameraInvLorentz, -cameraVelocity);

	for (Object &object : cpu_objects) {
		Identity(object.Lorentz);
		Identity(object.InvLorentz);
	}

	cl_float3 dir = float3(1, 0, 1);
	dir = normalize(dir);

	for (int i = 0; i < cpu_objects.size(); i++) {
		setLorentzBoost(cpu_objects[i], velocities[i]);
		MatrixMultiplyLeft(cpu_objects[i].Lorentz, cameraInvLorentz);
		MatrixMultiplyRight(cameraLorentz, cpu_objects[i].InvLorentz);
		cpu_objects[i].stationaryCam = float4(
			dot(cpu_objects[i].Lorentz[0], cameraPos),
			dot(cpu_objects[i].Lorentz[1], cameraPos),
			dot(cpu_objects[i].Lorentz[2], cameraPos),
			dot(cpu_objects[i].Lorentz[3], cameraPos)
		);
	}

	queue.enqueueWriteBuffer(cl_objects, CL_TRUE, 0, cpu_objects.size() * sizeof(Object), cpu_objects.size() > 0 ? &cpu_objects[0] : NULL);
	kernel.setArg(0, cl_objects);

	runKernel();

	drawGL();

}

void inputScene() {
	white_point = float3(1, 1, 1);
	ambient = 1.0;
	std::string line;
	bool done = false;
	do {
		std::getline(std::cin, line);
		char *str = strdup(line.c_str());
		char *tok = strtok(str, " ");
		char *endptr;
		char *curr;
		float args[10];
		while (!done && tok) {

			switch (tok[0]) {
			case 'O':
				if (strlen(tok) < 2) {
					std::cerr << "Object command missing argument" << std::endl;
					break;
				}
				switch (tok[1]) {
				case 's':
					cpu_objects.push_back(Object());
					velocities.push_back(float3(0, 0, 0));
					cpu_objects.back().type = SPHERE;
					break;
				case 'c':
					cpu_objects.push_back(Object());
					velocities.push_back(float3(0, 0, 0));
					cpu_objects.back().type = CUBE;
					break;
				case 'm':
					if (strlen(tok) != 3) {
						std::cerr << "Object mesh command missing argument" << std::endl;
						break;
					}
					cpu_objects.push_back(Object());
					velocities.push_back(float3(0, 0, 0));
					cpu_objects.back().type = MESH;
					cpu_objects.back().meshIndex = atoi(tok + 2);
					break;
				default:
					std::cerr << "Object command unrecognized argument: \"" << tok + 1 << "\"" << std::endl;
				}
				break;
			case 'p':
				if (cpu_objects.size() == 0) {
					std::cerr << "Object must be defined before applying a transformation" << std::endl;
					break;
				}
				if (strlen(tok) < 2) {
					std::cerr << "Transformation command missing argument" << std::endl;
					break;
				}
				curr = tok + 1;
				for (int arg = 0; arg < 10; arg++) {
					args[arg] = strtod(curr, &endptr);
					curr = endptr + 1;
				}
				TRS(cpu_objects.back(), float3(args[0], args[1], args[2]), args[3], float3(args[4], args[5], args[6]), float3(args[7], args[8], args[9]));
				break;
			case 'c':
				if (cpu_objects.size() == 0) {
					std::cerr << "Object must be defined before applying a transformation" << std::endl;
					break;
				}
				if (strlen(tok) < 2) {
					std::cerr << "Color command missing argument" << std::endl;
					break;
				}
				curr = tok + 1;
				for (int arg = 0; arg < 3; arg++) {
					args[arg] = strtod(curr, &endptr);
					curr = endptr + 1;
				}
				cpu_objects.back().color = float3(args[0], args[1], args[2]);
				break;
			case 't':
				if (cpu_objects.size() == 0) {
					std::cerr << "Object must be defined before applying a texture" << std::endl;
					break;
				}
				if (strlen(tok) < 2) {
					std::cerr << "Texture command missing argument" << std::endl;
					break;
				}
				cpu_objects.back().textureIndex = atoi(tok + 1);
				break;
			case 'l':
				if (cpu_objects.size() == 0) {
					std::cerr << "Object must be defined before applying a light" << std::endl;
					break;
				}
				if (strlen(tok) < 2) {
					std::cerr << "Velocity command missing argument" << std::endl;
					break;
				}
				cpu_objects.back().light = atoi(tok + 1);
				break;
			case 'v':
				if (cpu_objects.size() == 0) {
					std::cerr << "Object must be defined before applying a velocity" << std::endl;
					break;
				}
				if (strlen(tok) < 2) {
					std::cerr << "Velocity command missing argument" << std::endl;
					break;
				}
				curr = tok + 1;
				for (int arg = 0; arg < 3; arg++) {
					args[arg] = strtod(curr, &endptr);
					curr = endptr + 1;
				}
				velocities.back() = float3(args[0], args[1], args[2]);
				break;
			case 'f':
				if (cpu_objects.size() == 0) {
					std::cerr << "Object must be defined before applying a periodic flash" << std::endl;
					break;
				}
				if (strlen(tok) < 2) {
					std::cerr << "Flash command missing argument" << std::endl;
					break;
				}
				curr = tok + 1;
				for (int arg = 0; arg < 2; arg++) {
					args[arg] = strtod(curr, &endptr);
					curr = endptr + 1;
				}
				cpu_objects.back().flashPeriod = args[0];
				cpu_objects.back().flashDuration = args[1];
				break;
			case 'T':
				if (strlen(tok) < 2) {
					std::cerr << "Texture command missing argument" << std::endl;
					break;
				}
				if (!ReadTexture(tok + 1)) {
					exit(EXIT_FAILURE);
				}
				break;
			case 'M':
				if (strlen(tok) < 2) {
					std::cerr << "Mesh command missing argument" << std::endl;
					break;
				}
				if (!ReadOBJ(tok + 1, theMesh)) {
					exit(EXIT_FAILURE);
				}
				break;
			case 'A':
				if (strlen(tok) < 2) {
					std::cerr << "Ambient command missing argument" << std::endl;
					break;
				}
				ambient = atof(tok + 1);
				break;
			case 'W':
				if (strlen(tok) < 2) {
					std::cerr << "White-point command missing argument" << std::endl;
					break;
				}
				curr = tok + 1;
				for (int arg = 0; arg < 3; arg++) {
					args[arg] = strtod(curr, &endptr);
					curr = endptr + 1;
				}
				white_point = float3(args[0], args[1], args[2]);
				break;
			case 'I':
				interval = 0;
				break;
			case 'R':
				done = true;
				break;
			default:
				std::cerr << "Unrecognized command: \"" << tok << "\"" << std::endl;
			}
			tok = strtok(NULL, " ");
		}
		free(str);
	} while (!done);
	for (Object &object : cpu_objects) {
		int index = object.textureIndex;
		if (index != -1) {
			if (3 * (index + 1) > textureValues.size()) {
				std::cerr << "Error: Texture index " << index << " out of range";
				exit(EXIT_FAILURE);
			}
			object.textureIndex = textureValues[3 * index + 0];
			object.textureWidth = textureValues[3 * index + 1];
			object.textureHeight = textureValues[3 * index + 2];
		}

		if (object.type == MESH) {
			index = object.meshIndex;
			if (index < 0 || index >= theMesh.meshIndices.size()) {
				std::cerr << "Error: Mesh index " << index << " out of range";
				exit(EXIT_FAILURE);
			}
			object.meshIndex = theMesh.meshIndices[index];
		}
	}

	queue.enqueueWriteBuffer(cl_objects, CL_TRUE, 0, cpu_objects.size() * sizeof(Object), cpu_objects.size() > 0 ? &cpu_objects[0] : NULL);
}

bool ReadTexture(std::string path) {
	using namespace cimg_library;
	cimg::exception_mode(0);
	try {
		CImg<unsigned char> image(path.c_str());
		image.permute_axes("cxyz");
		textureValues.push_back(textures.size());
		textureValues.push_back(image._height); // Texture width
		textureValues.push_back(image._depth);  // Texture height
		textures.insert(textures.end(), image._data, image._data + 3 * image._height * image._depth);
		return true;
	}
	catch (CImgException &e) {
		std::cerr << e.what() << std::endl;
		return false;
	}
}

bool ReadOBJ(std::string path, Mesh &mesh) {
	if (path.substr(path.size() - 4, 4) != ".obj") return false;
	std::ifstream file(path);
	if (!file) {
		perror("Error opening OBJ file");
		return false;
	}
	std::string line;
	std::map<int, std::vector<int>> vertToTrisMap;
	int lineno = 0;
	int firstTriIndex = mesh.triangles.size();
	int firstVertIndex = mesh.vertices.size();
	int firstNormIndex = mesh.normals.size();
	int firstUVIndex = mesh.uvs.size();
	while (std::getline(file, line)) {
		std::istringstream stream(line);
		std::string prefix;
		stream >> prefix;
		if (prefix == "v") {
			cl_float3 vert;
			stream >> vert.x >> vert.y >> vert.z;
			if (stream.fail()) {
				std::cerr << "Error reading OBJ file \"" << path
					<< "\": Invalid syntax on line " << lineno << std::endl;
				return false;
			}
			mesh.vertices.push_back(vert);
		}
		else if (prefix == "vt") {
			cl_float2 uv;
			stream >> uv.x >> uv.y;
			if (stream.fail()) {
				std::cerr << "Error reading OBJ file \"" << path
					<< "\": Invalid syntax on line " << lineno << std::endl;
				return false;
			}
			mesh.uvs.push_back(uv);
		}
		else if (prefix == "vn") {
			cl_float3 norm;
			stream >> norm.x >> norm.y >> norm.z;
			if (stream.fail()) {
				std::cerr << "Error reading OBJ file \"" << path
					<< "\": Invalid syntax on line " << lineno << std::endl;
				return false;
			}
			norm = normalize(norm);
			mesh.normals.push_back(norm);
		}
		else if (prefix == "f") {
			std::string tri;
			int triIndex = mesh.triangles.size() / 9;
			for (int i = 0; i < 3; i++) {
				stream >> tri;
				std::istringstream vertstream(tri);
				std::string vert, uv, norm;
				std::getline(vertstream, vert, '/');
				int vertIndex = stoul(vert) - 1 + firstVertIndex;
				if (!std::getline(vertstream, uv, '/')) {
					uv = "1";
				}
				if (!std::getline(vertstream, norm, '/')) {
					norm = "1";
					vertToTrisMap[vertIndex].push_back(triIndex);
				}
				mesh.triangles.push_back(vertIndex);
				mesh.triangles.push_back(stoul(uv) - 1 + firstUVIndex);
				mesh.triangles.push_back(stol(norm) - 1 + firstNormIndex);
			}
		}
		lineno++;
	}
	for (const auto& kv : vertToTrisMap) {
		int vertIndex = kv.first;
		auto triList = kv.second;
		cl_float3 N = float3(0, 0, 0);
		for (int triIndex : triList) {
			int AIndex = mesh.triangles[9 * triIndex + 3 * 0];
			int BIndex = mesh.triangles[9 * triIndex + 3 * 1];
			int CIndex = mesh.triangles[9 * triIndex + 3 * 2];
			cl_float3 A = mesh.vertices[AIndex];
			cl_float3 B = mesh.vertices[BIndex];
			cl_float3 C = mesh.vertices[CIndex];
			// Don't normalize: the cross product is proportional to the area of the triangle,
			// and we want the normal contribution to be proportional to the area as well.
			N += cross(B - A, C - A);
			if (AIndex == vertIndex) {
				mesh.triangles[2 + 9 * triIndex + 3 * 0] = mesh.normals.size();
			}
			else if (BIndex == vertIndex) {
				mesh.triangles[2 + 9 * triIndex + 3 * 1] = mesh.normals.size();
			}
			else if (CIndex == vertIndex) {
				mesh.triangles[2 + 9 * triIndex + 3 * 2] = mesh.normals.size();
			}
		}
		mesh.normals.push_back(normalize(N));
	}
	int newOctreeIndex = mesh.octree.size();
	mesh.meshIndices.push_back(newOctreeIndex);
	mesh.GenerateOctree(firstTriIndex);
	return true;
}
