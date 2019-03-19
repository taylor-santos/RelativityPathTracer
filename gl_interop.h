#define GLEW_STATIC
#define FREEGLUT_STATIC

#include <GL\glew.h>
#include <GL\freeglut.h>
#include <CL\cl.hpp>

//#define GL_SHARING_EXTENSION "cl_khr_gl_sharing"

extern int window_width, window_height;

// OpenGL vertex buffer object
extern GLuint vbo;

void initGL(int argc, char** argv);
void createVBO(GLuint* vbo);
void drawGL();
void Timer(int value);
