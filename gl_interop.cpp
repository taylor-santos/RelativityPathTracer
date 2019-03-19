#include "gl_interop.h"
#include "Render.h"

int window_width, window_height;

// OpenGL vertex buffer object
GLuint vbo;


void initGL(int argc, char** argv) {
	// init GLUT for OpenGL viewport
	glutInit(&argc, argv);
	// specify the display mode to be RGB and single buffering
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	// specify the initial window size
	window_width = 960;
	window_height = 540;
	glutInitWindowPosition((glutGet(GLUT_SCREEN_WIDTH) - window_width) / 2,
		(glutGet(GLUT_SCREEN_HEIGHT) - window_height) / 2);
	glutInitWindowSize(window_width, window_height);
	// create the window and set title
	glutCreateWindow("Relativistic OpenCL Ray Tracer");

	// register GLUT callback function to display graphics:
	glutDisplayFunc(render);

	glutSetKeyRepeat(GLUT_KEY_REPEAT_OFF);
	glutKeyboardFunc(keyDown);
	glutKeyboardUpFunc(keyUp);
	// initialise OpenGL extensions
	glewInit();

	// initialise OpenGL
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0.0, window_width, 0.0, window_height);
}

void createVBO(GLuint* vbo)
{
	//create vertex buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	//initialise VBO
	unsigned int size = window_width * window_height * sizeof(cl_float3);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void drawGL() {

	//clear all pixels, then render from the vbo
	glClear(GL_COLOR_BUFFER_BIT);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(2, GL_FLOAT, 16, 0); // size (2, 3 or 4), type, stride, pointer
	glColorPointer(4, GL_UNSIGNED_BYTE, 16, (GLvoid*)8); // size (3 or 4), type, stride, pointer

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_POINTS, 0, window_width * window_height);
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);

	// flip backbuffer to screen
	glutSwapBuffers();
}

void Timer(int value) {
	glutPostRedisplay();
	glutTimerFunc(15, Timer, 0);
}
