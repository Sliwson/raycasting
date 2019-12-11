#include <helper_gl.h>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <GL/wglew.h>
#endif
#if defined(__APPLE__) || defined(__MACOSX)
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_functions.h>
#include <helper_cuda.h>

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>
#include <chrono>

#include "Geometry.h"
#include "raycasting_kernel.h"

//OpenGL PBO and texture "names"
GLuint gl_PBO, gl_Tex, gl_Shader;
struct cudaGraphicsResource *cuda_pbo_resource; // handles OpenGL-CUDA exchange

//Source image on the host side
uchar4 *h_Src = 0;

// Destination image on the GPU side
uchar4 *d_dst = NULL;

//Original image width and height
int imageW = 800, imageH = 600;

// User interface variables
int lastx = 0;
int lasty = 0;
bool leftClicked = false;
bool middleClicked = false;
bool rightClicked = false;

#define REFRESH_DELAY 2 //ms
#define BUFFER_DATA(i) ((char *)0 + i)

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
// This is specifically to enable the application to enable/disable vsync
typedef BOOL (WINAPI *PFNWGLSWAPINTERVALFARPROC)(int);

int vSyncInterval = 0;
void setVSync(int interval)
{
	printf("Setting vsync to %d\n", interval);
    if (WGL_EXT_swap_control)
    {
        wglSwapIntervalEXT = (PFNWGLSWAPINTERVALFARPROC)wglGetProcAddress("wglSwapIntervalEXT");
        wglSwapIntervalEXT(interval);
    }
}
#endif

void computeFPS()
{
	static int fpsCounter = 0;
	static auto timer = std::chrono::high_resolution_clock::now();
	
	fpsCounter++;

	const auto milisecondsElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - timer).count();
	if (milisecondsElapsed > 1000)
	{
		char title[256];
		sprintf(title, "Raycasting (%d fps)", fpsCounter);
		glutSetWindowTitle(title);

		fpsCounter = 0;
		timer = std::chrono::high_resolution_clock::now();
	}
}

float getDt()
{
	static int fpsCounter = 0;
	static auto timer = std::chrono::high_resolution_clock::now();

	fpsCounter++;

	const auto milisecondsElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - timer).count();
	timer = std::chrono::high_resolution_clock::now();

	return milisecondsElapsed / 1000.f;
}

void renderImage(float gameTimer)
{
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_dst, &num_bytes, cuda_pbo_resource));

	RenderScene(d_dst, imageW, imageH, gameTimer);

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

// OpenGL display function
void displayFunc(void)
{
	static float gameTimer = 0.f;

	gameTimer += getDt();
    renderImage(gameTimer);

    // load texture from PBO
    glBindTexture(GL_TEXTURE_2D, gl_Tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageW, imageH, GL_RGBA, GL_UNSIGNED_BYTE, BUFFER_DATA(0));

    // fragment program is required to display floating point texture
    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, gl_Shader);
    glEnable(GL_FRAGMENT_PROGRAM_ARB);
    glDisable(GL_DEPTH_TEST);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(0.0f, 0.0f);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(1.0f, 0.0f);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(0.0f, 1.0f);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_FRAGMENT_PROGRAM_ARB);

    glutSwapBuffers();

    computeFPS();
}

void cleanup()
{
    if (h_Src)
    {
        free(h_Src);
        h_Src = 0;
    }

    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    glDeleteBuffers(1, &gl_PBO);
    glDeleteTextures(1, &gl_Tex);
    glDeleteProgramsARB(1, &gl_Shader);
}

// OpenGL keyboard function
void keyboardFunc(unsigned char k, int, int)
{
    switch (k)
    {
        case '\033':
        case 'q':
        case 'Q':
            printf("Shutting down...\n");

            #if defined(__APPLE__) || defined(MACOSX)
            exit(EXIT_SUCCESS);
            #else
            glutDestroyWindow(glutGetWindow());
            return;
            #endif
            break;

           break;

        case '4':   // Left arrow key
            break;

        case '8':   // Up arrow key
            break;

        case '6':   // Right arrow key
            break;

        case '2':   // Down arrow key
            break;

        case '+':
            break;

        case '-':
            break;

		case 'V':
		case 'v':
			vSyncInterval = (vSyncInterval + 1) % 2;
			setVSync(vSyncInterval);
			break;

        default:
            break;
    }
} 

// OpenGL mouse click function
void clickFunc(int button, int state, int x, int y)
{
    if (button == 0)
        leftClicked = !leftClicked;

    if (button == 1)
        middleClicked = !middleClicked;

    if (button == 2)
        rightClicked = !rightClicked;

    int modifiers = glutGetModifiers();

    if (leftClicked && (modifiers & GLUT_ACTIVE_SHIFT))
    {
        leftClicked = 0;
        middleClicked = 1;
    }

    if (state == GLUT_UP)
    {
        leftClicked = 0;
        middleClicked = 0;
    }

    lastx = x;
    lasty = y;
} 

// OpenGL mouse motion function
void motionFunc(int x, int y)
{
    double fx = (double)(x - lastx) / 50.0 / (double)(imageW);
    double fy = (double)(lasty - y) / 50.0 / (double)(imageH);
}

void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    }
}

// gl_Shader for displaying floating-point texture
static const char *shader_code =
    "!!ARBfp1.0\n"
    "TEX result.color, fragment.texcoord, texture[0], 2D; \n"
    "END";

GLuint compileASMShader(GLenum program_type, const char *code)
{
    GLuint program_id;
    glGenProgramsARB(1, &program_id);
    glBindProgramARB(program_type, program_id);
    glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei) strlen(code), (GLubyte *) code);

    GLint error_pos;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

    if (error_pos != -1)
    {
        const GLubyte *error_string;
        error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
        fprintf(stderr, "Program error at position: %d\n%s\n", (int)error_pos, error_string);
        return 0;
    }

    return program_id;
}

void initOpenGLBuffers(int w, int h)
{
    // delete old buffers
    if (h_Src)
    {
        free(h_Src);
        h_Src = 0;
    }

    if (gl_Tex)
    {
        glDeleteTextures(1, &gl_Tex);
        gl_Tex = 0;
    }

    if (gl_PBO)
    {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &gl_PBO);
        gl_PBO = 0;
    }

    // allocate new buffers
    h_Src = (uchar4 *)malloc(w * h * 4);

    printf("Creating GL texture...\n");
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &gl_Tex);
    glBindTexture(GL_TEXTURE_2D, gl_Tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_Src);
    printf("Texture created.\n");

    printf("Creating PBO...\n");
    glGenBuffers(1, &gl_PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, w * h * 4, h_Src, GL_STREAM_COPY);
 
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO,
                                                 cudaGraphicsMapFlagsWriteDiscard));
    printf("PBO created.\n");

    // load shader program
    gl_Shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}

void reshapeFunc(int w, int h)
{
    glViewport(0, 0, w, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    if (w!=0 && h!=0)  // Do not call when window is minimized that is when width && height == 0
        initOpenGLBuffers(w, h);

    imageW = w;
    imageH = h;

    glutPostRedisplay();
}

void initGL(int *argc, char **argv)
{
    printf("Initializing GLUT...\n");
    glutInit(argc, argv);

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(imageW, imageH);
    glutInitWindowPosition(0, 0);
    glutCreateWindow(argv[0]);

    glutDisplayFunc(displayFunc);
    glutKeyboardFunc(keyboardFunc);
    glutMouseFunc(clickFunc);
    glutMotionFunc(motionFunc);
    glutReshapeFunc(reshapeFunc);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	glutSetWindowTitle("Raycasting (0 fps)");

    if (!isGLVersionSupported(1,5) ||
        !areGLExtensionsSupported("GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object"))
    {
        fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
        exit(EXIT_SUCCESS);
    }

    printf("OpenGL window created.\n");
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif
    initGL(&argc, argv);
    initOpenGLBuffers(imageW, imageH);

    printf("Starting GLUT main loop...\n");
    printf("\n");

    printf("Press [q] to exit\n");
    printf("\n");

#if defined (__APPLE__) || defined(MACOSX)
    atexit(cleanup);
#else
    glutCloseFunc(cleanup);
#endif

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    setVSync(vSyncInterval);
#endif

    glutMainLoop();
}
