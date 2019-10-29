#include "WindowManager.h"

namespace callbacks {

	void error_callback(int error, const char* description)
	{
		fprintf(stderr, "Error: %d: %s\n", error, description);
	}

	void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
	{
		if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
			glfwSetWindowShouldClose(window, GLFW_TRUE);
	}
}

namespace {
	
	GLFWwindow* window = nullptr;

	void SetCallbacks()
	{
		glfwSetErrorCallback(callbacks::error_callback);
		glfwSetKeyCallback(window, callbacks::key_callback);
	}
}

namespace WindowManager {

	int Init(int width, int height, std::string title)
	{
		if (!glfwInit())
			return -1;

		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
		window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
		
		if (!window)
			return -1;

		glfwMakeContextCurrent(window);
		SetCallbacks();
		if (!gladLoadGL())
			return -1;
		
		glfwSwapInterval(1);

		return 0;
	}

	void Destroy()
	{
		if (window)
			glfwDestroyWindow(window);

		glfwTerminate();
	}

	void RunLoop()
	{
		while (!glfwWindowShouldClose(window))
		{
			glClear(GL_COLOR_BUFFER_BIT);
			glfwSwapBuffers(window);
			glfwPollEvents();
		}
	}
}
