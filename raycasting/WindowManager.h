#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <memory>
#include <string>

namespace WindowManager {

	int Init(int width, int height, std::string title);
	void Destroy();
	void RunLoop();
}