#include <stdio.h>

#include "WindowManager.h"
#include <memory>

int main()
{
	if (WindowManager::Init(1200, 800, "Raycasting"))
	{
		WindowManager::Destroy();
		return -1;
	}

	WindowManager::RunLoop();
	WindowManager::Destroy();

	return 0;
}
