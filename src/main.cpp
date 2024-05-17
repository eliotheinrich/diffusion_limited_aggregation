#include <random>
#include <iostream>
#include "DiffusionController.hpp"
#include "DiffusionPhysicsModel.hpp"
#include "QuadCollection.hpp"
#include "Shader.hpp"
#include <unistd.h>

#include <nlohmann/json.hpp>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
  glViewport(0, 0, width, height);
}

void detect_escape(GLFWwindow *window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, true);
  }
}

void debug_callback(GLFWwindow *window) {
  //Shader shader("vertex_shader.vs", "fragment_shader.fs");
  //QuadCollection quads(&shader);

  //auto [vertices, indices] = make_quad(-0., -0., 1.0, 1.0, 0);
  //rotate(vertices, 0.785);
  //quads.add_vertices(vertices, indices, {1.0, 1.0, 0.0});

  //quads.draw();
}

nlohmann::json load_config(int argc, char** argv) {
  if (argc == 1) {
    return nlohmann::json();
  } else if (argc == 2) {
    return load_json(argv[1]);
  } else {
    throw std::invalid_argument("Need to provide a filename for config.");
  }
}

Color DEFAULT_BACKGROUND_COLOR = {0.0, 0.0, 0.0};

int main(int argc, char** argv) {
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_FALSE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

  const unsigned int WIDTH = 900;
  const unsigned int HEIGHT = 900;
  GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Diffusion Limited Aggregation", NULL, NULL);
  if (window == NULL) {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window); //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD" << std::endl;
    return -1;
  }  

  glViewport(0, 0, WIDTH, HEIGHT);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);  

  auto config = load_config(argc, argv);
  Color background_color = parse_color(config["background_color"], DEFAULT_BACKGROUND_COLOR);
  int target_fps = static_cast<int>(config.value("fps", 60));

  DiffusionController controller(config);
  controller.add_mouse_button_callback(window);

  float t1, t2, dt;
  float target_dt = 1.0/target_fps;

  while(!glfwWindowShouldClose(window)) {
    t1 = glfwGetTime();
    glClearColor(background_color.r, background_color.g, background_color.b, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    detect_escape(window);

    controller.process_input(window);
    controller.update();
    controller.draw();

    debug_callback(window);

    glfwSwapBuffers(window);
    glfwPollEvents();    


    t2 = glfwGetTime();
    dt = t2 - t1;
    if (dt < target_dt) {
      usleep((target_dt - dt)*1e6);
    }
  }

  glfwTerminate();
  return 0;
}
