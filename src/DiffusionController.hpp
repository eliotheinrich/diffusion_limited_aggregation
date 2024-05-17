#pragma once

#include <stdexcept>
#include <unistd.h>
#include <future>
#include <vector>
#include <map>
#include <fstream>
#include <algorithm>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <nlohmann/json.hpp>

#include "ParticlePhysicsModel.hpp"
#include "DiffusivePhysicsModel.hpp"
#include "Shader.hpp"
#include "QuadCollection.hpp"

Color DEFAULT_PARTICLE_COLOR = {1.0, 1.0, 1.0, 0.5};

std::vector<Color> DEFAULT_CLUSTER_COLORS = {
  {1.0, 0.0, 0.0, 1.0},
  {0.0, 1.0, 0.0, 1.0},
  {0.0, 0.0, 1.0, 1.0}
};

struct DLAConfig {
  InitialState initial_state;
  PhysicsModelType physics_model_type;
  size_t n;
  size_t m;

  bool paused;
  
  bool show_particles;

  std::vector<Color> cluster_colors;
  Color particle_color;

  int seed;

  DLAConfig()=default;

  static DLAConfig from_json(nlohmann::json& json) {
    DLAConfig config;

    config.n = static_cast<size_t>(json.value("n", 100));
    config.m = static_cast<size_t>(json.value("m", config.n));

    auto parse_initial_state = [](const std::string& s) {
      if (s == "central_cluster") {
        return InitialState::Central;
      } else if (s == "boundary_cluster") {
        return InitialState::Boundary;
      } else {
        return InitialState::Central;
      }
    };
    InitialState initial_state = parse_initial_state(static_cast<std::string>(json.value("initial_state", "central_cluster")));

    auto parse_physics_model = [](const std::string& s) {
      if (s == "particle") {
        return PhysicsModelType::Particle;
      } else if (s == "diffusive") {
        return PhysicsModelType::Diffusion;
      } else {
        return PhysicsModelType::Particle;
      }
    };

    PhysicsModelType model_type = parse_physics_model(static_cast<std::string>(json.value("physics_model", "particle")));
    config.physics_model_type = model_type;

    config.show_particles = static_cast<bool>(json.value("show_particles", false));
    config.particle_color = parse_color(json["particle_color"], DEFAULT_PARTICLE_COLOR);

    auto cluster_color_values = json["cluster_colors"];
    if (cluster_color_values.is_null() || !cluster_color_values.is_array()) {
      config.cluster_colors = DEFAULT_CLUSTER_COLORS;
    } else {
      std::vector<Color> cluster_colors;
      for (const auto& color : cluster_color_values) {
        cluster_colors.push_back(parse_color(color));
      }
      config.cluster_colors = cluster_colors;
    }

    config.seed = static_cast<int>(json.value("seed", -1));

    config.paused = static_cast<bool>(json.value("paused", false));

    return config;
  }
};

nlohmann::json load_json(const std::string& filename) {
  std::ifstream file(filename);

  if (!file.is_open()) {
    throw std::invalid_argument("Failed to open file.");
  }

  std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

  auto json = nlohmann::json::parse(content);
  return json;
}

// Global future so that async mouse hover function does not hang when going out of scope
std::future<void> hover_future;

class DiffusionController {
  public:
    size_t n;
    size_t m;

    DiffusionController(nlohmann::json& json) {
      glGenVertexArrays(1, &VAO);
      glGenBuffers(1, &VBO);
      glGenBuffers(1, &EBO);
      glGenTextures(1, &ctexture); 
      glGenTextures(1, &ptexture); 

      // Configure vertex attributes
      glBindVertexArray(VAO);
      glBindBuffer(GL_ARRAY_BUFFER, VBO);

      glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
      glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
      glEnableVertexAttribArray(0);
      glEnableVertexAttribArray(1);

      float vertices[] = {
        // positions         // texture coords
         1.0f,  1.0f, 0.0f,   1.0f, 1.0f,   // top right
         1.0f, -1.0f, 0.0f,   1.0f, 0.0f,   // bottom right
        -1.0f, -1.0f, 0.0f,   0.0f, 0.0f,   // bottom left
        -1.0f,  1.0f, 0.0f,   0.0f, 1.0f    // top left 
      };

      unsigned int indices[] = {
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
      };

      glBindBuffer(GL_ARRAY_BUFFER, VBO);
      glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);

      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_DYNAMIC_DRAW);

      // Activate shader
      shader = Shader("vertex_texture_shader.vs", "fragment_texture_shader.fs");
      shader.use();
      shader.set_int("texture0", 0);
      shader.set_int("texture1", 1);

      // Configure textures
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, ctexture);  
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_2D, ptexture);  
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

      // Unbind buffers
      glBindVertexArray(0);
      glBindBuffer(GL_ARRAY_BUFFER, 0);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
      glBindTexture(GL_TEXTURE_2D, 0);  

      DLAConfig config = DLAConfig::from_json(json);
      n = config.n;
      m = config.m;

      show_particles = config.show_particles;

      cluster_colors = config.cluster_colors;
      particle_color = config.particle_color;

      color_idx = 0;

      paused = config.paused;

      if (config.physics_model_type == PhysicsModelType::Particle) {
        ParticlePhysicsModelConfig physics_config = ParticlePhysicsModelConfig::from_json(json);
        physics_model = std::make_shared<ParticlePhysicsModel>(physics_config);
      } else if (config.physics_model_type == PhysicsModelType::Diffusion) {
        DiffusivePhysicsModelConfig physics_config = DiffusivePhysicsModelConfig::from_json(json);
        physics_model = std::make_shared<DiffusivePhysicsModel>(physics_config);
      }
      physics_model->set_cluster_color(cluster_colors[color_idx]);
      physics_model->set_particle_color(particle_color);

      if (config.initial_state == InitialState::Central) {
        //initialize_central_cluster();
      } else if (config.initial_state == InitialState::Boundary) {
        //initialize_boundary_cluster();
      }

      init_keymap();
    }



    ~DiffusionController() {
      glDeleteVertexArrays(1, &VAO);
      glDeleteBuffers(1, &VBO);
      glDeleteBuffers(1, &EBO);
    }

    void store_state() {
      //stored_data = DiffusionFieldData(*this);
    }

    void load_state() {
      //stored_data.load_data(*this);
    }

    double update_key_time(int key_code, double time) {
      if (!prevtime.contains(key_code)) {
        prevtime[key_code] = 0.0;
      }

      double t = time - prevtime[key_code];
      prevtime[key_code] = time;
      return t;
    }

    void init_keymap() {
      keymap[GLFW_KEY_P]     = [](DiffusionController& df, GLFWwindow* window) { df.show_particles = !df.show_particles; };
      keymap[GLFW_KEY_C]     = [](DiffusionController& df, GLFWwindow* window) { df.advance_color(); };
      keymap[GLFW_KEY_S]     = [](DiffusionController& df, GLFWwindow* window) { df.store_state(); };
      keymap[GLFW_KEY_R]     = [](DiffusionController& df, GLFWwindow* window) { df.load_state(); };
      keymap[GLFW_KEY_SPACE] = [](DiffusionController& df, GLFWwindow* window) { df.paused = !df.paused; };
    }

    void process_input(GLFWwindow *window) {
      double time = glfwGetTime();
      double delay = 0.02;

      for (auto const& [key, func] : keymap) {
        if (glfwGetKey(window, key) == GLFW_PRESS && update_key_time(key, time) > delay) {
          func(*this, window);
        }
      }

      physics_model->process_input(window);
    }

    enum IntersectionDirection { Right, Left, Top, Bottom };
    IntersectionDirection invert(IntersectionDirection dir) const {
      if (dir == IntersectionDirection::Right) {
        return IntersectionDirection::Left;
      } else if (dir == IntersectionDirection::Left) {
        return IntersectionDirection::Right;
      } else if (dir == IntersectionDirection::Bottom) {
        return IntersectionDirection::Top;
      } else {
        return IntersectionDirection::Bottom;
      }
    }

    struct Intersection {
      IntersectionDirection direction;
      float x;
      float y;
    };

    // Given three collinear points, the function checks if point q lies on line segment 'pr'
    bool on_segment(float p_x, float p_y, float q_x, float q_y, float r_x, float r_y) {
      if (q_x <= std::max(p_x, r_x) && q_x >= std::min(p_x, r_x) &&
          q_y <= std::max(p_y, r_y) && q_y >= std::min(p_y, r_y)) {
        return true;
      }
      return false;
    }

    // 0 if p, q, and r are collinear
    // 1 if they are clockwise
    // 2 if they are counterclockwise
    int orientation(float p_x, float p_y, float q_x, float q_y, float r_x, float r_y) const {
      float val = (q_y - p_y) * (r_x - q_x) - (q_x - p_x) * (r_y - q_y);
      if (val == 0)  {
        return 0;  // Collinear
      } else {
        return (val > 0) ? 1 : 2; // Clockwise or counterclockwise
      }
    }

    // Function to check if segments 'p1q1' and 'p2q2' intersect
    std::optional<std::pair<float, float>> segment_intersection(float p1_x, float p1_y, float q1_x, float q1_y, float p2_x, float p2_y, float q2_x, float q2_y) const {
      int o1 = orientation(p1_x, p1_y, q1_x, q1_y, p2_x, p2_y);
      int o2 = orientation(p1_x, p1_y, q1_x, q1_y, q2_x, q2_y);
      int o3 = orientation(p2_x, p2_y, q2_x, q2_y, p1_x, p1_y);
      int o4 = orientation(p2_x, p2_y, q2_x, q2_y, q1_x, q1_y);

      // General case
      if (o1 != o2 && o3 != o4) {
        float x_intersect = ((p1_x * q1_y - p1_y * q1_x) * (p2_x - q2_x) - (p1_x - q1_x) * (p2_x * q2_y - p2_y * q2_x)) /
          ((p1_x - q1_x) * (p2_y - q2_y) - (p1_y - q1_y) * (p2_x - q2_x));
        float y_intersect = ((p1_x * q1_y - p1_y * q1_x) * (p2_y - q2_y) - (p1_y - q1_y) * (p2_x * q2_y - p2_y * q2_x)) /
          ((p1_x - q1_x) * (p2_y - q2_y) - (p1_y - q1_y) * (p2_x - q2_x));
        return std::make_pair(x_intersect, y_intersect);
      }

      // If none of the cases
      return std::nullopt;
    }



    std::vector<Intersection> grid_intersection(const Point& p, float x1, float y1, float x2, float y2) const {
      float tile_width = 2.0/n;
      float tile_height = 2.0/m;

      float left = ((float) p.x / n - 0.5)*2.0;
      float bottom = ((float) p.y / m - 0.5)*2.0;
      float right = left + tile_width;
      float top = bottom + tile_height;

      std::vector<Intersection> intersections;

      // Bottom
      auto bottom_intersection = segment_intersection(left, bottom, right, bottom, x1, y1, x2, y2);
      if (bottom_intersection.has_value()) {
        auto [x, y] = bottom_intersection.value();
        intersections.push_back({IntersectionDirection::Bottom, x, y});
      }

      // Top
      auto top_intersection = segment_intersection(left, top, right, top, x1, y1, x2, y2);
      if (top_intersection.has_value()) {
        auto [x, y] = top_intersection.value();
        intersections.push_back({IntersectionDirection::Top, x, y});
      }

      // Left
      auto left_intersection = segment_intersection(left, bottom, left, top, x1, y1, x2, y2);
      if (left_intersection.has_value()) {
        auto [x, y] = left_intersection.value();
        intersections.push_back({IntersectionDirection::Left, x, y});
      }

      // Right
      auto right_intersection = segment_intersection(right, bottom, right, top, x1, y1, x2, y2);
      if (right_intersection.has_value()) {
        auto [x, y] = right_intersection.value();
        intersections.push_back({IntersectionDirection::Right, x, y});
      }

      return intersections;
    }

    std::vector<Point> get_points_between(float x1, float y1, float x2, float y2) const {
      Point p1 = Point::get_point_from_pos(x1, y1, n, m);
      Point p2 = Point::get_point_from_pos(x2, y2, n, m);

      if (p1.x == p2.x && p1.y == p2.y) {
        return std::vector<Point>{p1};
      }

      std::vector<Point> points{p1};
      Point p = p1;

      float endpoint_x = x1;
      float endpoint_y = y1;
      std::vector<Intersection> intersections = grid_intersection(p, endpoint_x, endpoint_y, x2, y2);
      if (intersections.size() != 1) {
        throw std::invalid_argument("gridpoint p1 has multiple intersections\n");
      }

      auto [direction, x, y] = intersections[0];

      bool keep_going = true;
      while (keep_going) {
        if (direction == IntersectionDirection::Bottom) {
          p = Point(p.x, p.y-1);
        } else if (direction == IntersectionDirection::Top) {
          p = Point(p.x, p.y+1);
        } else if (direction == IntersectionDirection::Left) {
          p = Point(p.x-1, p.y);
        } else if (direction == IntersectionDirection::Right) {
          p = Point(p.x+1, p.y);
        }

        points.push_back(p);
        intersections = grid_intersection(p, endpoint_x, endpoint_y, x2, y2);

        if (intersections.size() == 2) {
          endpoint_x = (intersections[0].x + intersections[1].x)/2.0;
          endpoint_y = (intersections[0].y + intersections[1].y)/2.0;

          if (intersections[0].direction != invert(direction)) {
            direction = intersections[0].direction;
          } else {
            direction = intersections[1].direction;
          }
        } else if (intersections.size() == 1) {
          keep_going = false;
        } else {
          std::string error_message = "Error with grid tracing; found " + std::to_string(intersections.size()) + " intersections.";
          throw std::runtime_error(error_message);
        }
      }

      return points;
    }
    
    void add_mouse_button_callback(GLFWwindow* window) {
      glfwSetWindowUserPointer(window, static_cast<void*>(this));

      auto mouse_callback = [](GLFWwindow* window, int button, int action, int mods) {
        auto self = static_cast<DiffusionController*>(glfwGetWindowUserPointer(window));
        glfwSetInputMode(window, GLFW_STICKY_MOUSE_BUTTONS, GLFW_TRUE);

        auto mouse_hover_event = [window, self]() {
          bool last_pos_exists = false;
          float xl, yl;
          while (self->lbutton_down) {
            int width, height;
            glfwGetWindowSize(window, &width, &height);

            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);
            xpos = (xpos/width - 0.5)*2.0;
            ypos = (-ypos/height + 0.5)*2.0;


            std::vector<Point> points;
            if (last_pos_exists) {
              points = self->get_points_between(xpos, ypos, xl, yl);
            } else {
              points = {Point::get_point_from_pos(xpos, ypos, self->n, self->m)};
            }

            for (auto const& p : points) {
              self->physics_model->insert_cluster(p);
            }

            xl = xpos;
            yl = ypos;
            last_pos_exists = true;
          }
        };


        if (button == GLFW_MOUSE_BUTTON_LEFT) {
          if (action == GLFW_PRESS) {
            self->lbutton_down = true;
            hover_future = std::async(std::launch::async, mouse_hover_event); 
          } else if (action == GLFW_RELEASE) {
            self->lbutton_down = false;
          }
        }

        return;
      };

      glfwSetMouseButtonCallback(window, mouse_callback);
    }

    void advance_color() {
      color_idx = (color_idx + 1) % cluster_colors.size();
      physics_model->set_cluster_color(cluster_colors[color_idx]);
    }

    void draw() {
      shader.use();
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

      auto cluster_texture = physics_model->get_cluster_texture();
      glBindTexture(GL_TEXTURE_2D, ctexture);  
      glActiveTexture(GL_TEXTURE0);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, n, m, 0, GL_RGBA, GL_FLOAT, cluster_texture.data());

      if (show_particles) {
        auto particle_texture = physics_model->get_particle_texture();
        glBindTexture(GL_TEXTURE_2D, ptexture);  
        glActiveTexture(GL_TEXTURE1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, n, m, 0, GL_RGBA, GL_FLOAT, particle_texture.data());
      }

      glBindVertexArray(VAO);
      glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
      glBindVertexArray(0);

      glBindTexture(GL_TEXTURE_2D, 0);

      physics_model->draw();
    }

    void update() {
      if (paused) {
        return;
      }

      physics_model->diffusion_step();
    }

  private:
    unsigned int VAO;
    unsigned int VBO;
    unsigned int EBO;
    unsigned int ctexture;
    unsigned int ptexture;

    std::shared_ptr<DiffusionPhysicsModel> physics_model;

    KeyMap<DiffusionController> keymap;
    std::map<decltype(GLFW_KEY_ESCAPE), double> prevtime;

    std::vector<Color> cluster_colors;
    Color particle_color;

    bool lbutton_down;

    size_t color_idx;
    std::vector<size_t> colors;

    bool paused;

    Shader shader;
    bool show_particles;
};
