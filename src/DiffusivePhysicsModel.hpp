#pragma once

#include "DiffusionPhysicsModel.hpp"
#include "QuadCollection.hpp"

float sigmoid(float x) {
  return 1.0/(1.0 + std::exp(-x));
}


struct DiffusivePhysicsModelConfig {
  size_t n;
  size_t m;

  bool insert_left;
  bool insert_right;
  bool insert_top;
  bool insert_bottom;

  bool draw_gradients;

  double initial_density;
  size_t bin_width;
  float epsilon;

  int seed;

  DiffusivePhysicsModelConfig()=default;

  static DiffusivePhysicsModelConfig from_json(nlohmann::json& json) {
    DiffusivePhysicsModelConfig config;

    config.n = static_cast<size_t>(json.value("n", 100));
    config.m = static_cast<size_t>(json.value("m", config.n));

    config.initial_density = static_cast<double>(json.value("initial_density", 0.0));

    InitialState initial_state = InitialState::Boundary;

    bool default_insert_left =   true; //initial_state != InitialState::Boundary;
    bool default_insert_right =  true; //initial_state != InitialState::Boundary;
    bool default_insert_bottom = true; //initial_state != InitialState::Boundary;
    bool default_insert_top =    true;

    config.insert_left = static_cast<bool>(json.value("insert_left", default_insert_left));
    config.insert_right = static_cast<bool>(json.value("insert_right", default_insert_right));
    config.insert_top = static_cast<bool>(json.value("insert_top", default_insert_top));
    config.insert_bottom = static_cast<bool>(json.value("insert_bottom", default_insert_bottom));

    config.bin_width = static_cast<size_t>(json.value("bin_width", config.n/32));
    config.epsilon = static_cast<float>(json.value("epsilon", 1e-5));

    config.draw_gradients = static_cast<bool>(json.value("draw_gradients", false));
    
    int seed = static_cast<int>(json.value("seed", -1));
    if (seed == -1) {
      thread_local std::random_device rd;
      config.seed = rd();
    } else {
      config.seed = seed;
    }

    return config;
  }
};

Color GRADIENT_COLOR = {1.0, 1.0, 0.0};

class DiffusivePhysicsModel : public DiffusionPhysicsModel {
  public:
    std::vector<float> current_field;
    std::vector<float> previous_field;
    std::vector<float> cluster_texture;
    std::vector<int> clusters;

    DiffusivePhysicsModel(const DiffusivePhysicsModelConfig& config) : DiffusionPhysicsModel(config.seed) {
      n = config.n;
      m = config.m;

      bin_width = config.bin_width;
      resolution_x = n/bin_width;
      resolution_y = m/bin_width;

      epsilon = config.epsilon;

      insert_left = config.insert_left;
      insert_right = config.insert_right;
      insert_top = config.insert_top;
      insert_bottom = config.insert_bottom;

      target_particle_density = config.initial_density;

      draw_gradients = config.draw_gradients;
      gradients = std::vector<std::pair<float, float>>(resolution_x*resolution_y);

      current_field = std::vector<float>(resolution_x*resolution_y, 0.0);
      previous_field = current_field;

      clusters = std::vector<int>(n*m);
      cluster_texture = std::vector<float>(4*n*m, 0.0);

      uniform_initial_condition();

      all_points = std::vector<Point>(0);
      for (size_t x = 0; x < n; x++) {
        for (size_t y = 0; y < m; y++) {
          all_points.push_back(Point(x, y));
        }
      }

      shader = Shader("vertex_shader.vs", "fragment_shader.fs");

      init_keymap();
    }

    virtual void draw() override {
      if (draw_gradients) {
        std::vector<float> vertices;
        std::vector<unsigned int> indices;

        float thickness = 1.0/resolution_x;

        int step_x = resolution_x / 20;
        int step_y = resolution_y / 20;
        for (int x = 0; x < resolution_x; x += step_x) {
          for (int y = 0; y < resolution_y; y += step_y) {
            float pos_x = (float(x) / resolution_x - 0.5)*2.0;
            float pos_y = (float(y) / resolution_y - 0.5)*2.0;

            size_t i = Point::to_index(x, y, resolution_x, resolution_y);
            auto [grad_x, grad_y] = gradients[i];

            float grad_len = std::sqrt(grad_x*grad_x + grad_y*grad_y)/max_gradient/20.0;
            auto [new_vertices, new_indices] = make_quad(pos_x, pos_y, grad_len, thickness, vertices.size() / 3);

            float angle = std::atan2(grad_x, grad_y);
            rotate(new_vertices, 1.57079632679 + angle);

            vertices.insert(vertices.end(), new_vertices.begin(), new_vertices.end());
            indices.insert(indices.end(), new_indices.begin(), new_indices.end());
          }
        }

        QuadCollection quads(&shader);
        quads.add_vertices(vertices, indices, GRADIENT_COLOR);
        quads.draw();
      }
    }

    void uniform_initial_condition() {
      for (size_t i = 0; i < current_field.size(); i++) {
        current_field[i] = target_particle_density;
      } 
      previous_field = current_field;
    }

    void add_particle_cluster(const Point& p, float num_particles) {
      size_t i;
      
      i = p.to_index(resolution_x, resolution_y);
      current_field[i] += num_particles/4.0;
      i = Point::to_index(p.x-1, p.y, resolution_x, resolution_y);
      current_field[i] += num_particles/4.0;
      i = Point::to_index(p.x, p.y-1, resolution_x, resolution_y);
      current_field[i] += num_particles/4.0;
      i = Point::to_index(p.x-1, p.y-1, resolution_x, resolution_y);
      current_field[i] += num_particles/4.0;
    }

    std::vector<Point> neighbors(const Point& p) const {
      std::vector<Point> points = {
        Point(Point::mod(p.x-1, n), p.y),
        Point(Point::mod(p.x+1, n), p.y),
        Point(p.x, Point::mod(p.y-1, m)),
        Point(p.x, Point::mod(p.y+1, m))
      };

      return points;
    }

    bool has_cluster_neighbors(const Point& p) const {
      std::vector<Point> ngbh = neighbors(p);

      for (auto const& p : ngbh) {
        if (clusters[p.to_index(n, m)]) {
          return true;
        }
      }

      return false;
    }

    virtual std::vector<float> get_cluster_texture() const override {
      return cluster_texture;
    }

    size_t get_binned_idx(size_t x, size_t y) const {
      return Point::to_index(x/bin_width, y/bin_width, resolution_x, resolution_y);
    }

    virtual std::vector<float> get_particle_texture() const override {
      std::vector<float> particle_texture(4*n*m, 0.0);
      for (size_t x = 0; x < n; x++) {
        for (size_t y = 0; y < m; y++) {
          size_t i = get_binned_idx(x, y);
          size_t j = Point::to_index(x, y, n, m);

          // TODO normalize intensity to [0, 1]
          float intensity = current_field[i];
          particle_texture[4*j]   = intensity*particle_color.r;
          particle_texture[4*j+1] = intensity*particle_color.g;
          particle_texture[4*j+2] = intensity*particle_color.b;
          particle_texture[4*j+3] = intensity*particle_color.w;
        }
      }

      return particle_texture;
    }

    float get_total_number_particles() const {
      float s = 0.0;
      for (size_t x = 0; x < resolution_x; x++) {
        for (size_t y = 0; y < resolution_y; y++) {
          size_t i = Point::to_index(x, y, resolution_x, resolution_y);
          s += current_field[i];
        }
      }
      return s;
    }

    // Subject to PBC for now
    virtual void diffusion_step() override {
      std::vector<float> next_field(resolution_x*resolution_y);

      float f1 = 1.0/(1.0 + 2.0*epsilon);
      float f2 = 1.0 - 2.0*epsilon;
      for (int x = 0; x < resolution_x; x++) {
        for (int y = 0; y < resolution_y; y++) {
          size_t i = Point::to_index(x, y, resolution_x, resolution_y);
          size_t i1, i2;

          i1 = Point::to_index(x-1, y, resolution_x, resolution_y); 
          i2 = Point::to_index(x+1, y, resolution_x, resolution_y); 
          float xterm = f1*epsilon*(current_field[i1] + current_field[i2]);

          i1 = Point::to_index(x, y-1, resolution_x, resolution_y); 
          i2 = Point::to_index(x, y+1, resolution_x, resolution_y); 
          float yterm = f1*epsilon*(current_field[i1] + current_field[i2]);

          float tterm = f1*f2*previous_field[i];

          next_field[i] = xterm + yterm + tterm;
        }
      }

      previous_field = current_field;
      current_field = next_field;

      // Compute gradients (if need be)
      if (draw_gradients) {
        float max_gradient_ = 0.0;
        for (int x = 0; x < resolution_x; x++) {
          for (int y = 0; y < resolution_y; y++) {
            size_t i  = Point::to_index(x, y, resolution_x, resolution_y);
            size_t i1 = Point::to_index(Point::mod(x+1, resolution_x), y, resolution_x, resolution_y);
            size_t i2 = Point::to_index(Point::mod(x-1, resolution_x), y, resolution_x, resolution_y);
            size_t i3 = Point::to_index(x, Point::mod(y-1, resolution_y), resolution_x, resolution_y);
            size_t i4 = Point::to_index(x, Point::mod(y+1, resolution_y), resolution_x, resolution_y);

            float gx = current_field[i1] - current_field[i2];
            float gy = current_field[i3] - current_field[i4];

            gradients[i] = {gx, gy};
            float gradient_magnitude = std::sqrt(gx*gx + gy*gy);
            if (gradient_magnitude > max_gradient_) {
              max_gradient_ = gradient_magnitude;
            }
          }
        }

        max_gradient = max_gradient_;
      }

      std::shuffle(all_points.begin(), all_points.end(), rng);

      // Field is updated; couple to cluster?

      for (auto const& p : all_points) {
        if (has_cluster_neighbors(p)) {
          size_t i = get_binned_idx(p.x, p.y);
          if (randf() < current_field[i]) {
            insert_cluster(p);
            remove_particle(p);
          }
        }
      }
    }

    virtual void insert_cluster(const Point& p) override {
      size_t i = p.to_index(n, m);
      if (clusters[i]) {
        return;
      }

      clusters[i] = 1;

      cluster_texture[4*i]   = cluster_color.r;
      cluster_texture[4*i+1] = cluster_color.g;
      cluster_texture[4*i+2] = cluster_color.b;
      cluster_texture[4*i+3] = cluster_color.w;
    }

    virtual void insert_particle(const Point& p) override {
    }

    void remove_particle(const Point& p) {
      size_t i = get_binned_idx(p.x, p.y);

      current_field[i] = std::max(current_field[i] - 1.0, 0.0);
    }

    virtual void set_cluster_color(Color color) override {
      cluster_color = color;
    }

    virtual void set_particle_color(Color color) override { 
      particle_color = color;
    }

    virtual void process_input(GLFWwindow* window) override {
      double time = glfwGetTime();
      double delay = 0.05;

      for (auto const& [key, func] : keymap) {
        if (glfwGetKey(window, key) == GLFW_PRESS && update_key_time(key, time) > delay) {
          func(*this, window);
        }
      }
    }

    void init_keymap() {
      keymap[GLFW_KEY_RIGHT] = [](DiffusivePhysicsModel& df, GLFWwindow* window) { df.insert_right = !df.insert_right; };
      keymap[GLFW_KEY_LEFT]  = [](DiffusivePhysicsModel& df, GLFWwindow* window) { df.insert_left = !df.insert_left; };
      keymap[GLFW_KEY_UP]    = [](DiffusivePhysicsModel& df, GLFWwindow* window) { df.insert_top = !df.insert_top; };
      keymap[GLFW_KEY_DOWN]  = [](DiffusivePhysicsModel& df, GLFWwindow* window) { df.insert_bottom = !df.insert_bottom; };
      keymap[GLFW_KEY_E]     = [](DiffusivePhysicsModel& df, GLFWwindow* window) { df.target_particle_density = std::min(1.0, df.target_particle_density + 0.01); };
      keymap[GLFW_KEY_D]     = [](DiffusivePhysicsModel& df, GLFWwindow* window) { df.target_particle_density = std::max(0.0, df.target_particle_density - 0.01); };
      keymap[GLFW_KEY_B]     = [](DiffusivePhysicsModel& df, GLFWwindow* window) { df.draw_gradients = !df.draw_gradients; }; 
      keymap[GLFW_KEY_O]     = [](DiffusivePhysicsModel& df, GLFWwindow* window) {
        int width, height;
        glfwGetWindowSize(window, &width, &height);

        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        xpos = (xpos/width - 0.5)*2.0;
        ypos = (-ypos/height + 0.5)*2.0;

        Point p = Point::get_point_from_pos(xpos, ypos, df.resolution_x, df.resolution_y);
        double num_particles = df.target_particle_density*df.resolution_x*df.resolution_y;
        df.add_particle_cluster(p, num_particles);
      };
    }

  private:
    KeyMap<DiffusivePhysicsModel> keymap;

    size_t n;
    size_t m;

    double target_particle_density;

    size_t bin_width;
    size_t resolution_x;
    size_t resolution_y;
    float epsilon;

    bool insert_left;
    bool insert_right;
    bool insert_top;
    bool insert_bottom;
    bool can_insert;


    bool draw_gradients;
    float max_gradient;
    std::vector<std::pair<float, float>> gradients;

    Shader shader;
    Color cluster_color;
    Color particle_color;

    std::vector<Point> all_points;
};

