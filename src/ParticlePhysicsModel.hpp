#pragma once

#include "DiffusionPhysicsModel.hpp"
#include "QuadCollection.hpp"

#define CLUSTER_POINT -1

struct ParticlePhysicsModelConfig {
  size_t n;
  size_t m;

  bool insert_left;
  bool insert_right;
  bool insert_top;
  bool insert_bottom;

  bool mobile_left;
  bool mobile_right;
  bool mobile_up;
  bool mobile_down;

  bool can_insert;

  double initial_density;

  size_t bbox_left;
  size_t bbox_right;
  size_t bbox_bottom;
  size_t bbox_top;
  int bbox_margin;

  bool show_bbox;

  bool pbc;

  std::string initial_state;

  int seed;

  ParticlePhysicsModelConfig()=default;

  static ParticlePhysicsModelConfig from_json(nlohmann::json& json) {
    ParticlePhysicsModelConfig config;

    config.n = static_cast<size_t>(json.value("n", 100));
    config.m = static_cast<size_t>(json.value("m", config.n));

    config.initial_density = static_cast<double>(json.value("initial_density", 0.0));

    config.insert_left = static_cast<bool>(json.value("insert_left", true));
    config.insert_right = static_cast<bool>(json.value("insert_right", true));
    config.insert_top = static_cast<bool>(json.value("insert_top", true));
    config.insert_bottom = static_cast<bool>(json.value("insert_bottom", true));

    config.mobile_left  = static_cast<bool>(json.value("mobile_left", true));
    config.mobile_right = static_cast<bool>(json.value("mobile_right", true));
    config.mobile_up    = static_cast<bool>(json.value("mobile_up", true));
    config.mobile_down  = static_cast<bool>(json.value("mobile_down", true));

    config.bbox_margin = static_cast<size_t>(json.value("bbox_margin", 10));
    config.show_bbox = static_cast<bool>(json.value("show_bbox", false));

    config.pbc = static_cast<bool>(json.value("pbc", false));

    config.initial_state = static_cast<std::string>(json.value("initial_state", "central_cluster"));

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

Color BBOX_COLOR_ACTIVE = {1.0, 0.5, 0.0, 1.0};
Color BBOX_COLOR_INACTIVE = {0.5, 0.5, 0.0, 1.0};

class ParticlePhysicsModel : public DiffusionPhysicsModel {
  public:
    std::vector<int> field;
    std::vector<float> cluster_texture;

    ParticlePhysicsModel(const ParticlePhysicsModelConfig& config) : DiffusionPhysicsModel(config.seed) {
      n = config.n;
      m = config.m;

      insert_left   = config.insert_left;
      insert_right  = config.insert_right;
      insert_top    = config.insert_top;
      insert_bottom = config.insert_bottom;

      mobile_left  = config.mobile_left;
      mobile_right = config.mobile_right;
      mobile_up    = config.mobile_up;
      mobile_down  = config.mobile_down;

      bbox_left   = config.bbox_left;
      bbox_right  = config.bbox_right;
      bbox_top    = config.bbox_top;
      bbox_bottom = config.bbox_bottom;
      bbox_margin = config.bbox_margin;

      show_bbox = config.show_bbox;

      target_particle_density = config.initial_density;

      pbc = config.pbc;

      field = std::vector<int>(n*m);
      cluster_texture = std::vector<float>(4*n*m, 0.0);

      std::string initial_state = config.initial_state;
      if (initial_state == "central_cluster") {
        initialize_central_cluster();
      } else if (initial_state == "boundary_cluster") {
        initialize_boundary_cluster();
      }

      shader = Shader("vertex_shader.vs", "fragment_shader.fs");

      can_insert = true;
      fix_bbox();
      init_keymap();
    }

    virtual std::vector<float> get_cluster_texture() const override {
      return cluster_texture;
    }

    virtual std::vector<float> get_particle_texture() const override {
      std::vector<float> particle_texture(4*n*m, 0.0);
      for (size_t x = 0; x < n; x++) {
        for (size_t y = 0; y < m; y++) {
          Point p(x, y);
          if (get(p) > 0) {
            size_t i = p.to_index(n, m);
            particle_texture[4*i] =   particle_color.r;
            particle_texture[4*i+1] = particle_color.g;
            particle_texture[4*i+2] = particle_color.b;
            particle_texture[4*i+3] = particle_color.w;
          }
        }
      }

      return particle_texture;
    }

    virtual void diffusion_step() override {
      double particle_density = 0.0;
      double cluster_size;
      for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
          Point site(i, j);
          if (get(site) == CLUSTER_POINT) {
            cluster_size += 1.0;
          } else {
            int num_particles = get(site);
            particle_density += num_particles;
            for (size_t k = 0; k < num_particles; k++) {
              std::vector<Point> targets = neighbors(site, true);
              size_t target_idx = rng() % targets.size();
              Point target = targets[target_idx];

              // Remove particle from site and either freeze it into the cluster or move it to a neighboring site
              get(site)--;
              bool freeze_target = has_cluster_neighbor(target);
              if (freeze_target) {
                insert_cluster(target);
              } else {
                get(target)++;
              }
            }
          }
        }
      }

      size_t bbox_width = bbox_right - bbox_left;
      size_t bbox_height = bbox_top - bbox_bottom;
      double available_volume = bbox_width*bbox_height - cluster_size;
      particle_density = particle_density / available_volume;

      int num_new_particles = (target_particle_density - particle_density)*available_volume;
      for (int i = 0; i < num_new_particles; i++) {
        insert_particle();
      }
    }

    virtual void insert_cluster(const Point& p) override {
      if (get(p) == CLUSTER_POINT) {
        return;
      }

      get(p) = CLUSTER_POINT;
      update_bbox(p);

      size_t i = p.to_index(n, m);
      cluster_texture[4*i] =   cluster_color.r;
      cluster_texture[4*i+1] = cluster_color.g;
      cluster_texture[4*i+2] = cluster_color.b;
      cluster_texture[4*i+3] = cluster_color.w;
    }

    void insert_particle() {
      if (!can_insert) {
        return;
      }

      // Need to add new particle
      std::vector<size_t> directions;
      if (insert_left) {
        directions.push_back(0);
      } if (insert_right) {
        directions.push_back(1);
      } if (insert_top) {
        directions.push_back(2);
      } if (insert_bottom) {
        directions.push_back(3);
      }

      // If insertions are not allowed at the boundary, randomly choose a point
      Point site;
      size_t bbox_width = bbox_right - bbox_left;
      size_t bbox_height = bbox_top - bbox_bottom;
      if (directions.size() == 0) {
        site = Point(rng() % bbox_width + bbox_left, rng() % bbox_height + bbox_bottom);

        size_t niter = 100;
        size_t k = 0;
        while (has_cluster_neighbor(site) && k < niter) {
          site = Point(rng() % bbox_width + bbox_left, rng() % bbox_height + bbox_bottom);
          k++;
        } 

        if (k == niter) {
          can_insert = false;
          return;
        }
        
      } else {
        size_t direction = directions[rng() % directions.size()];
        if (direction == 0) { // left
          site = Point(bbox_left, rng() % bbox_height + bbox_bottom);
        } else if (direction == 1) { // right
          site = Point(bbox_right, rng() % bbox_height + bbox_bottom);
        } else if (direction == 2) { // bottom
          site = Point(rng() % bbox_width + bbox_left, bbox_top);
        } else if (direction == 3) { // top
          site = Point(rng() % bbox_width + bbox_left, bbox_bottom);
        }
      } 

      insert_particle(site);
    }

    virtual void insert_particle(const Point& p) override {
      if (get(p) == CLUSTER_POINT) {
        return;
      }

      get(p)++;
    }

    virtual void set_cluster_color(Color color) override {
      cluster_color = color;
    }

    virtual void set_particle_color(Color color) override { 
      particle_color = color;
    }

    virtual void process_input(GLFWwindow* window) override {
      double time = glfwGetTime();
      double delay = 0.02;

      for (auto const& [key, func] : keymap) {
        if (glfwGetKey(window, key) == GLFW_PRESS && update_key_time(key, time) > delay) {
          func(*this, window);
        }
      }
    }

    void init_keymap() {
      keymap[GLFW_KEY_RIGHT] = [](ParticlePhysicsModel& df, GLFWwindow* window) { df.insert_right = !df.insert_right; };
      keymap[GLFW_KEY_LEFT]  = [](ParticlePhysicsModel& df, GLFWwindow* window) { df.insert_left = !df.insert_left; };
      keymap[GLFW_KEY_UP]    = [](ParticlePhysicsModel& df, GLFWwindow* window) { df.insert_top = !df.insert_top; };
      keymap[GLFW_KEY_DOWN]  = [](ParticlePhysicsModel& df, GLFWwindow* window) { df.insert_bottom = !df.insert_bottom; };
      keymap[GLFW_KEY_B]     = [](ParticlePhysicsModel& df, GLFWwindow* window) { df.show_bbox = !df.show_bbox; };
      keymap[GLFW_KEY_E]     = [](ParticlePhysicsModel& df, GLFWwindow* window) { df.target_particle_density = std::min(1.0, df.target_particle_density + 0.01); };
      keymap[GLFW_KEY_D]     = [](ParticlePhysicsModel& df, GLFWwindow* window) { df.target_particle_density = std::max(0.0, df.target_particle_density - 0.01); };
    }

  private:
    KeyMap<ParticlePhysicsModel> keymap;

    size_t n;
    size_t m;

    double target_particle_density;

    bool insert_left;
    bool insert_right;
    bool insert_top;
    bool insert_bottom;
    bool can_insert;

    bool mobile_left;
    bool mobile_right;
    bool mobile_down;
    bool mobile_up;

    size_t bbox_left;
    size_t bbox_right;
    size_t bbox_bottom;
    size_t bbox_top;
    size_t bbox_margin;

    bool show_bbox;

    bool pbc;

    Shader shader;
    Color cluster_color;
    Color particle_color;

    void fix_bbox() {
      bbox_left = n-1;
      bbox_right = 0;
      bbox_top = 0;
      bbox_bottom = m-1;

      bool found_cluster = false;
      for (size_t x = 0; x < n; x++) {
        for (size_t y = 0; y < m; y++) {
          Point p(x, y);
          if (get(p) == CLUSTER_POINT) {
            update_bbox(p);
            found_cluster = true;
          }
        }
      }

      if (!found_cluster) {
        int center = n/2;
        bbox_left = (n - bbox_margin)/2;
        bbox_right = (n + bbox_margin)/2;
        bbox_bottom = (m - bbox_margin)/2;
        bbox_top = (m + bbox_margin)/2;
      }
    }

    void update_bbox(const Point& p) {
      int x_left =   std::max(int(p.x) - int(bbox_margin), int(0));
      int x_right =  std::min(int(p.x) + int(bbox_margin), int(n-1));
      int y_bottom = std::max(int(p.y) - int(bbox_margin), int(0));
      int y_top =    std::min(int(p.y) + int(bbox_margin), int(m-1));

      if (x_right > bbox_right) {
        bbox_right = x_right;
      }

      if (x_left < bbox_left) {
        bbox_left = x_left;
      }

      if (y_top > bbox_top) {
        bbox_top = y_top;
      }

      if (y_bottom < bbox_bottom) {
        bbox_bottom = y_bottom;
      }
    }

    void initialize_central_cluster() {
      Point center(n/2, m/2);
      insert_cluster(center);
    }

    void initialize_boundary_cluster() {
      for (size_t i = 0; i < n; i++) {
        Point p(i, 0);
        insert_cluster(p);
      }

      insert_bottom = false;
      insert_left = false;
      insert_right = false;
      insert_top = true;
    }

    int& operator[](const Point& p) {
      return field[p.to_index(n, m)];
    }

    int operator[](const Point& p) const {
      return field[p.to_index(n, m)];
    }

    int& get(const Point& p) {
      return operator[](p);
    }

    int get(const Point& p) const {
      return operator[](p);
    }

    bool within_bbox(const Point& p) const {
      return p.x >= bbox_left && p.x <= bbox_right & p.y >= bbox_bottom & p.y <= bbox_top;
    }

    std::vector<Point> neighbors(const Point& p, bool bounded=false) const {
      size_t left = bounded ? bbox_left : 0;
      size_t right = bounded ? bbox_right : n-1;
      size_t bottom = bounded ? bbox_bottom : 0;
      size_t top = bounded ? bbox_top : m-1;

      std::vector<Point> points;
      if (pbc) {
        if (mobile_left) {
          points.push_back(Point((p.x == left) ? right-1 : p.x-1, p.y));
        }

        if (mobile_right) {
          points.push_back(Point((p.x == right) ? left+1 : p.x+1, p.y));
        }

        if (mobile_down) {
          points.push_back(Point(p.x, (p.y == bottom) ? top-1 : p.y-1));
        }

        if (mobile_up) {
          points.push_back(Point(p.x, (p.y == top) ? bottom+1 : p.y+1));
        }
      } else {
        if (p.x != left && mobile_left) {
          points.push_back(Point(p.x-1, p.y));
        }

        if (p.x != right && mobile_right) {
          points.push_back(Point(p.x+1, p.y));
        }

        if (p.y != bottom && mobile_down) {
          points.push_back(Point(p.x, p.y-1));
        }

        if (p.y != top && mobile_up) {
          points.push_back(Point(p.x, p.y+1));
        }
      }

      return points;
    }

    bool has_cluster_neighbor(const Point &p) {
      if (get(p) == CLUSTER_POINT) {
        return true;
      }

      std::vector<Point> target_neighbors = neighbors(p);
      for (auto const &t : target_neighbors) {
        if (get(t) == CLUSTER_POINT) {
          return true;
        }
      }

      return false;
    }

    std::pair<std::vector<float>, std::vector<unsigned int>> get_bbox_vertices(bool active) const {
      std::vector<float> vertices;
      std::vector<unsigned int> indices;

      std::vector<float> new_vertices;
      std::vector<unsigned int> new_indices;

      float thickness = 0.01;

      float left = ((float) bbox_left/n - 0.5)*2.0;
      float right = ((float) bbox_right/n - 0.5)*2.0;
      float top = ((float) bbox_top/m - 0.5)*2.0;
      float bottom = ((float) bbox_bottom/m - 0.5)*2.0;

      float width = right - left;
      float height = top - bottom;

      // Bottom
      if (insert_bottom == active) {
        std::tie(new_vertices, new_indices) = make_quad(left, bottom, width, thickness, vertices.size() / 3);
        vertices.insert(vertices.end(), new_vertices.begin(), new_vertices.end());
        indices.insert(indices.end(), new_indices.begin(), new_indices.end());
      }

      // Left
      if (insert_left == active) {
        std::tie(new_vertices, new_indices) = make_quad(left, bottom, thickness, height, vertices.size() / 3);
        vertices.insert(vertices.end(), new_vertices.begin(), new_vertices.end());
        indices.insert(indices.end(), new_indices.begin(), new_indices.end());
      }

      // Top
      if (insert_top == active) {
        std::tie(new_vertices, new_indices) = make_quad(left, top-thickness, width, thickness, vertices.size() / 3);
        vertices.insert(vertices.end(), new_vertices.begin(), new_vertices.end());
        indices.insert(indices.end(), new_indices.begin(), new_indices.end());
      }

      // Right
      if (insert_right == active) {
        std::tie(new_vertices, new_indices) = make_quad(right-thickness, bottom, thickness, height, vertices.size() / 3);
        vertices.insert(vertices.end(), new_vertices.begin(), new_vertices.end());
        indices.insert(indices.end(), new_indices.begin(), new_indices.end());
      }

      return {vertices, indices};
    }

    virtual void draw() override {
      if (show_bbox) {
        QuadCollection quads(&shader);

        auto [bbox_vertices_inactive, bbox_indices_inactive] = get_bbox_vertices(false);
        quads.add_vertices(bbox_vertices_inactive, bbox_indices_inactive, BBOX_COLOR_INACTIVE);
        auto [bbox_vertices_active, bbox_indices_active] = get_bbox_vertices(true);
        quads.add_vertices(bbox_vertices_active, bbox_indices_active, BBOX_COLOR_ACTIVE);

        quads.draw();
      }
    }
};
