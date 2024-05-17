#pragma once

#include "QuadCollection.hpp"

template<class T>
using KeyMap = std::map<decltype(GLFW_KEY_ESCAPE), std::function<void(T&, GLFWwindow*)>>;

struct Point {
  int x;
  int y;

  Point()=default;
  Point(int x, int y) : x(x), y(y) {}

  inline size_t to_index(size_t n, size_t m) const {
    return x + n*y;
  }

  inline static size_t to_index(size_t x, size_t y, size_t n, size_t m) {
    return x + n*y;
  }

  inline static size_t mod(int i, int n) {
    return (i % n + n) % n;
  }

  inline static Point get_point_from_pos(double x, double y, size_t n, size_t m) {
    x = (x + 1.0)/2.0;
    y = (y + 1.0)/2.0;

    return Point(int(x*n), int(y*m));
  }
};

enum PhysicsModelType { Particle, Diffusion };

enum InitialState { Central, Boundary };

// TODO template polymorphism??
class DiffusionPhysicsModel {
  public:
    DiffusionPhysicsModel(int seed) {
      rng.seed(seed);
    }

    virtual std::vector<float> get_cluster_texture() const=0;
    virtual std::vector<float> get_particle_texture() const=0;

    virtual void diffusion_step()=0;
    virtual void insert_cluster(const Point& p)=0;
    virtual void insert_particle(const Point& p)=0;

    virtual void set_cluster_color(Color color)=0;
    virtual void set_particle_color(Color color)=0;

    virtual void process_input(GLFWwindow *window) {}
    virtual void draw() {}

  protected:
    std::map<decltype(GLFW_KEY_ESCAPE), double> prevtime;

    double update_key_time(int key_code, double time) {
      if (!prevtime.contains(key_code)) {
        prevtime[key_code] = 0.0;
      }

      double t = time - prevtime[key_code];
      prevtime[key_code] = time;
      return t;
    }

    std::minstd_rand rng;

    double randf() {
      return double(rng())/RAND_MAX;
    }
};
