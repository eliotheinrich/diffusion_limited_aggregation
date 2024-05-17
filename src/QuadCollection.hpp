#pragma once

#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <nlohmann/json.hpp>

#include "Shader.hpp"

struct Color {
  float r;
  float g;
  float b;
  float w;

  std::string to_string() const {
    return std::to_string(r) + ", " + std::to_string(g) + ", " + std::to_string(b) + ", " + std::to_string(w); 
  }
};

Color parse_color(const nlohmann::json& json, std::optional<Color> default_color = std::nullopt) {
  if (json.is_null()) {
    if (default_color.has_value()) {
      return default_color.value();
    } else {
      throw std::invalid_argument("json does not contain color key and no default was provided.");
    }
  }

  if (!(json.is_array()) || !(json.size() == 3 || json.size() == 4)) {
    throw std::invalid_argument("Invalid value for color.");
  }

  if (json.size() == 3) {
    return Color{json[0], json[1], json[2], 1.0};
  } else if (json.size() == 4) {
    return Color{json[0], json[1], json[2], json[3]};
  }

  // Unreachable
  return Color{0.0, 0.0, 0.0, 0.0};
}

std::pair<std::vector<float>, std::vector<unsigned int>> make_quad(float x, float y, float width, float height, unsigned int v) {
  std::vector<float> vertices = {x,         y,          0.0,
                                 x + width, y,          0.0,
                                 x,         y + height, 0.0,
                                 x + width, y + height, 0.0};
  std::vector<unsigned int> indices = {v, v + 1, v + 3, v, v + 2, v + 3};
  return {vertices, indices};
}

void displace_x(std::vector<float>& vertices, float dx) {
  vertices[0] += dx;
  vertices[3] += dx;
  vertices[6] += dx;
  vertices[9] += dx;
}

void displace_y(std::vector<float>& vertices, float dy) {
  vertices[1]  += dy;
  vertices[4]  += dy;
  vertices[7]  += dy;
  vertices[10] += dy;
}

void displace(std::vector<float>& vertices, float dx, float dy) {
  displace_x(vertices, dx);
  displace_y(vertices, dy);
}

void rotate(std::vector<float>& vertices, float angle) {
  float center_x = (vertices[0] + vertices[3])/2.0;
  float center_y = (vertices[1] + vertices[7])/2.0;

  displace(vertices, -center_x, -center_y);

  float a1 = std::cos(angle);
  float a2 = -std::sin(angle);
  float a3 = -a2;
  float a4 = a1;

  for (size_t i = 0; i < 4; i++) {
    float x = vertices[3*i];
    float y = vertices[3*i+1];
    vertices[3*i]   = x*std::cos(angle) - y*std::sin(angle);
    vertices[3*i+1] = x*std::sin(angle) + y*std::cos(angle);
  }

  displace(vertices, center_x, center_y);
}

class QuadCollection {
  public:
    QuadCollection(Shader* shader) : shader(shader), num_categories(0) {
      // Initialize buffers
      glGenVertexArrays(1, &VAO);
      glGenBuffers(1, &VBO);
      glGenBuffers(1, &EBO);

      // Configure vertex attributes
      glBindVertexArray(VAO);
      glBindBuffer(GL_ARRAY_BUFFER, VBO);

      glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
      glEnableVertexAttribArray(0);

      // Unbind buffers
      glBindVertexArray(0);
      glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    ~QuadCollection() {
      glDeleteVertexArrays(1, &VAO);
      glDeleteBuffers(1, &VBO);
      glDeleteBuffers(1, &EBO);
    }

    void draw() {
      // Binding buffers
      glBindVertexArray(VAO);
      glBindBuffer(GL_ARRAY_BUFFER, VBO);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);

      // Load vertex data into VBO/EBO
      glBufferData(GL_ARRAY_BUFFER, vertices.size()*sizeof(float), vertices.data(), GL_STREAM_DRAW);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size()*sizeof(unsigned int), indices.data(), GL_STREAM_DRAW);
      
      shader->use();

      // Draw each category of vertex
      size_t n = 0;
      for (size_t i = 0; i < num_categories; i++) {
        shader->set_float4("color", colors[i].r, colors[i].g, colors[i].b, colors[i].w);

        glDrawElements(GL_TRIANGLES, index_counts[i], GL_UNSIGNED_INT, (void*)(n * sizeof(unsigned int)));
        n += index_counts[i];
      }

      // Unbind buffers
      glBindVertexArray(0);
      glBindBuffer(GL_ARRAY_BUFFER, 0);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }

    void add_vertices(const std::vector<float>& new_vertices, const std::vector<unsigned int>& new_indices, Color color) {
      size_t num_old_vertices = vertices.size() / 3;
      // push vertices
      vertices.insert(vertices.end(), new_vertices.begin(), new_vertices.end());
      // push shifted indices
      for (size_t i = 0; i < new_indices.size(); i++) {
        indices.push_back(new_indices[i] + num_old_vertices);
      }

      index_counts.push_back(new_indices.size());
      colors.push_back(color);
      num_categories++;
    }

  private:
    Shader* shader;

    unsigned int VAO;
    unsigned int VBO;
    unsigned int EBO;

    size_t num_categories;
    std::vector<float> vertices;
    std::vector<unsigned int> indices;
    std::vector<size_t> index_counts;
    std::vector<Color> colors;
};
