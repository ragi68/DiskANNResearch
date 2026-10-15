#pragma once
#include <stdint.h>
#include <vector>

struct Vertex {
  uint32_t id;
  std::vector<float> data;
  std::vector<Vertex*> out;
  std::vector<Vertex*> in;
};
