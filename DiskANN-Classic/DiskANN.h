#pragma once
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <algorithm>
#include <math.h>
#include <random>
#include <numeric>

struct Vertex {
  uint32_t id;
  std::vector<float> data;
  std::vector<Vertex*> out;
  std::vector<Vertex*> in;
};


std::vector<Vertex*> GreedySearch(Vertex& s, Vertex& q, uint32_t L);
void RobustPruning(Vertex& v, std::vector<Vertex*> U, float alpha, uint32_t R);
void RobustPruningOcclude(Vertex& v, std::vector<Vertex*> U, float alpha, uint32_t R);
void GraphCreation(std::vector<Vertex*>& vertices, float alpha, uint32_t R);
void FastGraphCreation(std::vector<Vertex*>& v, float alpha, uint32_t L, uint32_t R, std::mt19937& gen);
Vertex* Medoid(std::vector<Vertex*> v);
std::vector<Vertex*> initRegGraph();

float distance(const Vertex& u, const Vertex& v);

