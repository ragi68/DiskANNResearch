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
#include <unordered_set>
#include "../Core/core.h"  // Vertex, distance, RobustPruning, RobustPruningOcclude, Medoid


std::vector<Vertex*> GreedySearch(Vertex& s, Vertex& q, uint32_t L);
void GraphCreation(std::vector<Vertex*>& vertices, float alpha, uint32_t R);
void FastGraphCreation(std::vector<Vertex*>& v, float alpha, uint32_t L, uint32_t R, std::mt19937& gen);
std::vector<Vertex*> initRegGraph();

