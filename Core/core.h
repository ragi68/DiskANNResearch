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
#include "commonstructs.h"

void RobustPruning(Vertex& v, std::vector<Vertex*> U, float alpha, uint32_t R);
void RobustPruningOcclude(Vertex& v, std::vector<Vertex*> U, float alpha, uint32_t R);
Vertex* Medoid(std::vector<Vertex*> v);

float distance(const Vertex& u, const Vertex& v);
