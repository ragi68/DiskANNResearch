#pragma once
#include "DiskANN.h"
#include <random>
#include <stdio.h>
#include <iostream>


class TestANN{
public:

    std::vector<Vertex*> BruteForceKNN(const std::vector<Vertex*>& vertices, Vertex q, uint32_t k);

    std::vector<Vertex*> RandomVertices(uint32_t n, uint32_t dim, std::mt19937& rng);
};