#pragma once
#include "DiskANN.h"
#include <random>
#include <stdio.h>
#include <iostream>
#include <unordered_set>


class TestANN{
public:

    std::vector<Vertex*> BruteForceKNN(const std::vector<Vertex*>& vertices, Vertex q, uint32_t k);

    std::vector<Vertex*> RandomVertices(uint32_t n, uint32_t dim, std::mt19937& rng);

    // fraction of `truth` (the brute-force top-k) present in `result`
    float RecallAtK(const std::vector<Vertex*>& result, const std::vector<Vertex*>& truth);
};