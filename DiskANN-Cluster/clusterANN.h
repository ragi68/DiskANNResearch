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
#include <memory>
#include <limits>
#include <set>
#include <utility>
#include <unordered_map>
#include <unordered_set>

struct Vertex {
  uint32_t id;
  std::vector<float> data;
  std::vector<Vertex*> out;
};

struct Cluster{
  uint32_t clusterID;
  std::vector<Vertex*> clusterPoints;
};

struct ClusterGraph {
  std::unique_ptr<Vertex> hub;      // universal node; heap-allocated so its address stays stable
  std::vector<Cluster> clusters;
  std::vector<Vertex*> medoids;     // medoids[j] is the medoid of clusters[j]
};

std::vector<Cluster> KClustering(std::vector<Vertex*> v, uint32_t c, uint32_t I);
void RobustPruning(Vertex& v, std::vector<Vertex*> U, float alpha, uint32_t R);
ClusterGraph CreateClusterGraph(std::vector<Vertex*> v, float alpha,
                                uint32_t pruneR, uint32_t totalR,
                                uint32_t edgesPerPair, uint32_t poolSize);
std::vector<Vertex*> clusterGreedySearch(Vertex& s, Vertex& q, uint32_t L);
void AddBoundaryEdges(ClusterGraph& g, uint32_t edgesPerPair, uint32_t poolSize, uint32_t R, uint32_t seed = 12345);
Vertex* Medoid(std::vector<Vertex*> v);
float distance(const Vertex& u, const Vertex& v);
