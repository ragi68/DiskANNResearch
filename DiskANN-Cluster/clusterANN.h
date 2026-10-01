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



// Namespaced because clusterANN's Vertex/distance would otherwise collide
// with DiskANN-Classic/DiskANN.h's identically-named globals whenever both
// headers land in the same translation unit (e.g. Testing/main.cpp).
namespace ClusterANN {

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
  std::vector<std::vector<uint32_t>> medoidGraph;  // coarse layer: medoidGraph[j] = neighbor cluster indices of medoid j
  uint32_t medoidEntry = 0;         // coarse search start: index of the medoid nearest the medoids' mean
};

std::vector<Cluster> KClustering(std::vector<Vertex*> v, uint32_t c, uint32_t I);
void RobustPruning(Vertex& v, std::vector<Vertex*> U, float alpha, uint32_t R);

ClusterGraph CreateClusterGraph(std::vector<Vertex*> v, float alpha, uint32_t pruneR, uint32_t totalR,
    uint32_t edgesPerPair, uint32_t poolSize,
    uint32_t numAdjacentClusters = 1,
    bool rankedEdges = false,   // true: AddBoundaryEdgesRanked
    uint32_t clustersPerVertex = 7, uint32_t edgesPerCluster = 3,  // ranked knobs
    uint32_t medoidR = 24
);

// Coarse search picks the top p medoids to seed the fine beam; p = 1 starts from the single nearest medoid.
std::vector<Vertex*> clusterGreedySearch(const ClusterGraph& g, Vertex& q, uint32_t L, uint32_t p = 1, uint32_t coarseL = 16);

void AddBoundaryEdges(ClusterGraph& g, uint32_t edgesPerPair, uint32_t poolSize, uint32_t R,
   uint32_t numAdjacentClusters = 1, uint32_t seed = 12345);


void AddBoundaryEdgesRanked(ClusterGraph& g, uint32_t R, uint32_t candPerCluster = 50,
    uint32_t clustersPerVertex = 7, uint32_t edgesPerCluster = 3,
    uint32_t seed = 12345
);


// Coarse layer: slow preprocessing over the medoids only, stored in g.medoidGraph (kept out of Vertex::out).
void topLevelEdges(ClusterGraph& g, float alpha, uint32_t medoidR);


// Greedy beam search (width coarseL) over g.medoidGraph; returns the p medoids nearest q, nearest first.
std::vector<Vertex*> coarseGreedySearch(const ClusterGraph& g, const Vertex& q, uint32_t p = 1,
    uint32_t coarseL = 16);


Vertex* Medoid(std::vector<Vertex*> v);
float distance(const Vertex& u, const Vertex& v);

}  // namespace ClusterANN
