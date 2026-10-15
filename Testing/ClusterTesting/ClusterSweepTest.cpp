#include "../../DiskANN-Cluster/clusterANN.h"
#include "../../Core/core_testing.h"
#include <random>
#include <chrono>
#include <stdio.h>
#include <iostream>

// ===========================================================================
//  Boundary-proportion sweep for the backbone cluster graph.
//
//  Same data, queries and graph parameters as TestClusterInstance's backbone
//  run, but builds one backbone graph per entry in `proportions` and reports
//  build time, recall and query time for each. Lives in its own namespace so
//  it doesn't collide with ClusterTest::run() inside main.cpp.
//
//  Cost: the backbone prunes every boundary node against all boundary nodes of
//  other clusters, so its build time grows roughly with proportion^2
//  (~11 min at 0.2, n = 20000, 128-D).
// ===========================================================================

namespace ClusterSweep {

int run() {
    TestANN test;
    std::mt19937 rng(42); // same seed as TestClusterInstance -> same data and queries
    uint32_t numVertices = 20000;
    uint32_t dim = 128;
    uint32_t k = 10;
    uint32_t L = 125;

    uint32_t numQueries = 100;

    const double proportions[] = {0.05, 0.1, 0.2, 0.3, 0.4};

    std::vector<Vertex*> vertices = test.RandomVertices(numVertices, dim, rng);

    // Queries get ids outside the range of generated vertices.
    std::vector<Vertex> queries(numQueries);
    for (uint32_t i = 0; i < numQueries; i++) {
        queries[i].id = numVertices + i;
        for (uint32_t j = 0; j < dim; j++)
            queries[i].data.push_back(std::uniform_real_distribution<float>(0.0f, 1000.0f)(rng));
    }

    // Brute-force ground truth, computed once and shared by every proportion.
    std::vector<std::vector<Vertex*>> knnResults(numQueries);
    for (uint32_t i = 0; i < numQueries; i++)
        knnResults[i] = test.BruteForceKNN(vertices, queries[i], k);

    // Unused by the backbone build, but passed to keep CreateClusterGraph's
    // arguments identical to TestClusterInstance.
    uint32_t numClusters = std::max<uint32_t>(1, static_cast<uint32_t>(
        std::lround(std::sqrt(static_cast<double>(numVertices)))));
    uint32_t numAdjacentClusters = std::max<uint32_t>(1, static_cast<uint32_t>(
        std::lround(0.10 * numClusters)));

    using Clock = std::chrono::steady_clock;
    bool ok = true;

    printf("Backbone boundaryProportion sweep (n=%u, dim=%u, %u queries, L=%u)\n",
           numVertices, dim, numQueries, L);
    printf("  proportion |   build (s) | mean Recall@%u |  min | avg query (ms)\n", k);
    fflush(stdout);  // flush per line so progress shows when output is redirected

    for (double proportion : proportions) {
        auto t0 = Clock::now();
        ClusterANN::ClusterGraph g = ClusterANN::CreateClusterGraph(
            vertices, /*alpha=*/1.2f, /*pruneR=*/70,
            /*totalR=*/125, /*edgesPerPair=*/25, /*poolSize=*/40,
            numAdjacentClusters, ClusterANN::EdgeMode::Backbone,
            /*clustersPerVertex=*/numAdjacentClusters, /*edgesPerCluster=*/3,
            /*medoidR=*/24, proportion);
        double buildS = std::chrono::duration<double>(Clock::now() - t0).count();

        if (g.clusters.empty() || g.medoids.size() != g.clusters.size()) {
            printf("  %10.2f | cluster graph was not built correctly\n", proportion);
            fflush(stdout);
            ok = false;
            continue;
        }

        double sumRecall = 0.0, sumQueryMs = 0.0;
        float minRecall = 1.0f;
        for (uint32_t i = 0; i < numQueries; i++) {
            auto q0 = Clock::now();
            std::vector<Vertex*> results = ClusterANN::clusterGreedySearch(g, queries[i], L, /*p=*/1);
            sumQueryMs += std::chrono::duration<double, std::milli>(Clock::now() - q0).count();

            float recall = test.RecallAtK(results, knnResults[i]);
            sumRecall += recall;
            minRecall = std::min(minRecall, recall);
        }

        printf("  %10.2f | %11.1f | %14.3f | %4.1f | %14.2f\n",
               proportion, buildS, sumRecall / numQueries, minRecall, sumQueryMs / numQueries);
        fflush(stdout);
    }

    for (auto& v : vertices) delete v;

    if (ok) std::cout << "testClusterSweep PASSED\n";
    else std::cout << "testClusterSweep FAILED\n";

    return ok ? 0 : 1;
}

}  // namespace ClusterSweep

int testClusterSweep() {
    return ClusterSweep::run();
}
