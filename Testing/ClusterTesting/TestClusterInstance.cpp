#include "../../DiskANN-Cluster/clusterANN.h"
#include "../../Core/core_testing.h"
#include <random>
#include <chrono>
#include <stdio.h>
#include <iostream>

// ===========================================================================
//  Smoke test for clusterANN (DiskANN-Cluster/clusterANN.cpp).
//
//  Vertex and the brute-force/recall helpers come from Core/, shared with the
//  DiskANN.h-based tests. run() lives in its own namespace so it doesn't
//  collide with anything else included into main.cpp.
// ===========================================================================

namespace ClusterTest {

int run() {
    TestANN test;
    std::mt19937 rng(42); // fixed seed for reproducibility
    uint32_t numVertices = 20000;
    uint32_t dim = 128;
    uint32_t k = 10;

    uint32_t numQueries = 100;

    std::vector<Vertex*> vertices = test.RandomVertices(numVertices, dim, rng);

    // Queries get ids outside the range of generated vertices.
    std::vector<Vertex> queries(numQueries);
    for (uint32_t i = 0; i < numQueries; i++) {
        queries[i].id = numVertices + i;
        for (uint32_t j = 0; j < dim; j++)
            queries[i].data.push_back(std::uniform_real_distribution<float>(0.0f, 1000.0f)(rng));
    }

    // Brute-force ground truth, computed once and shared by both constructions.
    std::vector<std::vector<Vertex*>> knnResults(numQueries);
    for (uint32_t i = 0; i < numQueries; i++)
        knnResults[i] = test.BruteForceKNN(vertices, queries[i], k);

    // number of clusters C = round(sqrt(numVertices)) (see CreateClusterGraph);
    // ~10% of that many nearest foreign clusters get boundary-edge candidates
    // from each point, instead of a small fixed count.
    uint32_t numClusters = std::max<uint32_t>(1, static_cast<uint32_t>(
        std::lround(std::sqrt(static_cast<double>(numVertices)))));
    uint32_t numAdjacentClusters = std::max<uint32_t>(1, static_cast<uint32_t>(
        std::lround(0.10 * numClusters)));

    // Three constructions on the same vertices and queries: old random pool edges,
    // rank-weighted edges, then the boundary-node backbone (slow preprocessing over
    // the top boundaryProportion of each cluster). CreateClusterGraph clears every out list first,
    // so each build starts clean; each graph is searched before the next build
    // overwrites the shared vertices' edges. Reports numbers only, no recall
    // threshold: the test fails only if a graph is structurally broken.
    //
    // Ranked-edge knobs: each vertex sends edgesPerCluster edges into each of its
    // clustersPerVertex nearest foreign clusters (10% of all clusters, same as
    // numAdjacentClusters: 14 at n=20000).
    const uint32_t clustersPerVertex = numAdjacentClusters, edgesPerCluster = 3;
    const double boundaryProportion = 0.2;

    using Clock = std::chrono::steady_clock;
    using ClusterANN::EdgeMode;
    bool ok = true;
    for (EdgeMode mode : {EdgeMode::Random, EdgeMode::Ranked, EdgeMode::Backbone}) {
        const char* label = mode == EdgeMode::Ranked   ? "ranked"
                          : mode == EdgeMode::Backbone ? "backbone"
                          :                              "old";
        std::cout << "Cluster Graph Creation [" << label << " boundary edges]";
        if (mode == EdgeMode::Ranked)
            std::cout << " (clustersPerVertex=" << clustersPerVertex
                      << ", edgesPerCluster=" << edgesPerCluster << ")";
        if (mode == EdgeMode::Backbone)
            std::cout << " (boundaryProportion=" << boundaryProportion << ")";
        std::cout << "\n";

        auto t0 = Clock::now();
        ClusterANN::ClusterGraph g = ClusterANN::CreateClusterGraph(
            vertices, /*alpha=*/1.2f, /*pruneR=*/70,
            /*totalR=*/125, /*edgesPerPair=*/25, /*poolSize=*/40,
            numAdjacentClusters, mode, clustersPerVertex, edgesPerCluster,
            /*medoidR=*/24, boundaryProportion);
        double buildMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

        if (g.clusters.empty() || g.medoids.size() != g.clusters.size()) {
            std::cerr << "cluster graph [" << label << "] was not built correctly\n";
            ok = false;
            continue;
        }

        double sumRecall = 0.0, sumQueryMs = 0.0;
        float minRecall = 1.0f;
        for (uint32_t i = 0; i < numQueries; i++) {
            auto q0 = Clock::now();
            std::vector<Vertex*> results = ClusterANN::clusterGreedySearch(g, queries[i], 125, /*p=*/1);
            sumQueryMs += std::chrono::duration<double, std::milli>(Clock::now() - q0).count();

            float recall = test.RecallAtK(results, knnResults[i]);
            sumRecall += recall;
            minRecall = std::min(minRecall, recall);
        }

        printf("  build %.0f ms | mean Recall@%u %.3f | min %.1f | avg query %.2f ms  (%u queries, L=125)\n",
               buildMs, k, sumRecall / numQueries, minRecall, sumQueryMs / numQueries, numQueries);
    }

    for (auto& v : vertices) delete v;

    if (ok) std::cout << "testClusterInstance PASSED\n";
    else std::cout << "testClusterInstance FAILED\n";

    return ok ? 0 : 1;
}

}  // namespace ClusterTest

int testClusterInstance() {
    return ClusterTest::run();
}
