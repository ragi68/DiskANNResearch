#include "../DiskANN-Cluster/clusterANN.h"
#include <random>
#include <chrono>
#include <stdio.h>
#include <iostream>

// ===========================================================================
//  Smoke test for clusterANN (DiskANN-Cluster/clusterANN.cpp).
//
//  clusterANN.h puts its Vertex/distance/etc. under the ClusterANN namespace
//  so this file can be included alongside DiskANN.h-based tests (which have
//  identically-named globals) in the same main.cpp build. Everything here
//  lives in its own namespace too, so `using ClusterANN::Vertex` doesn't
//  collide with DiskANN.h's global ::Vertex in that shared build.
// ===========================================================================

namespace ClusterTest {

using ClusterANN::Vertex;

// brute force knn search for ground truth
std::vector<Vertex*> BruteForceKNN(const std::vector<Vertex*>& vertices, Vertex q, uint32_t k) {
    std::vector<Vertex*> sorted = vertices;
    std::sort(sorted.begin(), sorted.end(), [&](Vertex* a, Vertex* b) {
        return ClusterANN::distance(*a, q) < ClusterANN::distance(*b, q);
    });
    if (sorted.size() > k) sorted.resize(k);
    return sorted;
}

// generate random vertices
std::vector<Vertex*> RandomVertices(uint32_t n, uint32_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(0.0f, 1000.0f);
    std::vector<Vertex*> vertices;
    vertices.reserve(n);
    for (uint32_t i = 0; i < n; i++) {
        Vertex* v = new Vertex();
        v->id = i;
        for (uint32_t j = 0; j < dim; j++) v->data.push_back(dist(rng));
        vertices.push_back(v);
    }
    return vertices;
}

// fraction of `truth` (the brute-force top-k) present in `result`
float RecallAtK(const std::vector<Vertex*>& result, const std::vector<Vertex*>& truth) {
    if (truth.empty()) return 1.0f;
    std::unordered_set<uint32_t> found;
    for (Vertex* v : result) found.insert(v->id);
    uint32_t hits = 0;
    for (Vertex* v : truth)
        if (found.count(v->id)) ++hits;
    return static_cast<float>(hits) / static_cast<float>(truth.size());
}

int run() {
    std::mt19937 rng(42); // fixed seed for reproducibility
    uint32_t numVertices = 20000;
    uint32_t dim = 128;
    uint32_t k = 10;

    uint32_t numQueries = 100;

    std::vector<Vertex*> vertices = RandomVertices(numVertices, dim, rng);

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
        knnResults[i] = BruteForceKNN(vertices, queries[i], k);

    // number of clusters C = round(sqrt(numVertices)) (see CreateClusterGraph);
    // ~10% of that many nearest foreign clusters get boundary-edge candidates
    // from each point, instead of a small fixed count.
    uint32_t numClusters = std::max<uint32_t>(1, static_cast<uint32_t>(
        std::lround(std::sqrt(static_cast<double>(numVertices)))));
    uint32_t numAdjacentClusters = std::max<uint32_t>(1, static_cast<uint32_t>(
        std::lround(0.10 * numClusters)));

    // Two constructions on the same vertices and queries: old random pool edges,
    // then rank-weighted edges. CreateClusterGraph clears every out list first,
    // so each build starts clean; each graph is searched before the next build
    // overwrites the shared vertices' edges. Reports numbers only, no recall
    // threshold: the test fails only if a graph is structurally broken.
    //
    // Ranked-edge knobs: each vertex sends edgesPerCluster edges into each of its
    // clustersPerVertex nearest foreign clusters (10% of all clusters, same as
    // numAdjacentClusters: 14 at n=20000).
    const uint32_t clustersPerVertex = numAdjacentClusters, edgesPerCluster = 3;

    using Clock = std::chrono::steady_clock;
    bool ok = true;
    for (bool ranked : {false, true}) {
        const char* label = ranked ? "ranked" : "old";
        std::cout << "Cluster Graph Creation [" << label << " boundary edges]";
        if (ranked) std::cout << " (clustersPerVertex=" << clustersPerVertex
                              << ", edgesPerCluster=" << edgesPerCluster << ")";
        std::cout << "\n";

        auto t0 = Clock::now();
        ClusterANN::ClusterGraph g = ClusterANN::CreateClusterGraph(
            vertices, /*alpha=*/1.2f, /*pruneR=*/70,
            /*totalR=*/125, /*edgesPerPair=*/25, /*poolSize=*/40,
            numAdjacentClusters, ranked, clustersPerVertex, edgesPerCluster);
        double buildMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

        if (!g.hub || g.clusters.empty() || g.medoids.size() != g.clusters.size()) {
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

            float recall = RecallAtK(results, knnResults[i]);
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
