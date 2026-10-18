#include "../../DiskANN-Cluster/clusterANN.h"
#include "../../Core/core_testing.h"
#include "../HardInstance.h"
#include <chrono>
#include <stdio.h>
#include <iostream>

// ===========================================================================
//  Hard instance (Testing/HardInstance.h) on the cluster graph.
//
//  Same points, query and build parameters as testHardInstance (n = 20000,
//  S = 1000, alpha = 2.0, R = 70, L = 125), built with each cluster edge mode.
//  The classic test checks that fast preprocessing stays trapped in P; here
//  the search starts from the coarse layer's medoid nearest q instead of the
//  dataset medoid, so the trap may or may not apply. Reports numbers only, no
//  recall threshold: the test fails only if a graph is structurally broken.
// ===========================================================================

namespace ClusterHardTest {

int run() {
    TestANN test;
    const uint32_t n = 20000, L = 125;
    HardInstance inst = makeHardInstance(n, 1000.0);
    if (!inst.ok || !checkSeparations(inst, L)) { inst.freeAll(); return 1; }

    // which group (M | P | P' | a) a vertex id falls in; ids are blocked in that order
    const uint32_t pStart = inst.mCluster, ppStart = pStart + inst.pCluster,
                   aStart = ppStart + inst.pPrimeCluster;
    auto group = [&](const Vertex* v) {
        return v->id < pStart ? "M" : v->id < ppStart ? "P" : v->id < aStart ? "P'" : "a";
    };

    // Same cluster knobs as TestClusterInstance.
    uint32_t numClusters = std::max<uint32_t>(1, static_cast<uint32_t>(
        std::lround(std::sqrt(static_cast<double>(n)))));
    uint32_t numAdjacentClusters = std::max<uint32_t>(1, static_cast<uint32_t>(
        std::lround(0.10 * numClusters)));
    const uint32_t clustersPerVertex = numAdjacentClusters, edgesPerCluster = 3;
    const double boundaryProportion = 0.2;

    using Clock = std::chrono::steady_clock;
    using ClusterANN::EdgeMode;
    bool ok = true;
    for (EdgeMode mode : {EdgeMode::Random, EdgeMode::Ranked, EdgeMode::Backbone}) {
        const char* label = mode == EdgeMode::Ranked   ? "ranked"
                          : mode == EdgeMode::Backbone ? "backbone"
                          :                              "old";
        printf("Cluster Hard Instance [%s boundary edges]\n", label);
        fflush(stdout);

        auto t0 = Clock::now();
        ClusterANN::ClusterGraph g = ClusterANN::CreateClusterGraph(
            inst.vertices, /*alpha=*/2.0f, /*pruneR=*/70,
            /*totalR=*/125, /*edgesPerPair=*/25, /*poolSize=*/40,
            numAdjacentClusters, mode, clustersPerVertex, edgesPerCluster,
            /*medoidR=*/24, boundaryProportion);
        double buildS = std::chrono::duration<double>(Clock::now() - t0).count();

        if (g.clusters.empty() || g.medoids.size() != g.clusters.size()) {
            std::cerr << "cluster graph [" << label << "] was not built correctly\n";
            ok = false;
            continue;
        }

        std::vector<Vertex*> seeds = ClusterANN::coarseGreedySearch(g, *inst.query, /*p=*/1);
        auto q0 = Clock::now();
        std::vector<Vertex*> U = ClusterANN::clusterGreedySearch(g, *inst.query, L, /*p=*/1);
        double queryMs = std::chrono::duration<double, std::milli>(Clock::now() - q0).count();

        float recall = test.RecallAtK(U, inst.answers);
        printf("  build %.1f s | start medoid in %s | visited %zu | Recall@5 %.2f | query %.2f ms\n",
               buildS, seeds.empty() ? "-" : group(seeds[0]), U.size(), recall, queryMs);
        fflush(stdout);
    }

    inst.freeAll();

    if (ok) std::cout << "testClusterHardInstance PASSED\n";
    else std::cout << "testClusterHardInstance FAILED\n";

    return ok ? 0 : 1;
}

}  // namespace ClusterHardTest

int testClusterHardInstance() {
    return ClusterHardTest::run();
}
