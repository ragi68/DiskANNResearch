#include "../../DiskANN-Patch/patchANN.h"
#include "../../DiskANN-Classic/DiskANN.h"
#include "../../Core/core_testing.h"
#include <random>
#include <chrono>
#include <unordered_map>
#include <stdio.h>
#include <iostream>

// ===========================================================================
//  Membership-cap sweep for the patch graph (DiskANN-Patch/patchANN.cpp).
//
//  Same data, queries and seed as the cluster tests. For each k in `caps`,
//  builds k * sqrt(n) random patches (each vertex in at most k), then runs the
//  classic GreedySearch from the dataset medoid, as the Vamana tests do.
//  Reports numbers only, no recall threshold: the test fails only if some
//  vertex does not end up in exactly k patches. Lives in its own namespace so
//  it doesn't collide with the other tests included into main.cpp.
// ===========================================================================

namespace PatchSweep {

int run() {
    TestANN test;
    std::mt19937 rng(42); // same seed as the cluster tests -> same data and queries
    uint32_t numVertices = 20000;
    uint32_t dim = 128;
    uint32_t k = 10;
    uint32_t L = 125;

    uint32_t numQueries = 100;

    const uint32_t caps[] = {2, 3, 4, 8, 12};  // 12 ~ n^(1/4) at n = 20000

    std::vector<Vertex*> vertices = test.RandomVertices(numVertices, dim, rng);

    // Queries get ids outside the range of generated vertices.
    std::vector<Vertex> queries(numQueries);
    for (uint32_t i = 0; i < numQueries; i++) {
        queries[i].id = numVertices + i;
        for (uint32_t j = 0; j < dim; j++)
            queries[i].data.push_back(std::uniform_real_distribution<float>(0.0f, 1000.0f)(rng));
    }

    // Brute-force ground truth, computed once and shared by every cap.
    std::vector<std::vector<Vertex*>> knnResults(numQueries);
    for (uint32_t i = 0; i < numQueries; i++)
        knnResults[i] = test.BruteForceKNN(vertices, queries[i], k);

    // Search start, same as the Vamana tests; patches don't move points, so it is fixed.
    Vertex* start = Medoid(vertices);

    using Clock = std::chrono::steady_clock;
    bool ok = true;

    printf("Patch graph membership-cap sweep (n=%u, dim=%u, %u queries, L=%u, alpha=1.2, R=70)\n",
           numVertices, dim, numQueries, L);
    printf("   k | patches |   build (s) | avg out-deg | mean Recall@%u |  min | avg query (ms)\n", k);
    fflush(stdout);  // flush per line so progress shows when output is redirected

    for (uint32_t cap : caps) {
        auto t0 = Clock::now();
        std::vector<PatchGraph::Patch> patches = PatchGraph::PatchANN(vertices, /*alpha=*/1.2f, /*R=*/70, cap);
        double buildS = std::chrono::duration<double>(Clock::now() - t0).count();

        // Structural check: the cap rule should leave every vertex in exactly cap patches.
        std::unordered_map<Vertex*, uint32_t> memberships;
        for (const PatchGraph::Patch& p : patches)
            for (Vertex* x : p.patchPoints) ++memberships[x];
        uint32_t wrong = 0;
        for (Vertex* x : vertices) {
            auto it = memberships.find(x);
            if (it == memberships.end() || it->second != cap) ++wrong;
        }
        if (wrong > 0) {
            printf("  %2u | %u vertices not in exactly %u patches\n", cap, wrong, cap);
            fflush(stdout);
            ok = false;
        }

        double sumDeg = 0.0;
        for (Vertex* x : vertices) sumDeg += x->out.size();

        double sumRecall = 0.0, sumQueryMs = 0.0;
        float minRecall = 1.0f;
        for (uint32_t i = 0; i < numQueries; i++) {
            auto q0 = Clock::now();
            std::vector<Vertex*> results = GreedySearch(*start, queries[i], L);
            sumQueryMs += std::chrono::duration<double, std::milli>(Clock::now() - q0).count();

            float recall = test.RecallAtK(results, knnResults[i]);
            sumRecall += recall;
            minRecall = std::min(minRecall, recall);
        }

        printf("  %2u | %7zu | %11.1f | %11.1f | %14.3f | %4.1f | %14.2f\n",
               cap, patches.size(), buildS, sumDeg / numVertices,
               sumRecall / numQueries, minRecall, sumQueryMs / numQueries);
        fflush(stdout);
    }

    for (auto& v : vertices) delete v;

    if (ok) std::cout << "testPatchSweep PASSED\n";
    else std::cout << "testPatchSweep FAILED\n";

    return ok ? 0 : 1;
}

}  // namespace PatchSweep

int testPatchSweep() {
    return PatchSweep::run();
}
