#include "../../DiskANN-Patch/patchANN.h"
#include "../../DiskANN-Classic/DiskANN.h"
#include "../../Core/core_testing.h"
#include "../HardInstance.h"
#include <chrono>
#include <unordered_map>
#include <stdio.h>
#include <iostream>

// ===========================================================================
//  Hard instance (Testing/HardInstance.h) on the patch graph.
//
//  Same points, query and build parameters as testHardInstance (n = 20000,
//  S = 1000, alpha = 2.0, R = 70, L = 125), built with PatchANN for each
//  membership cap k, and searched like the classic test: GreedySearch from
//  the dataset medoid, which lies in M. Reports numbers only, no recall
//  threshold: the test fails only if some vertex is not in exactly k patches.
// ===========================================================================

namespace PatchHardTest {

int run() {
    TestANN test;
    const uint32_t n = 50000, L = 125;
    HardInstance inst = makeHardInstance(n, 1000.0);
    if (!inst.ok || !checkSeparations(inst, L)) { inst.freeAll(); return 1; }

    const uint32_t caps[] = {2, 3, 4, 8, 12};  // same caps as PatchSweepTest

    // Search start, same as testHardInstance; patches don't move points, so it is fixed.
    Vertex* start = Medoid(inst.vertices);

    using Clock = std::chrono::steady_clock;
    bool ok = true;

    printf("Patch Hard Instance membership-cap sweep (n=%u, L=%u, alpha=2.0, R=70)\n", n, L);
    printf("   k | patches |   build (s) | avg out-deg | visited | Recall@5 | query (ms)\n");
    fflush(stdout);  // flush per line so progress shows when output is redirected

    for (uint32_t cap : caps) {
        auto t0 = Clock::now();
        std::vector<PatchGraph::Patch> patches = PatchGraph::PatchANN(inst.vertices, /*alpha=*/2.0f, /*R=*/70, cap);
        double buildS = std::chrono::duration<double>(Clock::now() - t0).count();

        // Structural check: the cap rule should leave every vertex in exactly cap patches.
        std::unordered_map<Vertex*, uint32_t> memberships;
        for (const PatchGraph::Patch& p : patches)
            for (Vertex* x : p.patchPoints) ++memberships[x];
        uint32_t wrong = 0;
        for (Vertex* x : inst.vertices) {
            auto it = memberships.find(x);
            if (it == memberships.end() || it->second != cap) ++wrong;
        }
        if (wrong > 0) {
            printf("  %2u | %u vertices not in exactly %u patches\n", cap, wrong, cap);
            fflush(stdout);
            ok = false;
        }

        double sumDeg = 0.0;
        for (Vertex* x : inst.vertices) sumDeg += x->out.size();

        auto q0 = Clock::now();
        std::vector<Vertex*> U = GreedySearch(*start, *inst.query, L);
        double queryMs = std::chrono::duration<double, std::milli>(Clock::now() - q0).count();

        float recall = test.RecallAtK(U, inst.answers);
        printf("  %2u | %7zu | %11.1f | %11.1f | %7zu | %8.2f | %10.2f\n",
               cap, patches.size(), buildS, sumDeg / n, U.size(), recall, queryMs);
        fflush(stdout);
    }

    inst.freeAll();

    if (ok) std::cout << "testPatchHardInstance PASSED\n";
    else std::cout << "testPatchHardInstance FAILED\n";

    return ok ? 0 : 1;
}

}  // namespace PatchHardTest

int testPatchHardInstance() {
    return PatchHardTest::run();
}
