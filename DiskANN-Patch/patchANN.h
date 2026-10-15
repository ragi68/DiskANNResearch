#pragma once
#include <stdint.h>
#include <vector>
#include <algorithm>
#include <math.h>
#include <random>
#include <numeric>
#include "../Core/core.h"  // Vertex, distance, RobustPruning, RobustPruningOcclude, Medoid

// Patch-specific code lives in PatchGraph; the shared core is pulled in below
// so unqualified calls resolve to Core/.
namespace PatchGraph {

using ::Vertex;
using ::distance;
using ::RobustPruning;

struct Patch {
  uint32_t patchID;
  std::vector<Vertex*> patchPoints;
};

// Setup: draws up to k * s random patches of s = ceil(sqrt(n)) distinct vertices
// each. Patches are sampled from a pool, and a vertex leaves the pool once it has
// been in k patches, so when the pool runs dry every vertex is in exactly k patches
// (the last patch may be smaller). Patches overlap for k >= 2. If n is a perfect
// square there are no spare slots and the patch cap can be hit with a few vertices
// still short of k.
// Slow preprocessing inside each patch: every point is RobustPruned against its
// whole patch. A vertex in several patches keeps its earlier edges as candidates
// (RobustPruning merges v.out), capped at R.
std::vector<Patch> PatchANN(std::vector<Vertex*>& v, float alpha, uint32_t R, uint32_t k = 2,
    uint32_t seed = 12345);

}  // namespace PatchGraph
