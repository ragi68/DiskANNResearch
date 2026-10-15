#include "patchANN.h"

namespace PatchGraph {

    std::vector<Patch> PatchANN(std::vector<Vertex*>& v, float alpha, uint32_t R, uint32_t k, uint32_t seed) {
        std::vector<Patch> patches;
        const size_t n = v.size();
        if (n == 0 || k == 0) return patches;

        // up to k * ceil(sqrt(n)) patches of ceil(sqrt(n)) vertices each; ceil leaves
        // spare slots (k * s^2 > k * n) unless n is a perfect square
        const size_t s = std::max<size_t>(1, static_cast<size_t>(
            std::ceil(std::sqrt(static_cast<double>(n)))));
        const size_t maxPatches = static_cast<size_t>(k) * s;

        std::mt19937 rng(seed);  // fixed seed for reproducible experiments

        // Start from an empty graph: RobustPruning merges v.out into U,
        // so stale edges would leak into the candidate set.
        for (Vertex* p : v) p->out.clear();

        // pool[0, poolSize) holds the indices of vertices still under the cap.
        std::vector<size_t> pool(n);
        std::iota(pool.begin(), pool.end(), 0);
        size_t poolSize = n;
        std::vector<uint32_t> memberships(n, 0);

        patches.reserve(maxPatches);
        for (size_t j = 0; j < maxPatches && poolSize > 0; ++j) {
            const size_t patchSize = std::min(s, poolSize);

            // Partial Fisher-Yates over the pool: pool[0, patchSize) becomes a
            // uniform random subset of distinct vertices still under the cap.
            for (size_t i = 0; i < patchSize; ++i) {
                std::uniform_int_distribution<size_t> pick(i, poolSize - 1);
                std::swap(pool[i], pool[pick(rng)]);
            }

            Patch patch{static_cast<uint32_t>(j), {}};
            patch.patchPoints.reserve(patchSize);
            for (size_t i = 0; i < patchSize; ++i) patch.patchPoints.push_back(v[pool[i]]);

            // Count the memberships; a vertex at the cap is swapped out past the
            // end of the pool. Walk backwards so swaps only touch processed slots.
            for (size_t i = patchSize; i-- > 0; ) {
                if (++memberships[pool[i]] >= k) {
                    std::swap(pool[i], pool[poolSize - 1]);
                    --poolSize;
                }
            }

            // ---- Slow preprocessing inside the patch ----
            for (Vertex* p : patch.patchPoints)
                RobustPruning(*p, patch.patchPoints, alpha, R);

            patches.push_back(std::move(patch));
        }

        return patches;
    }

}  // namespace PatchGraph
