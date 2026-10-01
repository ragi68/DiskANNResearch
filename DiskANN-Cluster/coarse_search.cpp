#include "clusterANN.h"

// Coarse (top-level) layer of the cluster graph: a navigable graph over the cluster medoids.

namespace ClusterANN {

    // Slow preprocessing on the medoids alone: each medoid is RobustPruned against all medoids,
    // giving an alpha-RNG over clusters. Edges go to g.medoidGraph so medoids' fine-layer out lists are untouched.
    void topLevelEdges(ClusterGraph& g, float alpha, uint32_t medoidR) {
        const uint32_t C = static_cast<uint32_t>(g.medoids.size());
        g.medoidGraph.assign(C, {});
        if (C < 2 || medoidR == 0) return;

        // Proxies share the medoids' coordinates but have empty out lists; id = cluster index,
        // which also lets RobustPruning drop the self-candidate.
        std::vector<Vertex> proxy(C);
        std::vector<Vertex*> all(C);
        for (uint32_t j = 0; j < C; ++j) {
            proxy[j].id = j;
            proxy[j].data = g.medoids[j]->data;
            all[j] = &proxy[j];
        }

        for (uint32_t j = 0; j < C; ++j) {
            RobustPruning(proxy[j], all, alpha, medoidR);
            g.medoidGraph[j].reserve(proxy[j].out.size());
            for (Vertex* nb : proxy[j].out) g.medoidGraph[j].push_back(nb->id);
        }

        // Fixed coarse entry point: the medoid of the medoids.
        g.medoidEntry = Medoid(all)->id;
    }

    std::vector<Vertex*> coarseGreedySearch(const ClusterGraph& g, const Vertex& q, uint32_t p, uint32_t coarseL) {
        const uint32_t C = static_cast<uint32_t>(g.medoids.size());
        if (C == 0 || p == 0) return {};
        p = std::min(p, C);
        coarseL = std::max(coarseL, p);  // the beam must be able to hold p results

        using Entry = std::pair<float, uint32_t>;  // (distance to q, cluster index)
        auto byDist = [](const Entry& a, const Entry& b) { return a.first < b.first; };
        std::vector<Entry> U;  // visited medoids

        if (g.medoidGraph.size() != C) {
            // Coarse layer not built: fall back to scanning every medoid.
            for (uint32_t j = 0; j < C; ++j) U.push_back({distance(*g.medoids[j], q), j});
        } else {
            std::vector<Entry> A;  // beam
            std::vector<char> inA(C, 0), visited(C, 0);
            const uint32_t s = g.medoidEntry;
            A.push_back({distance(*g.medoids[s], q), s});
            inA[s] = 1;

            while (true) {
                // Closest medoid in A \ U
                int best = -1;
                for (size_t i = 0; i < A.size(); ++i) {
                    if (visited[A[i].second]) continue;
                    if (best < 0 || A[i].first < A[best].first) best = static_cast<int>(i);
                }
                if (best < 0) break;

                const Entry e = A[best];
                visited[e.second] = 1;
                U.push_back(e);

                for (uint32_t nb : g.medoidGraph[e.second])
                    if (!inA[nb]) { inA[nb] = 1; A.push_back({distance(*g.medoids[nb], q), nb}); }

                if (A.size() > coarseL) {
                    std::sort(A.begin(), A.end(), byDist);
                    for (size_t i = coarseL; i < A.size(); ++i) inA[A[i].second] = 0;
                    A.resize(coarseL);
                }
            }
        }

        const size_t k = std::min<size_t>(p, U.size());
        std::partial_sort(U.begin(), U.begin() + k, U.end(), byDist);
        std::vector<Vertex*> top;
        top.reserve(k);
        for (size_t i = 0; i < k; ++i) top.push_back(g.medoids[U[i].second]);
        return top;
    }

}  // namespace ClusterANN
