#include "clusterANN.h"

namespace ClusterANN {

    std::vector<Cluster> KClustering(std::vector<Vertex*> v, uint32_t c, uint32_t I) {
        std::vector<Cluster> clusters;
        const size_t n = v.size();
        if (n == 0 || c == 0) return clusters;
        if (c > n) c = static_cast<uint32_t>(n);
        const size_t dim = v[0]->data.size();

        // Squared distance: same ordering as distance(), no sqrt needed for argmin.
        auto sqDist = [dim](const std::vector<float>& a, const std::vector<float>& b) {
            float s = 0.0f;
            for (size_t i = 0; i < dim; ++i) { float d = a[i] - b[i]; s += d * d; }
            return s;
        };

        std::mt19937 rng(12345);  // fixed seed for reproducible experiments

        // ---- k-means++ initialization ----
        std::vector<std::vector<float>> cent;
        cent.reserve(c);
        std::uniform_int_distribution<size_t> pick(0, n - 1);
        cent.push_back(v[pick(rng)]->data);

        std::vector<float> minD(n);
        for (size_t i = 0; i < n; ++i) minD[i] = sqDist(v[i]->data, cent[0]);

        while (cent.size() < c) {
            double total = std::accumulate(minD.begin(), minD.end(), 0.0);
            size_t next = n - 1;
            if (total <= 0.0) {
                next = pick(rng);  // all remaining points duplicate existing centers
            } else {
                std::uniform_real_distribution<double> u(0.0, total);
                double r = u(rng), acc = 0.0;
                for (size_t i = 0; i < n; ++i) {
                    acc += minD[i];
                    if (acc >= r) { next = i; break; }
                }
            }
            cent.push_back(v[next]->data);
            for (size_t i = 0; i < n; ++i)
                minD[i] = std::min(minD[i], sqDist(v[i]->data, cent.back()));
        }

        // ---- Assignment step: each point to its nearest centroid ----
        std::vector<uint32_t> assign(n, UINT32_MAX);
        std::vector<float> assignD(n, 0.0f);
        auto assignAll = [&]() -> bool {
            bool changed = false;
            for (size_t i = 0; i < n; ++i) {
                uint32_t best = 0;
                float bestD = sqDist(v[i]->data, cent[0]);
                for (uint32_t j = 1; j < c; ++j) {
                    float d = sqDist(v[i]->data, cent[j]);
                    if (d < bestD) { bestD = d; best = j; }
                }
                if (best != assign[i]) changed = true;
                assign[i] = best;
                assignD[i] = bestD;
            }
            return changed;
        };

        assignAll();

        // ---- Lloyd iterations: update centroids, then reassign ----
        for (uint32_t it = 0; it < I; ++it) {
            std::vector<std::vector<double>> sum(c, std::vector<double>(dim, 0.0));
            std::vector<size_t> cnt(c, 0);
            for (size_t i = 0; i < n; ++i) {
                uint32_t j = assign[i];
                ++cnt[j];
                for (size_t k = 0; k < dim; ++k) sum[j][k] += v[i]->data[k];
            }
            for (uint32_t j = 0; j < c; ++j) {
                if (cnt[j] > 0) {
                    for (size_t k = 0; k < dim; ++k)
                        cent[j][k] = static_cast<float>(sum[j][k] / cnt[j]);
                } else {
                    // Empty cluster: reseed at the point farthest from its current centroid.
                    size_t far = static_cast<size_t>(
                        std::max_element(assignD.begin(), assignD.end()) - assignD.begin());
                    cent[j] = v[far]->data;
                    assignD[far] = 0.0f;  // avoid reusing the same point for another empty cluster
                }
            }
            if (!assignAll()) break;  // converged
        }

        // ---- Build output, skipping any cluster that ended up empty ----
        std::vector<uint32_t> remap(c, UINT32_MAX);
        for (size_t i = 0; i < n; ++i) {
            uint32_t j = assign[i];
            if (remap[j] == UINT32_MAX) {
                remap[j] = static_cast<uint32_t>(clusters.size());
                clusters.push_back(Cluster{remap[j], {}});
            }
            clusters[remap[j]].clusterPoints.push_back(v[i]);
        }
        return clusters;
    }

    ClusterGraph CreateClusterGraph(std::vector<Vertex*> v, float alpha, uint32_t pruneR, uint32_t totalR,
        uint32_t edgesPerPair, uint32_t poolSize, uint32_t numAdjacentClusters, EdgeMode edgeMode,
        uint32_t clustersPerVertex, uint32_t edgesPerCluster, uint32_t medoidR, double boundaryProportion){
        ClusterGraph g;
        if (v.empty()) return g;

        uint32_t c = std::max<uint32_t>(1, static_cast<uint32_t>(std::sqrt(static_cast<double>(v.size()))));
        g.clusters = KClustering(v, c, 20);
        g.medoids.reserve(g.clusters.size());

        // ---- Slow preprocessing inside each cluster ----
        for (Cluster& cl : g.clusters) {
            // Start from an empty graph: RobustPruning merges v.out into U,
            // so stale edges would leak into the candidate set.
            for (Vertex* p : cl.clusterPoints) p->out.clear();

            // Prune each point against every point in its own cluster.
            for (Vertex* p : cl.clusterPoints)
                RobustPruning(*p, cl.clusterPoints, alpha, pruneR);

            g.medoids.push_back(Medoid(cl.clusterPoints));
        }

        // ---- Coarse layer: slow preprocessing over the medoids ----
        topLevelEdges(g, alpha, medoidR);

        // Cross-cluster edges. Backbone must run on the in-cluster-only out lists
        // (its boundary score reads them), so it is an alternative, not an add-on.
        switch (edgeMode) {
            case EdgeMode::Random:
                AddBoundaryEdges(g, edgesPerPair, poolSize, totalR, numAdjacentClusters);
                break;
            case EdgeMode::Ranked:
                AddBoundaryEdgesRanked(g, totalR, /*candperCluster=*/50, clustersPerVertex, edgesPerCluster);
                break;
            case EdgeMode::Backbone:
                BackBoneGraph(g, alpha, pruneR, totalR, boundaryProportion);
                break;
        }

        return g;
    }

    // Coarse search over the medoid graph picks the top p medoids, then classic
    // best-first greedy search runs from all of them at once.
    std::vector<Vertex*> clusterGreedySearch(const ClusterGraph& g, Vertex& q, uint32_t L,
                                             uint32_t p, uint32_t coarseL) {
    if (L == 0) L = 1;

    // ---- Phase 1: coarse search -> top p medoids ----
    std::vector<Vertex*> seeds = coarseGreedySearch(g, q, p, coarseL);
    if (seeds.empty()) return {};

    // ---- Phase 2: classic greedy search seeded with those medoids ----
    using Entry = std::pair<float, Vertex*>;  // (distance to q, vertex), cached
    std::vector<Entry> A;                     // beam (may contain visited vertices, as in the original)
    std::vector<Entry> U;                     // visited, in expansion order
    std::unordered_set<Vertex*> inA, visited;

    for (Vertex* m : seeds)
        if (inA.insert(m).second) A.push_back({distance(*m, q), m});

    while (true) {
        // Closest vertex in A \ U
        int best = -1;
        for (size_t i = 0; i < A.size(); ++i) {
            if (visited.count(A[i].second)) continue;
            if (best < 0 || A[i].first < A[best].first) best = static_cast<int>(i);
        }
        if (best < 0) break;

        // Copy before modifying A (push_back can invalidate references)
        Vertex* v = A[best].second;
        float dv = A[best].first;
        visited.insert(v);
        U.push_back({dv, v});

        // Expand neighbors
        for (Vertex* nb : v->out)
            if (inA.insert(nb).second)
                A.push_back({distance(*nb, q), nb});

        // Trim the beam to the L closest to q
        if (A.size() > L) {
            std::sort(A.begin(), A.end(),
                      [](const Entry& a, const Entry& b) { return a.first < b.first; });
            for (size_t i = L; i < A.size(); ++i) inA.erase(A[i].second);
            A.resize(L);
        }
    }

    std::sort(U.begin(), U.end(),
              [](const Entry& a, const Entry& b) { return a.first < b.first; });
    std::vector<Vertex*> result;
    result.reserve(U.size());
    for (const Entry& e : U) result.push_back(e.second);
    return result;
}

}  

