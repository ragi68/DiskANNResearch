#include "clusterANN.h"

// Cross-cluster (boundary) edge construction for the cluster graph.
// Split out of clusterANN.cpp; declarations live in clusterANN.h.

namespace ClusterANN {

    // Pools each cluster's points nearest its numAdjacentClusters neighbor clusters, then adds
    // edgesPerPair random edges between the facing pools of every adjacent cluster pair.
    void AddBoundaryEdges(ClusterGraph& g, uint32_t edgesPerPair, uint32_t poolSize,
                        uint32_t R, uint32_t numAdjacentClusters, uint32_t seed) {
        const uint32_t C = static_cast<uint32_t>(g.clusters.size());
        if (C < 2 || edgesPerPair == 0 || poolSize == 0) return;
        if (numAdjacentClusters == 0) numAdjacentClusters = 1;

        using Cand = std::pair<float, Vertex*>;  // (margin, point); smaller margin = closer to the boundary
        // cand[i][j] = points of cluster i that face cluster j
        std::vector<std::unordered_map<uint32_t, std::vector<Cand>>> cand(C);

        // ---- Each point joins the pools facing its numAdjacentClusters nearest foreign medoids ----
        for (uint32_t i = 0; i < C; ++i) {
            const Vertex& mi = *g.medoids[i];
            const uint32_t k = std::min(numAdjacentClusters, C - 1);
            for (Vertex* x : g.clusters[i].clusterPoints) {
                float dOwn = distance(*x, mi);

                std::vector<std::pair<float, uint32_t>> foreign;  // (dist, cluster)
                foreign.reserve(C - 1);
                for (uint32_t j = 0; j < C; ++j) {
                    if (j == i) continue;
                    foreign.push_back({distance(*x, *g.medoids[j]), j});
                }
                std::partial_sort(foreign.begin(), foreign.begin() + k, foreign.end(),
                                [](const auto& a, const auto& b) { return a.first < b.first; });

                for (uint32_t t = 0; t < k; ++t)
                    cand[i][foreign[t].second].push_back({foreign[t].first - dOwn, x});
            }
        }

        // ---- Keep only the poolSize points closest to each boundary ----
        auto trim = [poolSize](std::vector<Cand>& vec) {
            if (vec.size() > poolSize) {
                std::nth_element(vec.begin(), vec.begin() + poolSize, vec.end(),
                                [](const Cand& a, const Cand& b) { return a.first < b.first; });
                vec.resize(poolSize);
            }
        };
        for (auto& row : cand)
            for (auto& kv : row) trim(kv.second);

        // ---- Symmetric adjacency ----
        std::set<std::pair<uint32_t, uint32_t>> adj;
        for (uint32_t i = 0; i < C; ++i)
            for (auto& kv : cand[i])
                adj.insert({std::min(i, kv.first), std::max(i, kv.first)});

        // Pool of cluster i facing j; if empty, fall back to i's points closest to j's medoid.
        auto getPool = [&](uint32_t i, uint32_t j) -> std::vector<Cand>& {
            std::vector<Cand>& pool = cand[i][j];  // references into unordered_map stay valid on insert
            if (pool.empty()) {
                const Vertex& mj = *g.medoids[j];
                for (Vertex* x : g.clusters[i].clusterPoints)
                    pool.push_back({distance(*x, mj), x});
                trim(pool);
            }
            return pool;
        };

        std::mt19937 rng(seed);
        auto hasEdge = [](Vertex* a, Vertex* b) {
            return std::find(a->out.begin(), a->out.end(), b) != a->out.end();
        };

        // ---- Random edges from boundary points of i to boundary points of j ----
        auto addDirected = [&](uint32_t i, uint32_t j) {
            std::vector<Cand>& src = getPool(i, j);
            std::vector<Cand>& dst = getPool(j, i);
            if (src.empty() || dst.empty()) return;
            std::uniform_int_distribution<size_t> ps(0, src.size() - 1), pd(0, dst.size() - 1);

            uint32_t added = 0, attempts = 0;
            while (added < edgesPerPair && attempts < 4 * edgesPerPair) {
                ++attempts;
                Vertex* s = src[ps(rng)].second;
                Vertex* t = dst[pd(rng)].second;
                if (s->out.size() >= R || hasEdge(s, t)) continue;  // cap blocks new edges; never evicts
                s->out.push_back(t);
                ++added;
            }
        };

        for (const auto& pr : adj) {
            addDirected(pr.first, pr.second);
            addDirected(pr.second, pr.first);
        }
    }

    // Each vertex sends edgesPerCluster edges (plus reverse edges) into each of its clustersPerVertex
    // nearest clusters by bisector distance, targets drawn with P ~ 1/rank from its candperCluster nearest there.
    void AddBoundaryEdgesRanked(ClusterGraph& g, uint32_t R, uint32_t candperCluster,
        uint32_t clustersPerVertex, uint32_t edgesPerCluster, uint32_t seed){
            const uint32_t C = static_cast<uint32_t>(g.clusters.size());
            if(C < 2 || candperCluster == 0) return;

            //pairwise medoid distances for bisectors
            std::vector<std::vector<float>> D(C, std::vector<float>(C, 0.0f));
            for(uint32_t i = 0; i < C; ++i){
                for(uint32_t j = i+ 1; j< C; ++j){
                    D[i][j] = D[j][i] = distance(*g.medoids[i], *g.medoids[j]);
                }
            }

            //rank weights to 1/r, shared every draw and truncated to candidate count
            std::vector<double> w(candperCluster);
            for (uint32_t r = 0; r < candperCluster; ++r) w[r] = 1.0 / (r + 1);
            auto hasEdge = [](Vertex* a, Vertex* b){
                return std::find(a->out.begin(), a->out.end(), b) != a->out.end();
            };

            using Cand = std::pair <float, Vertex*>; //distance to x, point
            //add up to 'count' rank weight edges from X into cluster j
            std::mt19937 rng(seed);
            auto addEdges = [&](Vertex* x, uint32_t j, uint32_t count){
                if (count == 0) return;
                const std::vector<Vertex*>& pts = g.clusters[j].clusterPoints;
                std::vector<Cand> cand;
                cand.reserve(pts.size());
                for(Vertex* y : pts) cand.push_back({distance(*x, *y), y});

                const size_t k = std::min<size_t>(candperCluster, cand.size());
                if(k == 0) return;
                std::partial_sort(cand.begin(), cand.begin() + k, cand.end(), [](const Cand& a, const Cand& b)
                    {return a.first < b.first;});
                std::discrete_distribution<size_t> pick(w.begin(), w.begin() + k);

                uint32_t added = 0, attempts = 0;
                while (added < count && attempts < 4 * count){
                    ++attempts;
                    if(x->out.size() >= R) return;
                    Vertex* y = cand[pick(rng)].second;
                    if(hasEdge(x, y)) continue;
                    x->out.push_back(y);
                    ++added;

                    // Reverse edge y -> x so the seam is crossable both ways.
                    // Subject to the same cap; doesn't count toward `count`.
                    if(y->out.size() < R && !hasEdge(y, x)) y->out.push_back(x);

                }
            };

            //for each point, find its clustersPerVertex nearest bisectors and add edges.
            if (clustersPerVertex == 0 || edgesPerCluster == 0) return;
            std::vector<std::pair<float, uint32_t>> bis;  // (bisector distance, cluster)
            bis.reserve(C);
            for(uint32_t i = 0; i < C; ++i){
                for(Vertex* x: g.clusters[i].clusterPoints){
                    // Distance from x to the (medoid i, medoid j) bisector; negative means x is past the seam.
                    const float dOwn = distance(*x, *g.medoids[i]);
                    bis.clear();
                    for (uint32_t j = 0; j < C; ++j) {
                        if (j == i || D[i][j] <= 0.0f) continue;  // skip coincident medoids
                        const float dj = distance(*x, *g.medoids[j]);
                        bis.push_back({(dj * dj - dOwn * dOwn) / (2.0f * D[i][j]), j});
                    }

                    const size_t t = std::min<size_t>(clustersPerVertex, bis.size());
                    std::partial_sort(bis.begin(), bis.begin() + t, bis.end());
                    for (size_t c = 0; c < t; ++c) addEdges(x, bis[c].second, edgesPerCluster);
                }
            }
        }

}  // namespace ClusterANN
