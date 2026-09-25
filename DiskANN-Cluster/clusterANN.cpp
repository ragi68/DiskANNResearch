#include "clusterANN.h"

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

void RobustPruning(Vertex& v, std::vector<Vertex*> U, float alpha, uint32_t R) {

    U.insert(U.end(), v.out.begin(), v.out.end());

    U.erase(std::remove_if(U.begin(), 
        U.end(),
        [&](Vertex* p) { return p->id == v.id; }), 
        U.end()
    );
    
    v.out.clear();

    //loop thru the vertices in U sorted by distance to V
    std::sort(U.begin(), U.end(), [&](Vertex* a, Vertex* b) {
        return distance(*a, v) < distance(*b, v);
    });
    
    while(!U.empty() && v.out.size() < R) {
        //pick lowest distance vertex and add to v.next.
        Vertex* u = U.front();
        v.out.push_back(u);

        //get remaining vertices that are not pruned   
        std::vector<Vertex*> remaining;
        remaining.reserve(U.size());

        for(int i = 1; i < U.size(); i++) {
            Vertex* w = U[i];
            if(!(distance(*u, *w) * alpha <= distance(v, *w))) {
                remaining.push_back(w);
            }
        }
        //reset U to remainders
        U = std::move(remaining);
    }


}

Vertex* Medoid(std::vector<Vertex*> v){
    std::vector<float> mean(v[0]->data.size(), 0.0f);
    for (Vertex* u : v)
        for (size_t i = 0; i < u->data.size(); i++) mean[i] += u->data[i];
    for (auto& m : mean) m /= v.size();

    Vertex meanV; meanV.data = mean;
    Vertex* best = nullptr; float bestD = 0;
    for (Vertex* u : v) {
        float d = distance(*u, meanV);
        if (!best || d < bestD) { best = u; bestD = d; }
    }
    return best;
}

float distance(const Vertex& u, const Vertex& v) {
  float dist = 0.0f;
  for (uint32_t i = 0; i < u.data.size(); ++i) {
    float diff = u.data[i] - v.data[i];
    dist += diff * diff;
  }
  return sqrt(dist);
}

ClusterGraph CreateClusterGraph(std::vector<Vertex*> v, float alpha, uint32_t pruneR, uint32_t totalR,
    uint32_t edgesPerPair, uint32_t poolSize){
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

    // ---- Universal node ----
    g.hub = std::make_unique<Vertex>();
    g.hub->id = UINT32_MAX;  // sentinel; assumes real ids are < UINT32_MAX

    // Give the hub real coordinates (the global mean) so distance(hub, x) is
    // well defined. Otherwise distance() on its empty data vector would read
    // out of bounds whenever the hub is the second argument.
    const size_t dim = v[0]->data.size();
    std::vector<double> mean(dim, 0.0);
    for (Vertex* p : v)
        for (size_t k = 0; k < dim; ++k) mean[k] += p->data[k];
    g.hub->data.resize(dim);
    for (size_t k = 0; k < dim; ++k)
        g.hub->data[k] = static_cast<float>(mean[k] / v.size());

    g.hub->out = g.medoids;  // hub -> medoid of every cluster

    AddBoundaryEdges(g, edgesPerPair, poolSize, totalR);
    
    return g;
}

void AddBoundaryEdges(ClusterGraph& g, uint32_t edgesPerPair, uint32_t poolSize,
                      uint32_t R, uint32_t seed) {
    const uint32_t C = static_cast<uint32_t>(g.clusters.size());
    if (C < 2 || edgesPerPair == 0 || poolSize == 0) return;

    using Cand = std::pair<float, Vertex*>;  // (margin, point); smaller margin = closer to the boundary
    // cand[i][j] = points of cluster i that face cluster j
    std::vector<std::unordered_map<uint32_t, std::vector<Cand>>> cand(C);

    // ---- Classify every point by its nearest foreign medoid ----
    for (uint32_t i = 0; i < C; ++i) {
        const Vertex& mi = *g.medoids[i];
        for (Vertex* x : g.clusters[i].clusterPoints) {
            float dOwn = distance(*x, mi);
            uint32_t bestJ = UINT32_MAX;
            float bestD = std::numeric_limits<float>::max();
            for (uint32_t j = 0; j < C; ++j) {
                if (j == i) continue;
                float d = distance(*x, *g.medoids[j]);
                if (d < bestD) { bestD = d; bestJ = j; }
            }
            cand[i][bestJ].push_back({bestD - dOwn, x});
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

    // Pool of cluster i facing j. If no point of i had j as its nearest foreign
    // medoid (the adjacency came from j's side only), fall back to the points
    // of i closest to j's medoid.
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


// Start from hub universal node, jump to the nearest medoid, then run
// classic best-first greedy search from there.
std::vector<Vertex*> clusterGreedySearch(Vertex& s, Vertex& q, uint32_t L) {
    if (L == 0) L = 1;

    // ---- Phase 1: hub -> nearest medoid ----
    Vertex* start = nullptr;
    float startD = 0.0f;
    for (Vertex* m : s.out) {
        if (m->id == UINT32_MAX) continue;
        float d = distance(*m, q);
        if (!start || d < startD) { start = m; startD = d; }
    }
    if (!start) return {};

    // ---- Phase 2: classic greedy search from the medoid ----
    using Entry = std::pair<float, Vertex*>;  // (distance to q, vertex), cached
    std::vector<Entry> A;                     // beam (may contain visited vertices, as in the original)
    std::vector<Entry> U;                     // visited, in expansion order
    std::unordered_set<Vertex*> inA, visited;

    A.push_back({startD, start});
    inA.insert(start);

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

        // Expand neighbors; never re-enter the hub
        for (Vertex* nb : v->out) {
            if (nb->id == UINT32_MAX) continue;
            if (inA.insert(nb).second)
                A.push_back({distance(*nb, q), nb});
        }

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

