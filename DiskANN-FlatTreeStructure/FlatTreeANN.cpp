#include "FlatTreeANN.h"
#include <cmath>
#include <stdexcept>

namespace FlatTree {
    // Shuffle and cut into numParts disjoint random slices (0 = sqrt(n)) of edge-free
    // mini vertices. Also indexes every point into global. The caller owns the nodes.
    std::vector<Node*> FlatTreeANN::makePartitions(std::vector<Vertex*> points, uint32_t numParts,
        uint32_t seed) {
        const uint32_t n = static_cast<uint32_t>(points.size());
        global.assign(n, nullptr);
        for (Vertex* v : points) {
            if (v->id >= n || global[v->id]) throw std::invalid_argument("ids must be dense 0..n-1");
            global[v->id] = v;
        }
        if (n == 0) return {};
        if (numParts == 0)
            numParts = static_cast<uint32_t>(std::lround(std::sqrt(static_cast<double>(n))));
        numParts = std::clamp<uint32_t>(numParts, 1, n);  // no empty partitions

        std::mt19937 rng(seed);  // fixed seed for reproducible experiments
        std::shuffle(points.begin(), points.end(), rng);

        std::vector<Node*> parts;
        parts.reserve(numParts);
        for (uint32_t i = 0; i < numParts; i++) {
            const uint32_t begin = static_cast<uint32_t>(static_cast<uint64_t>(n) * i / numParts);
            const uint32_t end = static_cast<uint32_t>(static_cast<uint64_t>(n) * (i + 1) / numParts);
            Node* node = new Node();
            node->points.reserve(end - begin);
            for (uint32_t j = begin; j < end; j++)
                node->points.push_back(MiniVertex{points[j]->id, {}});
            parts.push_back(node);
        }
        return parts;
    }

    // RobustPruning over ids, with each D(p, c) computed once. p's current edges are not
    // included; returns the kept ids, nearest first.
    std::vector<uint32_t> FlatTreeANN::pruneIds(uint32_t p, const std::vector<uint32_t>& candidates,
        float alpha, uint32_t R) const {
        const Vertex& pv = *global[p];
        std::vector<std::pair<float, uint32_t>> pool;
        pool.reserve(candidates.size());
        for (uint32_t c : candidates)
            if (c != p) pool.emplace_back(distance(pv, *global[c]), c);
        std::sort(pool.begin(), pool.end());

        std::vector<uint32_t> out;
        while (!pool.empty() && out.size() < R) {
            const uint32_t u = pool.front().second;
            out.push_back(u);
            std::vector<std::pair<float, uint32_t>> remaining;
            remaining.reserve(pool.size());
            for (size_t i = 1; i < pool.size(); i++)
                if (!(distance(*global[u], *global[pool[i].second]) * alpha <= pool[i].first))
                    remaining.push_back(pool[i]);
            pool = std::move(remaining);
        }
        return out;
    }

    // Parent positions of the half of child's points with the fewest out-neighbors
    // (ties random); parent[offset + i] is child.points[i]'s entry.
    static std::vector<size_t> leastConnectedHalf(const Node& child, size_t offset, std::mt19937& rng) {
        std::vector<size_t> idx(child.points.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng);  // random order among equal degrees
        std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
            return child.points[a].out.size() < child.points[b].out.size();
        });
        idx.resize((idx.size() + 1) / 2);
        for (size_t& i : idx) i += offset;
        return idx;
    }

    // New, separate parent level with empty out-lists; each side's least-connected
    // half is pruned against the other's. Caller owns it; L unused for now.
    std::unique_ptr<Node> FlatTreeANN::mergeNodetoNextLayer(Node& l, Node& r, uint32_t R,
        float alpha, uint32_t L, uint32_t seed) {
        std::mt19937 rng(seed);  // fixed seed for reproducible experiments
        auto parent = std::make_unique<Node>();
        parent->points.reserve(l.points.size() + r.points.size());
        for (const Node* child : {&l, &r})
            for (const MiniVertex& v : child->points)
                parent->points.push_back(MiniVertex{v.id, {}});
        parent->l_next = &l;
        parent->r_next = &r;

        // l's entries sit at [0, |l|), r's at [|l|, |l| + |r|)
        const std::vector<size_t> lHalf = leastConnectedHalf(l, 0, rng);
        const std::vector<size_t> rHalf = leastConnectedHalf(r, l.points.size(), rng);

        std::vector<uint32_t> lIds, rIds;
        for (size_t i : lHalf) lIds.push_back(parent->points[i].id);
        for (size_t i : rHalf) rIds.push_back(parent->points[i].id);

        for (size_t i : lHalf) parent->points[i].out = pruneIds(parent->points[i].id, rIds, alpha, R);
        for (size_t i : rHalf) parent->points[i].out = pruneIds(parent->points[i].id, lIds, alpha, R);

        return parent;
    }


}  // namespace FlatTree
