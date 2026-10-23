#include "FlatTreeANN.h"

namespace FlatTree {

    // The tree only points at the caller's vertices; it doesn't own them.
    FlatTreeANN::~FlatTreeANN() {}

    void FlatTreeANN::MakeGraph(const std::vector<Vertex*>& vertices, uint32_t R, uint32_t k,
        float alpha, uint32_t L, uint32_t seed) {
        this->R = R;
        this->alpha = alpha;
        this->L = L;

        points = vertices;
        nodes.clear();
        leaves.clear();

        const uint32_t n = static_cast<uint32_t>(points.size());
        if (n == 0) return;
        k = std::clamp<uint32_t>(k, 1, n);  // no empty leaves

        std::mt19937 rng(seed);  // fixed seed for reproducible experiments
        nodes.reserve(2 * k - 1);  // a binary tree with k leaves has 2k - 1 nodes
        leaves.reserve(k);
        Split(0, n, k, rng);
    }

    int32_t FlatTreeANN::Split(uint32_t begin, uint32_t end, uint32_t numLeaves, std::mt19937& rng) {
        const int32_t id = static_cast<int32_t>(nodes.size());
        nodes.push_back({begin, end});

        if (numLeaves == 1) {
            leaves.push_back(static_cast<uint32_t>(id));
            return id;
        }

        // Random halves: shuffle this node's points, then cut the range. The
        // left child takes ceil(numLeaves / 2) leaves and a matching share of
        // the points, so leaf sizes stay balanced when k is not a power of 2.
        std::shuffle(points.begin() + begin, points.begin() + end, rng);
        const uint32_t leftLeaves = (numLeaves + 1) / 2;
        const uint32_t mid = begin + static_cast<uint32_t>(
            static_cast<uint64_t>(end - begin) * leftLeaves / numLeaves);

        // children are pushed after the parent (preorder), so set them via
        // index: push_back may have moved nodes
        const int32_t left = Split(begin, mid, leftLeaves, rng);
        const int32_t right = Split(mid, end, numLeaves - leftLeaves, rng);
        nodes[id].left = left;
        nodes[id].right = right;
        return id;
    }

    std::vector<Vertex*> FlatTreeANN::Partition(uint32_t i) const {
        const Node& leaf = nodes[leaves[i]];
        return std::vector<Vertex*>(points.begin() + leaf.begin, points.begin() + leaf.end);
    }

}  // namespace FlatTree
