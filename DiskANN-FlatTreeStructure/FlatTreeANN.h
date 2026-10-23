#pragma once
#include "../Core/core.h"
#include "../Core/commonstructs.h"


namespace FlatTree{
    // One node of the flattened tree. Every node owns the contiguous range
    // [begin, end) of FlatTreeANN::points; a node's two children split that
    // range, so each leaf is one partition.
    struct Node {
        uint32_t begin, end;
        int32_t left = -1, right = -1;  // indices into nodes, -1 for a leaf
    };

    class FlatTreeANN{
        public:

            FlatTreeANN() = default;
            FlatTreeANN(const std::vector<Vertex*>& vertices,
                const uint32_t& alpha, const uint32_t& R, const uint32_t& L,
                const uint32_t& edgesPerVertex, const uint32_t& poolSize);
            ~FlatTreeANN();
            void search(const std::vector<float>& query, const uint32_t& k, std::vector<std::pair<uint32_t,float>>& result);

            // Divide and conquer into k partitions: the root holds every point,
            // and each node shuffles its points and hands its first half to the
            // left child and the rest to the right, until there are k leaves.
            // Leaves are balanced (sizes differ by at most one) and the points
            // are distinct, but the split ignores geometry, so partitions
            // overlap in space. k is clamped to [1, n]. Partition only for now:
            // R, alpha and L are stored for the graph build that comes next.
            void MakeGraph(const std::vector<Vertex*>& vertices, uint32_t R, uint32_t k,
                float alpha, uint32_t L, uint32_t seed = 12345);

            uint32_t numPartitions() const { return static_cast<uint32_t>(leaves.size()); }
            std::vector<Vertex*> Partition(uint32_t i) const;

            // Tree stored flat in preorder: nodes[0] is the root, and leaves
            // lists the leaf node indices left to right.
            std::vector<Vertex*> points;
            std::vector<Node> nodes;
            std::vector<uint32_t> leaves;

        private:
            // Splits the points [begin, end) into numLeaves leaves under a new
            // node; returns that node's index.
            int32_t Split(uint32_t begin, uint32_t end, uint32_t numLeaves, std::mt19937& rng);
            
            float alpha = 1.2f;
            uint32_t R = 0, L = 0;

            
    };
}
