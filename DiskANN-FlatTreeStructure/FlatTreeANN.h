#pragma once
#include "../Core/core.h"
#include "../Core/commonstructs.h"
#include <memory>


namespace FlatTree{
    // A point's entry in one level: just its id and that level's out-edges (as ids).
    struct MiniVertex {
        uint32_t id;
        std::vector<uint32_t> out;
    };

    // One tree node = one separate level; coordinates live in FlatTreeANN::global.
    struct Node {
        std::vector<MiniVertex> points;
        Node* l_next = nullptr;
        Node* r_next = nullptr;
    };

    class FlatTreeANN{
        public:

            FlatTreeANN() = default;
            ~FlatTreeANN() = default;


            void search(const std::vector<float>& query, const uint32_t& k, std::vector<std::pair<uint32_t,float>>& result);

            // Divide in partition count and then in a tree like fashion marge partitions
            // by taking half (random/geometric) and building a 'flattened tree'
            void MakeGraph(const std::vector<Vertex*>& vertices, uint32_t R, uint32_t k,
                float alpha, uint32_t L, uint32_t seed = 12345);

        private:
            // global[id] -> the caller's vertex (ids must be dense, 0..n-1); not owned.
            std::vector<Vertex*> global;

            std::vector<Node*> makePartitions(std::vector<Vertex*> points, uint32_t numParts,
                uint32_t seed = 12345);
            std::unique_ptr<Node> mergeNodetoNextLayer(Node& l, Node& r, uint32_t R, float alpha,
                uint32_t L, uint32_t seed);
            Node mergeLevels(Node& top, Node& l, Node& r, uint32_t R, float alpha, uint32_t L, uint32_t seed);

            std::vector<uint32_t> pruneIds(uint32_t p, const std::vector<uint32_t>& candidates,
                float alpha, uint32_t R) const;
    };
}
