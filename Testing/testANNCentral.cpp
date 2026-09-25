#include "testANNCentral.h"



    


//brute force knn search for ground truth
    std::vector<Vertex*> TestANN::BruteForceKNN(const std::vector<Vertex*>& vertices, Vertex q, 
        uint32_t k) {
        std::vector<Vertex*> sorted = vertices;
        std::sort(sorted.begin(), sorted.end(), [&](Vertex* a, Vertex* b) {
            return distance(*a, q) < distance(*b, q);
        });
        if (sorted.size() > k) sorted.resize(k);
        return sorted;
    }


//generate random vertices
std::vector<Vertex*> TestANN::RandomVertices(uint32_t n, uint32_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(0.0f, 1000.0f);
    std::vector<Vertex*> vertices;
    for (uint32_t i = 0; i < n; i++) {
        Vertex* v = new Vertex();
        v->id = i;
        for (uint32_t j = 0; j < dim; j++) v->data.push_back(dist(rng));
        vertices.push_back(v);
    }
    return vertices;
}




