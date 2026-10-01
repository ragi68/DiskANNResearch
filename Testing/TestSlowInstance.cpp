#include "testANNCentral.h"

int testSlow(){
    TestANN test;

    std::mt19937 rng(42); // Fixed seed for reproducibility
    uint32_t numVertices = 20000;
    uint32_t dim = 128;
    uint32_t k = 10;

    // Generate random vertices
    std::vector<Vertex*> vertices = test.RandomVertices(numVertices, dim, rng);

    Vertex q;
    q.id = numVertices; // Assign an ID outside the range of generated vertices
    for (uint32_t j = 0; j < dim; j++) q.data.push_back(std::uniform_real_distribution<float>(0.0f, 1000.0f)(rng));

    std::vector<Vertex*> knnResults = test.BruteForceKNN(vertices, q, k);
    std::cout << "Brute Force KNN\n";
    for (const auto& v : knnResults) {
        std::cout << v->id << " ";
    }

    std::cout << "Slow Preprocessing";
    GraphCreation(vertices, 1.0, vertices.size());
    std::vector<Vertex*> results = GreedySearch(*Medoid(vertices), q, 125);
    std::cout << "Slow Disk ANN\n";
    for (const auto& v : results) {
        std::cout << v->id << " ";
    }

    float recall = test.RecallAtK(results, knnResults);
    std::cout << "\nRecall@" << k << " = " << recall << "\n";

    bool ok = true;
    const float minRecall = 0.5f;
    if (recall < minRecall) {
        std::cerr << "Recall@" << k << " (" << recall << ") below threshold ("
                   << minRecall << ")\n";
        ok = false;
    }

    for (auto& v : vertices) {
        delete v;
    }

    if (ok) std::cout << "testSlow PASSED\n";
    else std::cout << "testSlow FAILED\n";

    return ok ? 0 : 1;
}
