#include "../../DiskANN-Classic/DiskANN.h"
#include "../../Core/core_testing.h"
#include "../HardInstance.h"

int testHardInstance(){
    TestANN test;
    HardInstance inst = makeHardInstance(20000, 1000.0);
    if (!inst.ok || !checkSeparations(inst, 125)) { inst.freeAll(); return 1; }

    std::vector<Vertex*> knnResults = test.BruteForceKNN(inst.vertices, *inst.query, 10);
    std::cout << "Brute Force KNN\n";
    for (const auto& v : knnResults) {
        std::cout << v->id << " ";
    }

    std::mt19937 rng(42);
    FastGraphCreation(inst.vertices, 2.0f, 125, 70, rng);
    Vertex* s = Medoid(inst.vertices);
    std::vector<Vertex*> U = GreedySearch(*s, *inst.query, 125);
    

    std::cout << "Fast Disk ANN\n";
    for (const auto& v : U) {
        std::cout << v->id << " ";
    }

    float recall = test.RecallAtK(U, inst.answers);
    std::cout << "\nRecall@5 = " << recall << "\n";

    // This instance is built to TRAP fast preprocessing (see header comment):
    // Recall@5 should stay ~0 at buildL=125, since |P| = 0.1n = 2000 >> L.
    // A pass here means the trap still works, not that search succeeded.
    bool ok = true;
    const float maxRecall = 0.5f;
    if (recall >= maxRecall) {
        std::cerr << "Recall@5 (" << recall << ") at or above threshold ("
                   << maxRecall << ") -- hard instance no longer traps fast preprocessing\n";
        ok = false;
    }

    inst.freeAll();

    if (ok) std::cout << "testHardInstance PASSED\n";
    else std::cout << "testHardInstance FAILED\n";

    return ok ? 0 : 1;
}