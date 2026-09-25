#include "testANNCentral.h"
// ===========================================================================
//  HARD INSTANCE for DiskANN with FAST preprocessing
//  (Indyk & Xu, NeurIPS 2023, section 4.1, adapted to this implementation)
//
//  Four groups of points in the plane, plus a query q that is NOT in the set.
//  Everything is a multiple of the scale S, so the shape is the same for any n.
//
//     q  = (0, 0)            the query
//     a  = (0, 0.80 S)       true nearest neighbour, + 4 duplicates (Recall@5)
//     P  = 0.1n points       THE TRAP.  Thin strip, x in [-1.01,-1.00] S,
//                            y in [0, 0.80] S.  Nearest cluster to q.
//     P' = 0.1n points       THE SHIELD.  0.1S square just above a,
//                            y in [1.50, 1.60] S.  Nearest cluster to a.
//     M  = 0.8n points       THE BALLAST.  0.3S square, bottom-right corner
//                            at (-2.5, 2.5) S.  Holds the medoid / start vertex.
//
//  Distance ordering from q:   a  <  P  <  P'  <  M
//  Distance ordering from a:   P' <  P  <  M
//
//  Why fast preprocessing fails on it:
//   * The start vertex s is the medoid, which lies in M (80% of the mass).
//   * From s, greedy search takes the closest candidate to q: that is P, and
//     EVERY vertex of P is closer to q than EVERY vertex of P' and M, so once
//     the search is in P it cannot leave until P is exhausted.
//   * Fast preprocessing prunes each vertex against GreedySearch(s, v, L),
//     i.e. against ~L vertices near v.  For v in P that pool is other P
//     vertices, so no edge P -> a is ever created.  Edges into a are made only
//     as back-edges when a is inserted, and the search that inserts a lands in
//     P' (nearest cluster to a), so those back-edges all go to P'.
//   * Result: GreedySearch(s, q, L) scans s plus ~L vertices of P and stops.
//     Recall@5 = 0 until L is around |P| = 0.1n.
//
//  Two deviations from the paper's figure, both needed because this
//  implementation prunes with a single alpha instead of DiskANN's alpha ramp:
//   * M sits at 2.5S instead of 1.2S.  The alpha=2 rule only removes a from an
//     M vertex's neighbour list when min D(a,M) > 2 * max D(a,P'); at 1.2S that
//     fails and the start vertex keeps a direct edge to a, ending the search in
//     one step.
//   * P is a thin strip with a level with its top end, instead of a square.
//     The random R-regular initialisation gives ~R*|P|/n vertices of P an edge
//     into the answer cluster; the ones that survive pruning are the ones
//     nearest a.  In a square those are also the ones nearest q, i.e. the first
//     ones scanned, and the trap leaks.  Along the strip the distance to q
//     increases while the distance to a decreases, so those vertices are
//     scanned LAST.  (The paper uses a chain for P in its Figure 7 instance.)
// ===========================================================================

struct HardInstance {
    std::vector<Vertex*> vertices;   // the database, ids 0..n-1, blocked M | P | P' | a
    Vertex* query = nullptr;         // q, deliberately NOT part of vertices
    std::vector<Vertex*> answers;    // a + 4 duplicates = ground-truth top 5
    uint32_t mCluster = 0, pCluster = 0, pPrimeCluster = 0;
    bool ok = false;                 // false if n was too small to build

    // no copying: the struct owns raw pointers, a copy would double free
    HardInstance() = default;
    HardInstance(const HardInstance&) = delete;
    HardInstance& operator=(const HardInstance&) = delete;
    HardInstance(HardInstance&&) = default;
    HardInstance& operator=(HardInstance&&) = default;

    void freeAll() {
        for (Vertex* v : vertices) delete v;
        vertices.clear();
        answers.clear();
        delete query;
        query = nullptr;
        ok = false;
    }
};

HardInstance makeHardInstance(uint32_t n, double S, double cM = 2.5){
    HardInstance inst;
    const uint32_t trueNNCount = 5;

    const uint32_t cols = 12;
    uint32_t rows = std::max<uint32_t>(2u, uint32_t(0.1 * n) / cols);
    uint32_t pCluster = cols * rows;

    uint32_t sidePprime = uint32_t(std::lround(std::sqrt(0.1 * double(n))));
    if(sidePprime < 2) sidePprime = 2;
    uint32_t pPrimeCluster = sidePprime * sidePprime;

    if(pCluster + pPrimeCluster + trueNNCount >= n){
        std::cerr << "Hard instance is too small\n";
        return inst;                       // inst.ok stays false
    }


    uint32_t mCluster = n - pCluster - pPrimeCluster - trueNNCount;
    const uint32_t mGridLen = uint32_t(std::ceil(std::sqrt(double(mCluster))));

    std::vector<Vertex*>& vertices = inst.vertices;
    vertices.reserve(n);
    uint32_t id = 0;

    //create M central cluster
    const double mSideGeometry = 0.3 * S, mstep = mSideGeometry / double(mGridLen - 1);
    const double mx = -cM * S, my = cM * S;
    for(uint32_t k = 0; k < mCluster; k++){
        Vertex* v = new Vertex();
        v->id = id++;
        v->data.push_back(static_cast<float>(mx - double(k % mGridLen) * mstep));
        v->data.push_back(static_cast<float>(my + double(k / mGridLen) * mstep));
        vertices.push_back(v);
    }

    //P -> diversion cluster bottom end nearest to Q but top end nearest to A wihtin cluster not overall
    const double pxStep = (0.01 * S) / double(cols - 1);
    const double pyStep = (0.80 * S) / double(rows - 1);
    for(uint32_t k = 0; k <  pCluster; k++){
        Vertex* v = new Vertex();
        v->id = id++;
        v->data.push_back(static_cast<float>(-1.00 * S - double(k % cols) * pxStep));
        v->data.push_back(static_cast<float>(0.0 + double(k / cols) * pyStep));
        vertices.push_back(v);
    }

    // P' -> true path to NN values, layered directly on top of it
    const double pPrimeStep = (0.10 * S) / double(sidePprime - 1);
    for(uint32_t k = 0; k < pPrimeCluster; k++){
        Vertex* v = new Vertex();
        v->id = id++;
        v->data.push_back(static_cast<float>(-0.05 * S + double(k % sidePprime) * pPrimeStep));
        v->data.push_back(static_cast<float>(1.50 * S + double(k / sidePprime) * pPrimeStep));
        vertices.push_back(v);
    }

    // ---- a and its 4 duplicates (so Recall@5 is well defined)
    const double ay = 0.80 * S, e = 1e-4 * S;
    const double off[5][2] = {{0, 0}, {e, 0}, {-e, 0}, {0, e}, {0, -e}};
    for (int k = 0; k < 5; k++) {
        Vertex* v = new Vertex();
        v->id = id++;
        v->data.push_back(static_cast<float>(off[k][0]));
        v->data.push_back(static_cast<float>(ay + off[k][1]));
        vertices.push_back(v);
        inst.answers.push_back(v);
    }

    Vertex* query = new Vertex();
    query->id = id;
    query->data.push_back(0.0f);
    query->data.push_back(0.0f);

    inst.query         = query;
    inst.mCluster      = mCluster;
    inst.pCluster      = pCluster;
    inst.pPrimeCluster = pPrimeCluster;
    inst.ok            = true;
    return inst;
}




bool checkSeparations(const std::vector<Vertex*>& vertices, Vertex* query,
                      const std::vector<Vertex*>& answers,
                      uint32_t mCluster, uint32_t pCluster, uint32_t pPrimeCluster,
                      uint32_t buildL = 125) {
    const uint32_t pStart  = mCluster;
    const uint32_t ppStart = mCluster + pCluster;
    const uint32_t aStart  = mCluster + pCluster + pPrimeCluster;

    if (vertices.size() != aStart + answers.size()) {
        std::cerr << "size mismatch: vertices=" << vertices.size()
                  << " expected=" << aStart + answers.size() << "\n";
        return false;
    }

    float qa = 1e30f;
    float qPmin = 1e30f, qPmax = 0.0f, qPpmin = 1e30f, qMmin = 1e30f;
    float aPmin = 1e30f, aPpmax = 0.0f, aMmin = 1e30f;

    Vertex* a = answers[0];
    for (Vertex* v : vertices) {
        if (!v) { std::cerr << "null vertex in list\n"; return false; }
        float dq = distance(*v, *query), da = distance(*v, *a);
        if (v->id < pStart)       { qMmin  = std::min(qMmin, dq);  aMmin = std::min(aMmin, da); }
        else if (v->id < ppStart) { qPmin  = std::min(qPmin, dq);  qPmax = std::max(qPmax, dq);
                                    aPmin  = std::min(aPmin, da); }
        else if (v->id < aStart)  { qPpmin = std::min(qPpmin, dq); aPpmax = std::max(aPpmax, da); }
        else                      { qa     = std::min(qa, dq); }
    }

    struct Cond { const char* name; float lhs, rhs; };
    const Cond conds[] = {
        {"D(q,a) < min D(q,P)        [a is the true NN]",        qa,         qPmin },
        {"max D(q,P) < min D(q,P')   [P scanned before P']",     qPmax,      qPpmin},
        {"max D(q,P) < min D(q,M)    [search leaves M for P]",   qPmax,      qMmin },
        {"max D(a,P') < min D(a,P)   [P stays out of a's pool]", aPpmax,     aPmin },
        {"2*max D(a,P') < min D(a,M) [alpha=2 prunes a from M]", 2 * aPpmax, aMmin },
    };

    bool ok = true;
    for (const Cond& c : conds) {
        bool pass = c.lhs < c.rhs;
        ok &= pass;
        printf("   %-48s %9.1f < %9.1f   %s (margin %+.1f%%)\n",
               c.name, c.lhs, c.rhs, pass ? "ok  " : "FAIL",
               100.0 * (c.rhs - c.lhs) / c.lhs);
    }

    bool fill = pPrimeCluster >= buildL;
    ok &= fill;
    printf("   %-48s %9u >= %8u   %s\n",
           "|P'| >= build L            [P' fills a's pool]", pPrimeCluster, buildL,
           fill ? "ok  " : "FAIL");
    return ok;
}

// convenience wrapper: build and verify in one call
bool checkSeparations(const HardInstance& inst, uint32_t buildL = 125) {
    if (!inst.ok) { std::cerr << "instance was not built\n"; return false; }
    return checkSeparations(inst.vertices, inst.query, inst.answers,
                            inst.mCluster, inst.pCluster, inst.pPrimeCluster, buildL);
}



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

    // score Recall@5 against inst.answers; bucket U by the id ranges
    inst.freeAll();

    return 0;
}