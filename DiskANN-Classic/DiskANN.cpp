#include "DiskANN.h"

std::vector<Vertex*> GreedySearch(Vertex& s, Vertex& q, uint32_t L) {
    std::vector<Vertex*> A;
    std::vector<Vertex*> U;

    A.push_back(&s);

    while(true){
        //find closest v in A \ U 
        Vertex* closest = nullptr;
        float bestDist = 0.0f;

        for(Vertex* v : A){
            bool visited = std::any_of(U.begin(), U.end(), [&](Vertex* u) { return u == v; });
            if(visited) continue;

            float d = distance(*v, q);
            if(closest == nullptr || d < bestDist) {
                closest = v;
                bestDist = d;
            }
        }

        if(closest == nullptr){
            break;
        }
        
        //expand A to include neighbors of closest
        for(Vertex* neighbor : closest->out) {
            bool alreadyInA = std::any_of(A.begin(), A.end(), [&](Vertex* u) { return u == neighbor; });
            if(!alreadyInA) {
                A.push_back(neighbor);
            }
        }
        
        U.push_back(closest);

        //trim A to closest L vertices to q
        if(A.size() > L){
            std::sort(A.begin(), A.end(), [&](Vertex* a, Vertex* b) {
                return distance(*a, q) < distance(*b, q);
            });
            A.resize(L);
        }

        //sort visited by increasing dist to q
        std::sort(U.begin(), U.end(), [&](Vertex* a, Vertex* b) {
            return distance(*a, q) < distance(*b, q);
        });
    }
    return U;
}

void GraphCreation(std::vector<Vertex*>& vertices, float alpha, uint32_t R) {
    //get all next vertex for each vertex
    for(Vertex* v : vertices) {
        RobustPruning(*v, vertices, alpha, R);
    }
}

//only this -> slow preprocessing
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

void RobustPruningOcclude(Vertex& v, std::vector<Vertex*> U, float alpha, uint32_t R){
    U.insert(U.end(), v.out.begin(), v.out.end());
    U.erase(std::remove_if(U.begin(), U.end(),
        [&](Vertex* p) { return p->id == v.id; }), U.end());

    // pool must be deduped: a repeated pointer would be admitted twice
    std::sort(U.begin(), U.end());
    U.erase(std::unique(U.begin(), U.end()), U.end());

    v.out.clear();
    if (U.empty()) return;

    // cache D(v, w) and sort the pool by it, once
    std::vector<std::pair<float, Vertex*>> pool;
    pool.reserve(U.size());
    for (Vertex* w : U) pool.emplace_back(distance(*w, v), w);
    std::sort(pool.begin(), pool.end(),
        [](const std::pair<float, Vertex*>& a, const std::pair<float, Vertex*>& b) {
            return a.first < b.first;
        });

    std::vector<float> occlude(pool.size(), 0.0f);   // max over kept u of D(v,w)/D(u,w)
    std::vector<char>  kept(pool.size(), 0);

    for (float cur = 1.0f; cur <= alpha && v.out.size() < R; cur *= 1.2f) {
        for (size_t i = 0; i < pool.size() && v.out.size() < R; i++) {
            if (kept[i] || occlude[i] > cur) continue;

            kept[i] = 1;
            v.out.push_back(pool[i].second);

            for (size_t j = i + 1; j < pool.size(); j++) {
                if (kept[j]) continue;
                float duw = distance(*pool[i].second, *pool[j].second);
                if (duw <= 0.0f) {                    // coincident points
                    occlude[j] = std::numeric_limits<float>::max();
                    continue;
                }
                occlude[j] = std::max(occlude[j], pool[j].first / duw);
            }
        }
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


//fast preprocessing
void FastGraphCreation(std::vector<Vertex*>& v, float alpha, uint32_t L, uint32_t R, std::mt19937& gen){

    //initialize to random R graph 
    uint32_t n = v.size();

    for(Vertex* j : v) j->out.clear();
    if(n <= 1) return;

    uint32_t deg = std::min(R, n - 1);

    std::vector<uint32_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::uniform_int_distribution<uint32_t> pick(0, n - 1);

    
    //each round adds one permutation: +1 out-edge and +1 in-edge for every node
    for(uint32_t k = 0; k < deg; k++) {
        std::shuffle(perm.begin(), perm.end(), gen);

        //perm[i] must not be i (self-loop) and must not already be an out-neighbor
        auto bad = [&](uint32_t i) {
            if(perm[i] == i) return true;
            return std::any_of(v[i]->out.begin(), v[i]->out.end(),
                [&](Vertex* w) { return w == v[perm[i]]; });
        };

        //swapping two entries repairs a bad position and keeps perm a bijection
        bool dirty = true;
        while(dirty) {
            dirty = false;
            for(uint32_t i = 0; i < n; i++) {
                while(bad(i)) {
                    std::swap(perm[i], perm[pick(gen)]);
                    dirty = true;
                }
            }
        }

        for(uint32_t i = 0; i < n; i++) {
            v[i]->out.push_back(v[perm[i]]);
        }
    }

    std::cout << "Finished R regular graph construction\n";

    Vertex* s = Medoid(v);
    std::cout << "Found Medoid";

    std::vector<uint32_t> sigma(v.size());
    std::iota(sigma.begin(), sigma.end(), 0);

    for(float a : {1.0f, alpha}) {
        std::shuffle(sigma.begin(), sigma.end(), gen);
        for(int i = 0; i < v.size(); i++){
            std::vector<Vertex*> N_V = GreedySearch(*s, *v[sigma[i]], L);
            RobustPruning(*v[sigma[i]], N_V, a, R);
            for(Vertex* j : v[sigma[i]]->out){
                if(std::any_of(j->out.begin(), j->out.end(), [&](Vertex* w){ return w == v[sigma[i]]; })) continue;
                j->out.push_back(v[sigma[i]]);
                if(j->out.size()  > R)
                    RobustPruning(*j, j->out, a, R);
            }
        }
    }
}


float distance(const Vertex& u, const Vertex& v) {
  float dist = 0.0f;
  for (uint32_t i = 0; i < u.data.size(); ++i) {
    float diff = u.data[i] - v.data[i];
    dist += diff * diff;
  }
  return sqrt(dist);
}
