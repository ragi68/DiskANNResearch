#include "core.h"

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


float distance(const Vertex& u, const Vertex& v) {
  float dist = 0.0f;
  for (uint32_t i = 0; i < u.data.size(); ++i) {
    float diff = u.data[i] - v.data[i];
    dist += diff * diff;
  }
  return sqrt(dist);
}
