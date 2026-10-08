#include "clusterANN.h"

// Boundary-node selection for the backbone graph over the disjoint clusters.

namespace ClusterANN {

    void findGlobalBoundaryNodes(const ClusterGraph& g, std::vector<Vertex*>& boundaryNodes, double proportion) {
        boundaryNodes.clear();
        if(proportion <= 0.0 || proportion > 1.0) return;
        using score = std::pair<float, Vertex*>;
        std::vector<score> scores;

        for(const Cluster& c : g.clusters){ 
            const std::vector<Vertex*>& pts = c.clusterPoints;
            if(pts.empty()) continue;
            scores.clear();
            scores.reserve(pts.size());
            for(Vertex* x : pts){
                const size_t dim = x->data.size();
                std::vector<float> mean(dim, 0.0f);
                uint32_t k = 0;
                for(Vertex* y : x->out){
                    float d = distance(*x, *y);
                    if(d <= 0.0f) continue; //no direction (coincident)
                    for(size_t i = 0; i < dim; ++i) mean[i] += (y->data[i] - x->data[i]) / d;
                    ++k;
                }

                float score = 1.0f; //default to 1 if no neighbors
                if(k > 0){
                    double sq = 0.0;
                    for(double m : mean) sq += (m/k) * (m/k);
                    score = static_cast<float>(std::sqrt(sq));
                }
                scores.push_back({score, x});
            }

            //pick proportion of clsuter by score 
            size_t pointsToPick = static_cast<size_t>(std::round(proportion * pts.size()));
            pointsToPick = std::min(std::max<size_t>(pointsToPick,  1), pts.size());
            std::partial_sort(scores.begin(), scores.begin() + pointsToPick, scores.end(),
                [](const score& a, const score& b){return a.first > b.first;});
            for(size_t i = 0; i < pointsToPick; ++i) boundaryNodes.push_back(scores[i].second);
        }
    }
}  // namespace ClusterANN
