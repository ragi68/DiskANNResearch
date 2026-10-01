#include "TestSlowInstance.cpp"
#include "TestFastInstance.cpp"
#include "TestHardInstance.cpp"
#include "TestClusterInstance.cpp"
#include <cstring>
#include <iostream>

// Usage:
//   DiskANNTests               runs slow, fast, and hard instance tests
//   DiskANNTests -cluster      runs only the cluster ANN instance test
//   DiskANNTests -slow -fast   runs only the named tests (any combination of
//                              -slow -fast -hard -cluster)
int main(int argc, char** argv){
    bool runSlow = false, runFast = false, runHard = false, runCluster = false;
    bool any = false;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "-slow") == 0)         { runSlow = true;    any = true; }
        else if (std::strcmp(argv[i], "-fast") == 0)    { runFast = true;    any = true; }
        else if (std::strcmp(argv[i], "-hard") == 0)    { runHard = true;    any = true; }
        else if (std::strcmp(argv[i], "-cluster") == 0) { runCluster = true; any = true; }
        else std::cerr << "unrecognized argument: " << argv[i] << "\n";
    }

    // no flags given -> run the default suite (slow, fast, hard)
    if (!any) { runSlow = runFast = runHard = true; }

    int result = 0;
    if (runSlow)    result |= testSlow();
    if (runFast)    result |= testFast();
    if (runHard)    result |= testHardInstance();
    if (runCluster) result |= testClusterInstance();

    return result;
}
