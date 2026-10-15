#include "ClassicTesting/TestSlowInstance.cpp"
#include "ClassicTesting/TestFastInstance.cpp"
#include "ClassicTesting/TestHardInstance.cpp"
#include "ClusterTesting/TestClusterInstance.cpp"
#include "ClusterTesting/ClusterSweepTest.cpp"
#include "PatchTesting/PatchSweepTest.cpp"
#include <cstring>
#include <iostream>

// Usage:
//   DiskANNTests               runs slow, fast, and hard instance tests
//   DiskANNTests -cluster      runs only the cluster ANN instance test
//   DiskANNTests -sweep        runs only the backbone boundaryProportion sweep
//   DiskANNTests -patch        runs only the patch graph membership-cap sweep
//   DiskANNTests -slow -fast   runs only the named tests (any combination of
//                              -slow -fast -hard -cluster -sweep -patch)
int main(int argc, char** argv){
    bool runSlow = false, runFast = false, runHard = false, runCluster = false, runSweep = false, runPatch = false;
    bool any = false;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "-slow") == 0)         { runSlow = true;    any = true; }
        else if (std::strcmp(argv[i], "-fast") == 0)    { runFast = true;    any = true; }
        else if (std::strcmp(argv[i], "-hard") == 0)    { runHard = true;    any = true; }
        else if (std::strcmp(argv[i], "-cluster") == 0) { runCluster = true; any = true; }
        else if (std::strcmp(argv[i], "-sweep") == 0)   { runSweep = true;   any = true; }
        else if (std::strcmp(argv[i], "-patch") == 0)   { runPatch = true;   any = true; }
        else std::cerr << "unrecognized argument: " << argv[i] << "\n";
    }

    // no flags given -> run the default suite (slow, fast, hard)
    if (!any) { runSlow = runFast = runHard = true; }

    int result = 0;
    if (runSlow)    result |= testSlow();
    if (runFast)    result |= testFast();
    if (runHard)    result |= testHardInstance();
    if (runCluster) result |= testClusterInstance();
    if (runSweep)   result |= testClusterSweep();
    if (runPatch)   result |= testPatchSweep();

    return result;
}
