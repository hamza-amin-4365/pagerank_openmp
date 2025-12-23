#include <omp.h>
#include <assert.h>

typedef struct {
    int* outLinks;
    int numOutLinks;
    int capacity;
} Node;

typedef struct {
    int numNodes;
    int numEdges;
    Node* nodes;
    int** inEdges;
    int* inDegree;
    int* inCapacity;
} Graph;

typedef struct {
    int numThreads;
    double sequentialTime;
    double parallelTime;
    double speedup;
    double efficiency;
    int iterations;
    double finalDiff;
} BenchmarkResult;

