#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <time.h>

/* ============================================================================
   CONFIGURATION
   ============================================================================ */

#define INITIAL_CAPACITY 10
#define MAX_NODES 100000000
#define VERBOSE 0

/* ============================================================================
   DATA STRUCTURES
   ============================================================================ */

typedef struct {
    int* outLinks;
    int numOutLinks;
    int capacity;
} Node;

typedef struct {
    int numNodes;
    long numEdges;
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
} BenchmarkResult;

/* ============================================================================
   GRAPH LOADING
   ============================================================================ */

Graph* createGraph(int numNodes) {
    if (numNodes <= 0 || numNodes > MAX_NODES) {
        fprintf(stderr, "Error: Invalid number of nodes: %d\n", numNodes);
        return NULL;
    }
    
    Graph* g = (Graph*)malloc(sizeof(Graph));
    if (!g) {
        fprintf(stderr, "Error: Failed to allocate graph structure\n");
        return NULL;
    }
    
    g->numNodes = numNodes;
    g->numEdges = 0;
    
    g->nodes = (Node*)calloc(numNodes, sizeof(Node));
    g->inEdges = (int**)calloc(numNodes, sizeof(int*));
    g->inDegree = (int*)calloc(numNodes, sizeof(int));
    g->inCapacity = (int*)calloc(numNodes, sizeof(int));
    
    if (!g->nodes || !g->inEdges || !g->inDegree || !g->inCapacity) {
        fprintf(stderr, "Error: Failed to allocate graph arrays\n");
        free(g->nodes);
        free(g->inEdges);
        free(g->inDegree);
        free(g->inCapacity);
        free(g);
        return NULL;
    }
    
    for (int i = 0; i < numNodes; i++) {
        g->nodes[i].outLinks = (int*)malloc(INITIAL_CAPACITY * sizeof(int));
        g->inEdges[i] = (int*)malloc(INITIAL_CAPACITY * sizeof(int));
        
        if (!g->nodes[i].outLinks || !g->inEdges[i]) {
            fprintf(stderr, "Error: Failed to allocate node %d\n", i);
            for (int j = 0; j < i; j++) {
                free(g->nodes[j].outLinks);
                free(g->inEdges[j]);
            }
            free(g->nodes);
            free(g->inEdges);
            free(g->inDegree);
            free(g->inCapacity);
            free(g);
            return NULL;
        }
        
        g->nodes[i].capacity = INITIAL_CAPACITY;
        g->nodes[i].numOutLinks = 0;
        g->inDegree[i] = 0;
        g->inCapacity[i] = INITIAL_CAPACITY;
    }
    
    return g;
}

void addEdge(Graph* g, int src, int dst) {
    if (src < 0 || src >= g->numNodes || dst < 0 || dst >= g->numNodes) {
        return;
    }
    
    if (src == dst) {
        return;
    }
    
    // Add to outgoing
    if (g->nodes[src].numOutLinks == g->nodes[src].capacity) {
        int newCap = g->nodes[src].capacity * 2;
        int* newLinks = (int*)realloc(g->nodes[src].outLinks, newCap * sizeof(int));
        if (!newLinks) return;
        g->nodes[src].outLinks = newLinks;
        g->nodes[src].capacity = newCap;
    }
    g->nodes[src].outLinks[g->nodes[src].numOutLinks++] = dst;
    
    // Add to incoming
    if (g->inDegree[dst] == g->inCapacity[dst]) {
        int newCap = g->inCapacity[dst] * 2;
        int* newEdges = (int*)realloc(g->inEdges[dst], newCap * sizeof(int));
        if (!newEdges) return;
        g->inEdges[dst] = newEdges;
        g->inCapacity[dst] = newCap;
    }
    g->inEdges[dst][g->inDegree[dst]++] = src;
    
    g->numEdges++;
}

Graph* loadGraphFromFile(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open %s\n", filename);
        return NULL;
    }
    
    printf("Loading graph from: %s\n", filename);
    
    int maxNode = -1;
    int src, dst;
    long edgeCount = 0;
    
    // First pass
    while (fscanf(file, "%d %d", &src, &dst) == 2) {
        if (src >= 0 && dst >= 0) {
            edgeCount++;
            if (src > maxNode) maxNode = src;
            if (dst > maxNode) maxNode = dst;
        }
    }
    
    if (maxNode < 0) {
        fprintf(stderr, "Error: No valid edges found\n");
        fclose(file);
        return NULL;
    }
    
    int numNodes = maxNode + 1;
    printf("Creating graph: %d nodes, %ld edges\n", numNodes, edgeCount);
    
    Graph* g = createGraph(numNodes);
    if (!g) {
        fclose(file);
        return NULL;
    }
    
    // Second pass
    rewind(file);
    long added = 0;
    
    while (fscanf(file, "%d %d", &src, &dst) == 2) {
        if (src >= 0 && dst >= 0 && src != dst) {
            addEdge(g, src, dst);
            added++;
            if (added % 1000000 == 0) {
                printf("  Loaded %ld edges...\n", added);
            }
        }
    }
    
    fclose(file);
    printf("Loaded: %d nodes, %ld edges\n", g->numNodes, g->numEdges);
    
    return g;
}

void freeGraph(Graph* g) {
    if (!g) return;
    
    if (g->nodes) {
        for (int i = 0; i < g->numNodes; i++) {
            free(g->nodes[i].outLinks);
        }
        free(g->nodes);
    }
    
    if (g->inEdges) {
        for (int i = 0; i < g->numNodes; i++) {
            free(g->inEdges[i]);
        }
        free(g->inEdges);
    }
    
    free(g->inDegree);
    free(g->inCapacity);
    free(g);
}

/* ============================================================================
   PAGERANK - SEQUENTIAL
   ============================================================================ */

void pageRankSequential(Graph* g, double* pageRank, 
                       double damping, double threshold, int maxIters) {
    int n = g->numNodes;
    // Keep track of the pointer WE allocated
    double* localAlloc = (double*)malloc(n * sizeof(double));
    if (!localAlloc) return;

    double* newPageRank = localAlloc;
    double* originalPtr = pageRank; // Keep track of the pointer passed by caller
    
    double base = (1.0 - damping) / n;
    
    // Initialize
    for (int i = 0; i < n; i++) {
        pageRank[i] = 1.0 / n;
    }
    
    // Iterate
    for (int iter = 0; iter < maxIters; iter++) {
        for (int i = 0; i < n; i++) {
            newPageRank[i] = base;
        }
        
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < g->inDegree[i]; j++) {
                int src = g->inEdges[i][j];
                int outDeg = g->nodes[src].numOutLinks;
                if (outDeg > 0) {
                    sum += pageRank[src] / outDeg;
                }
            }
            newPageRank[i] += damping * sum;
        }
        
        double diff = 0.0;
        for (int i = 0; i < n; i++) {
            diff += fabs(newPageRank[i] - pageRank[i]);
        }
        
        // Swap pointers
        double* tmp = pageRank;
        pageRank = newPageRank;
        newPageRank = tmp;
        
        if (iter % 5 == 0) {
            printf("  Iter %d: diff = %.2e\n", iter, diff);
        }
        
        if (diff < threshold) {
            printf("  Converged at iteration %d\n", iter);
            break;
        }
    }
    
    // Restore results to original pointer if necessary
    if (pageRank != originalPtr) {
        memcpy(originalPtr, pageRank, n * sizeof(double));
    }

    // Always free the memory WE allocated, not the one passed to us
    free(localAlloc);
}

/* ============================================================================
   PAGERANK - PARALLEL
   ============================================================================ */

void pageRankParallel(Graph* g, double* pageRank, int numThreads,
                     double damping, double threshold, int maxIters) {
    int n = g->numNodes;
    
    // Track local allocation
    double* localAlloc = (double*)malloc(n * sizeof(double));
    if (!localAlloc) return;
    
    double* newPageRank = localAlloc;
    double* originalPtr = pageRank;

    double base = (1.0 - damping) / n;
    
    omp_set_num_threads(numThreads);
    
    // Initialize
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        pageRank[i] = 1.0 / n;
    }
    
    // Iterate
    for (int iter = 0; iter < maxIters; iter++) {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            newPageRank[i] = base;
        }
        
        #pragma omp parallel for schedule(dynamic, 128)
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < g->inDegree[i]; j++) {
                int src = g->inEdges[i][j];
                int outDeg = g->nodes[src].numOutLinks;
                if (outDeg > 0) {
                    sum += pageRank[src] / outDeg;
                }
            }
            newPageRank[i] += damping * sum;
        }
        
        double diff = 0.0;
        #pragma omp parallel for reduction(+:diff)
        for (int i = 0; i < n; i++) {
            diff += fabs(newPageRank[i] - pageRank[i]);
        }
        
        // Swap
        double* tmp = pageRank;
        pageRank = newPageRank;
        newPageRank = tmp;
        
        if (iter % 5 == 0) {
            printf("  Iter %d: diff = %.2e\n", iter, diff);
        }
        
        if (diff < threshold) {
            printf("  Converged at iteration %d\n", iter);
            break;
        }
    }
    
    // Restore and Free
    if (pageRank != originalPtr) {
        #pragma omp parallel for
        for(int i=0; i<n; i++) originalPtr[i] = pageRank[i];
    }
    
    free(localAlloc);
}

/* ============================================================================
   MAIN
   ============================================================================ */

int main(int argc, char** argv) {
    printf("================================================================================\n");
    printf("PARALLEL PAGERANK WITH OPENMP\n");
    printf("================================================================================\n\n");
    
    if (argc < 2) {
        printf("Usage: %s <graph_file>\n", argv[0]);
        return 1;
    }
    
    Graph* g = loadGraphFromFile(argv[1]);
    if (!g) {
        fprintf(stderr, "Failed to load graph\n");
        return 1;
    }
    
    printf("\nGraph loaded successfully\n");
    printf("Nodes: %d, Edges: %ld\n\n", g->numNodes, g->numEdges);
    
    int maxThreads = omp_get_max_threads();
    printf("Maximum threads available: %d\n\n", maxThreads);
    
    // Prepare thread counts
    int threadCounts[6] = {1, 2, 4, 8, 16, 32};
    int numTests = 0;
    BenchmarkResult results[6];
    
    for (int i = 0; i < 6; i++) {
        if (threadCounts[i] <= maxThreads) {
            numTests++;
        }
    }
    
    double damping = 0.85;
    double threshold = 1e-6;
    int maxIters = 100;
    
    // Run sequential once
    printf("================================================================================\n");
    printf("SEQUENTIAL BENCHMARK\n");
    printf("================================================================================\n\n");
    
    double* pageRankSeq = (double*)calloc(g->numNodes, sizeof(double));
    if (!pageRankSeq) {
        fprintf(stderr, "Failed to allocate memory\n");
        freeGraph(g);
        return 1;
    }
    
    double seqStart = omp_get_wtime();
    pageRankSequential(g, pageRankSeq, damping, threshold, maxIters);
    double seqEnd = omp_get_wtime();
    double seqTime = seqEnd - seqStart;
    
    printf("\nSequential time: %.4f seconds\n", seqTime);
    
    free(pageRankSeq);
    
    // Run parallel benchmarks
    printf("\n================================================================================\n");
    printf("PARALLEL BENCHMARKS\n");
    printf("================================================================================\n\n");
    
    for (int i = 0; i < numTests; i++) {
        int threads = threadCounts[i];
        printf(">>> Test %d/%d: %d threads\n\n", i+1, numTests, threads);
        
        double* pageRankPar = (double*)calloc(g->numNodes, sizeof(double));
        if (!pageRankPar) {
            fprintf(stderr, "Failed to allocate memory for threads=%d\n", threads);
            continue;
        }
        
        double parStart = omp_get_wtime();
        pageRankParallel(g, pageRankPar, threads, damping, threshold, maxIters);
        double parEnd = omp_get_wtime();
        double parTime = parEnd - parStart;
        
        results[i].numThreads = threads;
        results[i].sequentialTime = seqTime;
        results[i].parallelTime = parTime;
        results[i].speedup = seqTime / parTime;
        results[i].efficiency = (results[i].speedup / threads) * 100.0;
        
        printf("Parallel time: %.4f seconds\n", parTime);
        printf("Speedup: %.2fx\n", results[i].speedup);
        printf("Efficiency: %.2f%%\n\n", results[i].efficiency);
        
        free(pageRankPar);
    }
    
    // Print summary
    printf("================================================================================\n");
    printf("SUMMARY\n");
    printf("================================================================================\n\n");
    
    printf("Threads | Time (s)  | Speedup | Efficiency\n");
    printf("--------|-----------|---------|----------\n");
    printf("   1    | %.4f    | 1.00x   | 100.00%%\n", seqTime);
    
    for (int i = 0; i < numTests; i++) {
        printf("  %2d    | %.4f    | %.2fx    | %.2f%%\n",
               results[i].numThreads,
               results[i].parallelTime,
               results[i].speedup,
               results[i].efficiency);
    }
    
    printf("\n");
    
    // Save CSV
    FILE* csv = fopen("pagerank_results.csv", "w");
    if (csv) {
        fprintf(csv, "Threads,SequentialTime,ParallelTime,Speedup,Efficiency\n");
        fprintf(csv, "1,%.4f,%.4f,1.00,100.00\n", seqTime, seqTime);
        
        for (int i = 0; i < numTests; i++) {
            fprintf(csv, "%d,%.4f,%.4f,%.4f,%.2f\n",
                    results[i].numThreads,
                    results[i].sequentialTime,
                    results[i].parallelTime,
                    results[i].speedup,
                    results[i].efficiency);
        }
        fclose(csv);
        printf("Results saved to pagerank_results.csv\n");
    }
    
    freeGraph(g);
    printf("\nDone!\n");
    
    return 0;
}