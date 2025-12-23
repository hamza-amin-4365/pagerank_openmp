#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <time.h>
#include <assert.h>

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

/* ============================================================================
   FORWARD DECLARATIONS
   ============================================================================ */

void freeGraph(Graph* g);

typedef struct {
    int numThreads;
    double sequentialTime;
    double parallelTime;
    double speedup;
    double efficiency;
    int iterations;
    double finalDiff;
} BenchmarkResult;

/* ============================================================================
   GRAPH LOADING - FIXED VERSION
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
   
    // Allocate node array
    g->nodes = (Node*)calloc(numNodes, sizeof(Node));
    if (!g->nodes) {
        fprintf(stderr, "Error: Failed to allocate nodes array\n");
        free(g);
        return NULL;
    }
   
    // Allocate incoming edges arrays
    g->inEdges = (int**)calloc(numNodes, sizeof(int*));
    if (!g->inEdges) {
        fprintf(stderr, "Error: Failed to allocate inEdges array\n");
        free(g->nodes);
        free(g);
        return NULL;
    }
   
    g->inDegree = (int*)calloc(numNodes, sizeof(int));
    g->inCapacity = (int*)calloc(numNodes, sizeof(int));
   
    if (!g->inDegree || !g->inCapacity) {
        fprintf(stderr, "Error: Failed to allocate degree arrays\n");
        free(g->nodes);
        free(g->inEdges);
        free(g->inDegree);
        free(g->inCapacity);
        free(g);
        return NULL;
    }
   
    // Initialize node capacities
    for (int i = 0; i < numNodes; i++) {
        g->nodes[i].outLinks = (int*)malloc(INITIAL_CAPACITY * sizeof(int));
        if (!g->nodes[i].outLinks) {
            fprintf(stderr, "Error: Failed to allocate outLinks for node %d\n", i);
            // Cleanup and return NULL
            for (int j = 0; j < i; j++) {
                free(g->nodes[j].outLinks);
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
       
        g->inEdges[i] = (int*)malloc(INITIAL_CAPACITY * sizeof(int));
        if (!g->inEdges[i]) {
            fprintf(stderr, "Error: Failed to allocate inEdges for node %d\n", i);
            free(g->nodes[i].outLinks);
            free(g->nodes);
            free(g->inEdges);
            free(g->inDegree);
            free(g->inCapacity);
            free(g);
            return NULL;
        }
        g->inDegree[i] = 0;
        g->inCapacity[i] = INITIAL_CAPACITY;
    }
   
    if (VERBOSE) {
        printf("Created graph with %d nodes\n", numNodes);
    }
   
    return g;
}

void addEdge(Graph* g, int src, int dst) {
    // Validate nodes
    if (src < 0 || src >= g->numNodes || dst < 0 || dst >= g->numNodes) {
        fprintf(stderr, "Warning: Invalid edge (%d, %d) - skipping\n", src, dst);
        return;
    }
   
    // Skip self-loops
    if (src == dst) {
        return;
    }
   
    // Add to outgoing edges
    if (g->nodes[src].numOutLinks == g->nodes[src].capacity) {
        int newCapacity = g->nodes[src].capacity * 2;
        int* newLinks = (int*)realloc(g->nodes[src].outLinks,
                                      newCapacity * sizeof(int));
        if (!newLinks) {
            fprintf(stderr, "Error: Failed to expand outLinks for node %d\n", src);
            return;
        }
        g->nodes[src].outLinks = newLinks;
        g->nodes[src].capacity = newCapacity;
    }
    g->nodes[src].outLinks[g->nodes[src].numOutLinks++] = dst;
   
    // Add to incoming edges
    if (g->inDegree[dst] == g->inCapacity[dst]) {
        int newCapacity = g->inCapacity[dst] * 2;
        int* newEdges = (int*)realloc(g->inEdges[dst],
                                      newCapacity * sizeof(int));
        if (!newEdges) {
            fprintf(stderr, "Error: Failed to expand inEdges for node %d\n", dst);
            return;
        }
        g->inEdges[dst] = newEdges;
        g->inCapacity[dst] = newCapacity;
    }
    g->inEdges[dst][g->inDegree[dst]++] = src;
   
    g->numEdges++;
}

Graph* loadGraphFromFile(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        return NULL;
    }
   
    printf("Loading graph from: %s\n", filename);
    printf("Reading file to determine graph size...\n");
   
    int maxNode = -1;
    int src, dst;
    long edgeCount = 0;
   
    // First pass: find max node ID and count edges
    while (fscanf(file, "%d %d", &src, &dst) == 2) {
        // Skip comments or headers
        if (src < 0 || dst < 0) {
            continue;
        }
        edgeCount++;
        if (src > maxNode) maxNode = src;
        if (dst > maxNode) maxNode = dst;
    }
   
    if (maxNode < 0) {
        fprintf(stderr, "Error: No valid edges found in file\n");
        fclose(file);
        return NULL;
    }
   
    int numNodes = maxNode + 1;
    printf("Graph size: %d nodes, %ld edges\n", numNodes, edgeCount);
   
    // Check if graph is too large
    if (numNodes > MAX_NODES) {
        fprintf(stderr, "Error: Graph too large (%d nodes > %d max)\n",
                numNodes, MAX_NODES);
        fclose(file);
        return NULL;
    }
   
    // Check memory requirements
    long memRequired = (long)numNodes * (8 + 8);  // pageRank arrays
    memRequired += (long)edgeCount * (4 + 4);     // edge arrays
    memRequired += (long)numNodes * 16;           // overhead
   
    printf("Estimated memory: %.2f MB\n", memRequired / (1024.0 * 1024.0));
   
    // Create graph
    Graph* g = createGraph(numNodes);
    if (!g) {
        fclose(file);
        return NULL;
    }
   
    // Second pass: add edges
    rewind(file);
    long addedEdges = 0;
    long skippedEdges = 0;
   
    while (fscanf(file, "%d %d", &src, &dst) == 2) {
        if (src >= 0 && dst >= 0 && src != dst) {
            addEdge(g, src, dst);
            addedEdges++;
           
            if (addedEdges % 1000000 == 0) {
                printf("  Loaded %ld edges...\n", addedEdges);
            }
        } else {
            skippedEdges++;
        }
    }
   
    fclose(file);
   
    printf("Loaded graph: %d nodes, %ld edges\n", g->numNodes, g->numEdges);
    if (skippedEdges > 0) {
        printf("Skipped %ld invalid edges\n", skippedEdges);
    }
   
    // Verify graph
    if (g->numEdges == 0) {
        fprintf(stderr, "Error: Graph has no edges\n");
        freeGraph(g);
        return NULL;
    }
   
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
   SEQUENTIAL PAGERANK - BASELINE
   ============================================================================ */

void pageRankSequential(Graph* g, double* pageRank,
                       double damping, double threshold, int maxIters) {
    int n = g->numNodes;
    double* newPageRank = (double*)malloc(n * sizeof(double));
   
    if (!newPageRank) {
        fprintf(stderr, "Error: Failed to allocate memory for newPageRank\n");
        return;
    }
   
    double base = (1.0 - damping) / n;
   
    printf("\nRunning Sequential PageRank...\n");
   
    // Initialize
    for (int i = 0; i < n; i++) {
        pageRank[i] = 1.0 / n;
    }
   
    // Iterate
    for (int iter = 0; iter < maxIters; iter++) {
        // Reset new scores
        for (int i = 0; i < n; i++) {
            newPageRank[i] = base;
        }
       
        // Pull-based: each node gathers from incoming links
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < g->inDegree[i]; j++) {
                int source = g->inEdges[i][j];
                int outDegree = g->nodes[source].numOutLinks;
                if (outDegree > 0) {
                    sum += pageRank[source] / outDegree;
                }
            }
            newPageRank[i] += damping * sum;
        }
       
        // Check convergence
        double diff = 0.0;
        for (int i = 0; i < n; i++) {
            diff += fabs(newPageRank[i] - pageRank[i]);
        }
       
        // Swap arrays
        double* temp = pageRank;
        pageRank = newPageRank;
        newPageRank = temp;
       
        if (iter % 5 == 0) {
            printf("  Iter %d: diff = %.2e\n", iter, diff);
        }
       
        if (diff < threshold) {
            printf("  Converged at iteration %d with diff = %.2e\n", iter, diff);
            break;
        }
    }
   
    free(newPageRank);
}

/* ============================================================================
   PARALLEL PAGERANK - OPENMP (PULL-BASED)
   ============================================================================ */

void pageRankParallel(Graph* g, double* pageRank,
                     double damping, double threshold, int maxIters) {
    int n = g->numNodes;
    double* newPageRank = (double*)malloc(n * sizeof(double));
   
    if (!newPageRank) {
        fprintf(stderr, "Error: Failed to allocate memory for newPageRank\n");
        return;
    }
   
    double base = (1.0 - damping) / n;
   
    printf("\nRunning Parallel PageRank (OpenMP)...\n");
   
    // Initialize
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        pageRank[i] = 1.0 / n;
    }
   
    // Iterate
    for (int iter = 0; iter < maxIters; iter++) {
        // Reset new scores
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            newPageRank[i] = base;
        }
       
        // Pull-based: each node gathers from incoming links (NO ATOMICS!)
        #pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < g->inDegree[i]; j++) {
                int source = g->inEdges[i][j];
                int outDegree = g->nodes[source].numOutLinks;
                if (outDegree > 0) {
                    sum += pageRank[source] / outDegree;
                }
            }
            newPageRank[i] += damping * sum;
        }
       
        // Check convergence (with reduction)
        double diff = 0.0;
        #pragma omp parallel for schedule(static) reduction(+:diff)
        for (int i = 0; i < n; i++) {
            diff += fabs(newPageRank[i] - pageRank[i]);
        }
       
        // Swap arrays
        double* temp = pageRank;
        pageRank = newPageRank;
        newPageRank = temp;
       
        if (iter % 5 == 0) {
            printf("  Iter %d: diff = %.2e\n", iter, diff);
        }
       
        if (diff < threshold) {
            printf("  Converged at iteration %d with diff = %.2e\n", iter, diff);
            break;
        }
    }
   
    free(newPageRank);
}

/* ============================================================================
   BENCHMARKING
   ============================================================================ */

BenchmarkResult runBenchmark(Graph* g, int numThreads,
                            double damping, double threshold, int maxIters) {
    BenchmarkResult result;
    result.numThreads = numThreads;
    result.iterations = 0;
    result.finalDiff = 0.0;
    result.sequentialTime = 0.0;
    result.parallelTime = 0.0;
    result.speedup = 0.0;
    result.efficiency = 0.0;
   
    double* pageRankSeq = (double*)calloc(g->numNodes, sizeof(double));
    double* pageRankPar = (double*)calloc(g->numNodes, sizeof(double));
   
    if (!pageRankSeq || !pageRankPar) {
        fprintf(stderr, "Error: Failed to allocate PageRank arrays\n");
        free(pageRankSeq);
        free(pageRankPar);
        return result;
    }
   
    // Sequential benchmark
    printf("\n--- Sequential Execution (1 thread) ---\n");
    double seqStart = omp_get_wtime();
    pageRankSequential(g, pageRankSeq, damping, threshold, maxIters);
    double seqEnd = omp_get_wtime();
    result.sequentialTime = seqEnd - seqStart;
    printf("Sequential time: %.4f seconds\n", result.sequentialTime);
   
    // Parallel benchmark
    printf("\n--- Parallel Execution (%d threads) ---\n", numThreads);
    omp_set_num_threads(numThreads);
    double parStart = omp_get_wtime();
    pageRankParallel(g, pageRankPar, damping, threshold, maxIters);
    double parEnd = omp_get_wtime();
    result.parallelTime = parEnd - parStart;
    printf("Parallel time: %.4f seconds\n", result.parallelTime);
   
    // Calculate metrics
    if (result.parallelTime > 0.0001) {
        result.speedup = result.sequentialTime / result.parallelTime;
        result.efficiency = (result.speedup / numThreads) * 100.0;
    }
   
    printf("\nSpeedup: %.2fx\n", result.speedup);
    printf("Efficiency: %.2f%%\n", result.efficiency);
   
    free(pageRankSeq);
    free(pageRankPar);
   
    return result;
}

/* ============================================================================
   SYNTHETIC GRAPH GENERATION FOR TESTING
   ============================================================================ */

Graph* generateRandomGraph(int numNodes, int avgDegree, unsigned int seed) {
    printf("Generating synthetic graph: %d nodes, avg degree %d\n",
           numNodes, avgDegree);
   
    srand(seed);
    Graph* g = createGraph(numNodes);
    if (!g) return NULL;
   
    long edgesAdded = 0;
    for (int i = 0; i < numNodes; i++) {
        int degree = (rand() % (avgDegree * 2)) + 1;
        for (int j = 0; j < degree; j++) {
            int target = rand() % numNodes;
            if (target != i) {
                addEdge(g, i, target);
                edgesAdded++;
            }
        }
       
        if ((i + 1) % (numNodes / 10 + 1) == 0) {
            printf("  Generated %d/%d nodes...\n", i + 1, numNodes);
        }
    }
   
    printf("Generated graph: %d nodes, %ld edges\n", g->numNodes, g->numEdges);
    return g;
}

/* ============================================================================
   ANALYSIS AND REPORTING
   ============================================================================ */

void printAnalysis(Graph* g, BenchmarkResult* results, int numResults) {
    printf("\n");
    printf("================================================================================\n");
    printf("PAGERANK PARALLEL ANALYSIS AND BENCHMARKING\n");
    printf("================================================================================\n\n");
   
    printf("DATASET CHARACTERISTICS:\n");
    printf("  Nodes: %d\n", g->numNodes);
    printf("  Edges: %ld\n", g->numEdges);
    printf("  Density: %.6f\n", (double)g->numEdges / (g->numNodes * (g->numNodes - 1)));
    printf("  Avg In-Degree: %.2f\n", (double)g->numEdges / g->numNodes);
    printf("  Avg Out-Degree: %.2f\n", (double)g->numEdges / g->numNodes);
   
    printf("\n================================================================================\n");
    printf("COMPLEXITY ANALYSIS\n");
    printf("================================================================================\n\n");
   
    printf("SEQUENTIAL COMPLEXITY:\n");
    printf("  Time per iteration: O(n + m)\n");
    printf("    where n = number of nodes = %d\n", g->numNodes);
    printf("          m = number of edges = %ld\n", g->numEdges);
    printf("  Total time: O(k * (n + m))\n");
    printf("    where k = number of iterations (typically 30-50)\n");
    printf("  Space complexity: O(n + m)\n\n");
   
    printf("PARALLEL COMPLEXITY (OPENMP):\n");
    printf("  Work (total operations): W = O(k * (n + m))\n");
    printf("  Span (critical path depth): S = O(k * log(n))\n");
    printf("    - Each iteration: O(n/p) for computation + O(log(n)) for reduction\n");
    printf("    - Reduction on diff variable uses tree-based summation\n");
    printf("  Parallelism: W/S = O((n + m) / log(n))\n");
    printf("  Optimal threads: min(p, (n + m) / log(n))\n\n");
   
    printf("================================================================================\n");
    printf("BENCHMARK RESULTS\n");
    printf("================================================================================\n\n");
   
    printf("%-10s %-15s %-15s %-12s %-12s\n",
           "Threads", "Time (sec)", "Speedup", "Efficiency", "Speedup/Thread");
    printf("%-10s %-15s %-15s %-12s %-12s\n",
           "-------", "----------", "--------", "----------", "-------------");
   
    for (int i = 0; i < numResults; i++) {
        printf("%-10d %-15.4f %-15.2fx %-11.2f%% %-11.4f\n",
               results[i].numThreads,
               results[i].parallelTime,
               results[i].speedup,
               results[i].efficiency,
               results[i].speedup / results[i].numThreads);
    }
   
    printf("\n================================================================================\n");
    printf("PERFORMANCE CHARACTERISTICS\n");
    printf("================================================================================\n\n");
   
    if (numResults > 0) {
        int bestEffIdx = 0;
        for (int i = 1; i < numResults; i++) {
            if (results[i].efficiency > results[bestEffIdx].efficiency) {
                bestEffIdx = i;
            }
        }
       
        printf("Peak speedup: %.2fx at %d threads\n",
               results[numResults-1].speedup, results[numResults-1].numThreads);
        printf("Best efficiency: %.2f%% at %d threads\n",
               results[bestEffIdx].efficiency, results[bestEffIdx].numThreads);
    }
   
    printf("\n================================================================================\n");
}

void generateCSVReport(BenchmarkResult* results, int numResults, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error: Could not open %s for writing\n", filename);
        return;
    }
   
    fprintf(file, "Threads,SequentialTime,ParallelTime,Speedup,Efficiency\n");
   
    for (int i = 0; i < numResults; i++) {
        fprintf(file, "%d,%.4f,%.4f,%.4f,%.2f\n",
                results[i].numThreads,
                results[i].sequentialTime,
                results[i].parallelTime,
                results[i].speedup,
                results[i].efficiency);
    }
    fclose(file);
    printf("\nResults saved to %s\n", filename);
}

/* ============================================================================
   MAIN PROGRAM
   ============================================================================ */

int main(int argc, char** argv) {
    printf("================================================================================\n");
    printf("PARALLEL PAGERANK WITH OPENMP\n");
    printf("================================================================================\n\n");
   
    printf("Maximum threads available: %d\n\n", omp_get_max_threads());
   
    Graph* g = NULL;
   
    // Load graph from file or generate synthetic graph
    if (argc > 1) {
        printf("Loading graph from file: %s\n", argv[1]);
        g = loadGraphFromFile(argv[1]);
        if (!g) {
            fprintf(stderr, "Failed to load graph. Exiting.\n");
            return 1;
        }
    } else {
        printf("No input file provided. Generating synthetic test graphs...\n\n");
       
        int test_sizes[] = {10000, 50000};
       
        for (int sz_idx = 0; sz_idx < 2; sz_idx++) {
            int size = test_sizes[sz_idx];
            printf("\n>>> Testing with synthetic graph: %d nodes\n", size);
           
            Graph* test_g = generateRandomGraph(size, 25, 42);
            if (!test_g) {
                fprintf(stderr, "Failed to generate test graph of size %d\n", size);
                continue;
            }
           
            int thread_counts[] = {1, 2, 4, 8};
            BenchmarkResult results[4];
            int num_valid_threads = 0;
           
            for (int i = 0; i < 4; i++) {
                if (thread_counts[i] <= omp_get_max_threads()) {
                    results[i] = runBenchmark(test_g, thread_counts[i], 0.85, 1e-6, 100);
                    num_valid_threads++;
                }
            }
           
            if (num_valid_threads > 0) {
                printAnalysis(test_g, results, num_valid_threads);
               
                char csv_file[256];
                snprintf(csv_file, sizeof(csv_file), "pagerank_results_%d_nodes.csv", size);
                generateCSVReport(results, num_valid_threads, csv_file);
            }
           
            freeGraph(test_g);
        }
       
        return 0;
    }
   
    // Single test with provided graph
    if (g) {
        int thread_counts[] = {1, 2, 4, 8, 16};
        BenchmarkResult results[5];
        int num_threads = 0;
       
        // Determine how many thread counts to test
        for (int i = 0; i < 5; i++) {
            if (thread_counts[i] <= omp_get_max_threads()) {
                results[num_threads] = runBenchmark(g, thread_counts[i], 0.85, 1e-6, 100);
                num_threads++;
            }
        }
       
        if (num_threads > 0) {
            printAnalysis(g, results, num_threads);
            generateCSVReport(results, num_threads, "pagerank_results.csv");
        }
       
        freeGraph(g);
    }
   
    return 0;
}
