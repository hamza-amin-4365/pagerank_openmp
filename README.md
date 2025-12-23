# Google Page Rank Algorithm

In this repo, we implemented page rank on stanford webpage directed dataset which has more than 280,000 nodes and 2312497 edges.

## OpenMP and Parallelization 

We used OpenMp to explore the possiblity of parallelization of this algorithm by comparing the effeciency against differnt number of threads. 

To be more precise:
```
Threads | Time (s)  | Speedup | Efficiency
--------|-----------|---------|----------
   1    | 1.1807    | 1.00x   | 100.00%
   1    | 1.2357    | 0.96x   | 95.55%
   2    | 0.6816    | 1.73x   | 86.61%
   4    | 0.5117    | 2.31x   | 57.69%
   8    | 0.6325    | 1.87x   | 23.33%
  16    | 0.5440    | 2.17x   | 13.57%
  32    | 0.5362    | 2.20x   | 6.88%

```

## How to Get Started

### Running Using **Makefile**

Build the compiled file

```
make
```

Run the script according to how many threads you like

```
# num_threads = 1
make run-1 

# num_threads = 2
make run-2

# num_threads = 4
make run-4

# num_threads = 8
make run-8

# num_threads = 16
make run-16
```

Clean the compiled versions

```
make clean
```

- - -

Run the following cmd to compile and make exe of the algorithm.

```
gcc -O3 -fopenmp pagerank.c -o pagerank -lm
```

Now Execute the exe file:
```
./pagerank test_data/web-Stanford.txt
```

## Generate Plots using Python

The python script is utilized to generate plots of the results. Ensure to install basic ds libs like numpy, pandas & matplotlib.

```
python3 analyze_pagerank.py pagerank_results.csv

```
