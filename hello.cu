#include <stdio.h>

__global__ void hello()
{
  printf ("Pozdrav s GPU-a!\n");
}

// Calculates a sum of an array of integers
__global__ void sum(int *a, int *b, int *c, int n)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n)
    c[index] = a[index] + b[index];
}

// CUDA function equivalent to the Python function isSubgraph


__global__ void isSubgraph(int *result,int *graph, int *subgraph, int n, int m)
{
  int index = threadIdx.y * blockDim.x + threadIdx.x;
  if (index < n)
  {
    int i = index / m;
    int j = index % m;
    if (graph[i * m + j] != subgraph[j])
    {
      result[index] = 0;
    }
    else{
      result[index] = 1;
    }
  }
}

// Check if a subgraph is a motif by comparing it to a list of motif_candidates
__global__ void isMotif(int *subgraph, int *motif_candidates, int *result, int n)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n)
  {
    if (subgraph[0] == motif_candidates[index * 3] && subgraph[1] == motif_candidates[index * 3 + 1] && subgraph[2] == motif_candidates[index * 3 + 2])
      result[index] = 1;
  }
}