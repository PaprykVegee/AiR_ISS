#include "bfs.h"

namespace bfs
{
    void readTestcaseFromFile(const fs::path &filePath, GraphCSR &graph, unsigned int &startNode)
    {
        std::ifstream file(filePath);
        if (!file.is_open())
        {
            throw std::runtime_error("Could not open file: " + filePath.string());
        }

        // Read source node idx
        file >> startNode;

        // Read edges array
        int numEdges;
        file >> numEdges;
        graph.edges.resize(numEdges);
        for (int i = 0; i < numEdges; ++i)
        {
            file >> graph.edges[i];
        }

        // Read dest array
        int numDest;
        file >> numDest;
        graph.dest.resize(numDest);
        for (int i = 0; i < numDest; ++i)
        {
            file >> graph.dest[i];
        }
    }

    // ------------------- Kernel globalnej kolejki -------------------
    __global__ void kernelGlobalQueue(int *edges, int *dest, int *label,
                                      int *pFrontier, int *cFrontier,
                                      int *pFrontierTail, int *cFrontierTail)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= *pFrontierTail) return;

        int v = pFrontier[idx];

        for (int i = edges[v]; i < edges[v + 1]; ++i)
        {
            int neighbor = dest[i];
            // atomicCAS zapewnia, że tylko jeden wątek oznaczy sąsiada
            if (atomicCAS(&label[neighbor], -1, label[v] + 1) == -1)
            {
                int pos = atomicAdd(cFrontierTail, 1);
                cFrontier[pos] = neighbor;
            }
        }
    }

    // ------------------- Kernel block-level queue -------------------
    __global__ void kernelBlockQueue(int *edges, int *dest, int *label,
                                     int *pFrontier, int *cFrontier,
                                     int *pFrontierTail, int *cFrontierTail)
    {
        __shared__ int blockQueue[1024];
        __shared__ int blockTail;

        if (threadIdx.x == 0) blockTail = 0;
        __syncthreads();

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= *pFrontierTail) return;

        int v = pFrontier[idx];

        for (int i = edges[v]; i < edges[v + 1]; ++i)
        {
            int neighbor = dest[i];
            if (atomicCAS(&label[neighbor], -1, label[v] + 1) == -1)
            {
                int pos = atomicAdd(&blockTail, 1);
                blockQueue[pos] = neighbor;
            }
        }

        __syncthreads();

        if (threadIdx.x == 0)
        {
            int pos = atomicAdd(cFrontierTail, blockTail);
            for (int i = 0; i < blockTail; ++i)
                cFrontier[pos + i] = blockQueue[i];
        }
    }

    // ------------------- BFS na GPU -------------------
    std::vector<int> bfsOnDevice(const GraphCSR &graph, unsigned int source, BFSQueueType queueType)
    {
        int nodes = static_cast<int>(graph.edges.size() - 1);
        int edgesSize = static_cast<int>(graph.edges.size());
        int destSize  = static_cast<int>(graph.dest.size());

        // Alokacja pamięci GPU
        int *d_edges, *d_dest, *d_label;
        int *d_pFrontier, *d_cFrontier;
        int *d_pFrontierTail, *d_cFrontierTail;

        cudaMalloc(&d_edges, edgesSize * sizeof(int));
        cudaMalloc(&d_dest, destSize * sizeof(int));
        cudaMalloc(&d_label, nodes * sizeof(int));
        cudaMalloc(&d_pFrontier, nodes * sizeof(int));
        cudaMalloc(&d_cFrontier, nodes * sizeof(int));
        cudaMalloc(&d_pFrontierTail, sizeof(int));
        cudaMalloc(&d_cFrontierTail, sizeof(int));

        cudaMemcpy(d_edges, graph.edges.data(), edgesSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dest,  graph.dest.data(),  destSize * sizeof(int), cudaMemcpyHostToDevice);

        std::vector<int> h_label(nodes, -1);
        h_label[source] = 0;
        cudaMemcpy(d_label, h_label.data(), nodes * sizeof(int), cudaMemcpyHostToDevice);

        int h_pFrontierTail = 1;
        cudaMemcpy(d_pFrontier, &source, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pFrontierTail, &h_pFrontierTail, sizeof(int), cudaMemcpyHostToDevice);

        // BFS while
        while (true)
        {
            int zero = 0;
            cudaMemcpy(d_cFrontierTail, &zero, sizeof(int), cudaMemcpyHostToDevice);

            int threadsPerBlock = 256;
            int blocks = (h_pFrontierTail + threadsPerBlock - 1) / threadsPerBlock;

            if (queueType == BFSQueueType::Global)
                kernelGlobalQueue<<<blocks, threadsPerBlock>>>(d_edges, d_dest, d_label,
                                                               d_pFrontier, d_cFrontier,
                                                               d_pFrontierTail, d_cFrontierTail);


            cudaDeviceSynchronize();

            int newFrontierSize;
            cudaMemcpy(&newFrontierSize, d_cFrontierTail, sizeof(int), cudaMemcpyDeviceToHost);

            if (newFrontierSize == 0) break;

            // swap frontier
            std::swap(d_pFrontier, d_cFrontier);
            cudaMemcpy(d_pFrontierTail, &newFrontierSize, sizeof(int), cudaMemcpyHostToDevice);
            h_pFrontierTail = newFrontierSize;
        }

        // Odczyt wyniku
        cudaMemcpy(h_label.data(), d_label, nodes * sizeof(int), cudaMemcpyDeviceToHost);

        // Cleanup
        cudaFree(d_edges);
        cudaFree(d_dest);
        cudaFree(d_label);
        cudaFree(d_pFrontier);
        cudaFree(d_cFrontier);
        cudaFree(d_pFrontierTail);
        cudaFree(d_cFrontierTail);

        return h_label;
    }

    std::vector<int> bfsOnHost(const GraphCSR &graph, unsigned int source)
    {
        int nodes = static_cast<int>(graph.edges.size() - 1);
        std::vector<int> label(nodes, -1);
        label[source] = 0;

        std::vector<int> pFrontier;
        pFrontier.push_back(source);

        while (!pFrontier.empty())
        {
            std::vector<int> cFrontier;
            for (const auto &cVertex : pFrontier)
            {
                for (int i = graph.edges[cVertex]; i < graph.edges[cVertex + 1]; ++i)
                {
                    int neighbor = graph.dest[i];
                    if (label[neighbor] == -1)
                    {
                        label[neighbor] = label[cVertex] + 1;
                        cFrontier.push_back(neighbor);
                    }
                }
            }
            pFrontier.swap(cFrontier);
        }

        return label;
    }
} // namespace bfs
