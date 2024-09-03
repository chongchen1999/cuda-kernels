#include <cuda_runtime.h>
#include <stdio.h>
#include <fstream>
#include <sstream>

// A simple kernel that adds two vectors
__global__ void vectorAdd(const int *A, const int *B, int *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// Function to export CUDA graph to a DOT file
void exportGraphToDot(cudaGraph_t graph, const char* filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << "digraph CUDA_Graph {\n";

        size_t numNodes;
        cudaGraphNode_t* nodes;

        // Get nodes from the graph
        cudaGraphGetNodes(graph, NULL, &numNodes);
        nodes = new cudaGraphNode_t[numNodes];
        cudaGraphGetNodes(graph, nodes, &numNodes);

        for (size_t i = 0; i < numNodes; ++i) {
            cudaGraphNode_t node = nodes[i];

            cudaKernelNodeParams kernelNodeParams;
            cudaMemcpyNodeParams memcpyNodeParams;

            cudaGraphNodeType nodeType;
            cudaGraphNodeGetType(node, &nodeType);

            std::stringstream ss;
            ss << "Node" << i;

            // Identify the node type
            if (nodeType == cudaGraphNodeTypeKernel) {
                cudaGraphKernelNodeGetParams(node, &kernelNodeParams);
                file << "  " << ss.str() << " [label=\"Kernel Node\"];\n";
            } else if (nodeType == cudaGraphNodeTypeMemcpy) {
                cudaGraphMemcpyNodeGetParams(node, &memcpyNodeParams);
                file << "  " << ss.str() << " [label=\"Memcpy Node\"];\n";
            }

            // Get dependencies and add edges
            size_t numDependencies;
            cudaGraphNode_t* dependencies;
            cudaGraphNodeGetDependencies(node, NULL, &numDependencies);
            dependencies = new cudaGraphNode_t[numDependencies];
            cudaGraphNodeGetDependencies(node, dependencies, &numDependencies);

            for (size_t j = 0; j < numDependencies; ++j) {
                std::stringstream ssDep;
                ssDep << "Node" << i << " -> Node" << j << ";\n";
                file << "  " << ssDep.str();
            }

            delete[] dependencies;
        }

        delete[] nodes;

        file << "}\n";
        file.close();
    }
}

int main() {
    // Vector size
    int N = 1 << 20;
    size_t size = N * sizeof(int);

    // Allocate input vectors A and B in host memory
    int *h_A = (int *)malloc(size);
    int *h_B = (int *)malloc(size);
    int *h_C = (int *)malloc(size);

    // Initialize input vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = i;
        h_B[i] = i;
    }

    // Allocate vectors in device memory
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Create a CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Create a CUDA graph
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    // Begin recording graph
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    // Copy vectors from host memory to device memory
    cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, N);

    // Copy result from device memory to host memory
    cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream);

    // End recording graph
    cudaStreamEndCapture(stream, &graph);

    // Export the graph to a DOT file
    exportGraphToDot(graph, "cuda_graph.dot");

    // Instantiate and launch the graph
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    cudaGraphLaunch(graphExec, stream);

    // Wait for the graph execution to finish
    cudaStreamSynchronize(stream);

    // Check the result for correctness
    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            printf("Mismatch at index %d: %d != %d + %d\n", i, h_C[i], h_A[i], h_B[i]);
            break;
        }
    }

    // Clean up
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(graphExec);
    cudaStreamDestroy(stream);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Completed successfully.\n");
    return 0;
}
