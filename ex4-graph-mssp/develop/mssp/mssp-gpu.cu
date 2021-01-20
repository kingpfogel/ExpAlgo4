#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <stdint.h>

#include "csr.hpp"

__global__ void bf_iteration(int n, int s,
                             unsigned int *csr_index, unsigned int *csr_cols, float *csr_weights,
                             float *d, float *d_new, int *result) {
    uint64_t thisThread = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    uint64_t numThreads = (uint64_t)gridDim.x + (uint64_t)blockDim.x;
    uint64_t maxIndex = (uint64_t)n*(uint64_t)s;
    bool changes = false;
    for (uint64_t v = thisThread; v < maxIndex; v += numThreads) {
        float dist;
        for(uint64_t i = csr_index[v]; i < csr_index[v + 1]; ++i) {
            auto u = csr_cols[i];
            auto weight = csr_weights[i];
            uint_64_t sss = (uint64_t)u*(uint64_t)s;
            for(uint64_t o = 0; o < s; ++o){
                dist = d[v+o];
                if(dist > d[sss+o] + weight) {
                    d_new[v+o] = d[sss+o] + weight;
                    changes = true;
                }
            }

        }
    }
    if(changes)
        *result = 1;
}


void run_bf(const csr_matrix &tr, unsigned int batchsize,
		const std::vector<unsigned int> &sources) {
	// TODO
    unsigned int num_blocks = (tr.n + 255) / 256;
    unsigned int n_sources = sources.size();
    unsigned int *csr_index;
    unsigned int *csr_cols;
    float *csr_weights;
    float *d;
    float *d_new;
    int *result;

    cudaMalloc(&csr_index, (tr.n + 1) * sizeof(unsigned int));
    cudaMalloc(&csr_cols, tr.nnz * sizeof(unsigned int));
    cudaMalloc(&csr_weights, tr.nnz * sizeof(float));
    cudaMalloc(&d, (uint64_t)tr.n * (uint64_t)n_sources * (uint64_t)sizeof(float));
    cudaMalloc(&d_new, (uint64_t)tr.n * (uint64_t)n_sources * (uint64_t)sizeof(float));
    cudaMalloc(&result, sizeof(int));

    cudaMemcpy(csr_index, tr.ind.data(), (tr.n + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(csr_cols, tr.cols.data(), tr.nnz * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(csr_weights, tr.weights.data(), tr.nnz * sizeof(unsigned int), cudaMemcpyHostToDevice);

    auto algo_start = std::chrono::high_resolution_clock::now();
    std::vector<float> initial;
    initial.resize((uint64_t)tr.n * (uint64_t)n_sources);
    std::fill(initial.begin(), initial.end(), FLT_MAX);

    for(unsigned int i = 0; i < sources.size(); ++i) {
        initial[(uint64_t)sources[i] + i] = 0;
    }
    cudaMemcpy(d, initial.data(), (uint64_t)n_sources * (uint64_t)tr.n * (uint64_t)sizeof(float), cudaMemcpyHostToDevice);

    for(unsigned int i = 0; i< sources.size(); ++i){
        while(true) {
            cudaMemset(result, 0, sizeof(int));
            bf_iteration<<<num_blocks, 256>>>(tr.n, i, csr_index, csr_cols, csr_weights,
                                              d, d_new, result);

            unsigned int c;
            cudaMemcpy(&c, result, sizeof(int), cudaMemcpyDeviceToHost);
            if(!c)
                break;
            std::swap(d, d_new);
        }
    }
    auto t_algo = std::chrono::high_resolution_clock::now() - algo_start;

    std::cout << "time_sssp: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t_algo).count() << std::endl;

    cudaFree(csr_index);
    cudaFree(csr_cols);
    cudaFree(csr_weights);
    cudaFree(d);
    cudaFree(d_new);
    cudaFree(result);

}

int main(int argc, char **argv) {
	if(argc != 3)
		throw std::runtime_error("Expected instance and batch size as argument");

	unsigned int batchsize = std::atoi(argv[2]);

	std::mt19937 prng{42};
	std::uniform_real_distribution<float> weight_distrib{0.0f, 1.0f};

	// Load the graph.
	std::cout << "instance: " << argv[1] << std::endl;
	std::cout << "batchsize: " << batchsize << std::endl;

	std::ifstream ins(argv[1]);
	std::vector<std::tuple<unsigned int, unsigned int, float>> cv;

	auto io_start = std::chrono::high_resolution_clock::now();
	read_graph_unweighted(ins, [&] (unsigned int u, unsigned int v) {
		// Generate a random edge weight in [a, b).
		cv.push_back({u, v, weight_distrib(prng)});
	});

	auto mat = coordinates_to_csr(std::move(cv));
	auto t_io = std::chrono::high_resolution_clock::now() - io_start;

	std::cout << "time_io: "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(t_io).count() << std::endl;
	std::cout << "n_nodes: " << mat.n << std::endl;
	std::cout << "n_edges: " << mat.nnz << std::endl;

	auto tr = transpose(std::move(mat));

	// Generate random sources.
	std::uniform_int_distribution<unsigned int> s_distrib{0, mat.n - 1};
	std::vector<unsigned int> sources;
	for(unsigned int i = 0; i < batchsize; ++i)
		sources.push_back(s_distrib(prng));

	// Run the algorithm.
	auto algo_start = std::chrono::high_resolution_clock::now();
	run_bf(tr, batchsize, sources);
	auto t_algo = std::chrono::high_resolution_clock::now() - algo_start;

	std::cout << "time_mssp: "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(t_algo).count() << std::endl;
}
