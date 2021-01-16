#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

#include "csr.hpp"

void run_bf(const csr_matrix &tr, unsigned int batchsize,
		const std::vector<unsigned int> &sources) {
	// TODO
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
