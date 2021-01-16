#include <omp.h>
#include <chrono>
#include <cstring>
#include <fstream>
#include <random>

#include "csr.hpp"

struct dijkstra {
	void run(const csr_matrix &mat, unsigned int s) {
		d.resize(mat.n);

		std::fill(d.begin(), d.end(), FLT_MAX);
		d[s] = 0;
		pq.push(s);

		while(!pq.empty()) {
			auto u = pq.extract_top();

			// Iterate over the neighbors in CSR style.
			for(unsigned int i = mat.ind[u]; i < mat.ind[u + 1]; ++i) {
				auto v = mat.cols[i];
				auto weight = mat.weights[i];
				
				if(d[v] > d[u] + weight) {
					d[v] = d[u] + weight;
					pq.update(v);
				}
			}
		}
	}

private:
	class compare {
	public:
		bool operator() (unsigned int u, unsigned int v) const {
			return self->d[u] < self->d[v];
		}

		dijkstra *self;
	};

	std::vector<float> d;
	DAryAddressableIntHeap<unsigned int, 2, compare> pq{compare{this}};
};

struct bellman_ford {
	void run(const csr_matrix &tr, unsigned int s) {
		d.resize(tr.n);
		d_new.resize(tr.n);

		std::fill(d.begin(), d.end(), FLT_MAX);
		d[s] = 0;

		bool changes = false;

		#pragma omp parallel
		{
			do {
				#pragma omp for reduction(||: changes)
				for(unsigned int v = 0; v < tr.n; ++v) {
					d_new[v] = d[v];

					for(unsigned int i = tr.ind[v]; i < tr.ind[v + 1]; ++i) {
						auto u = tr.cols[i]; // Kante: v -> u in tr, u -> v im Input.
						auto weight = tr.weights[i];

						if(d_new[v] > d[u] + weight) {
							d_new[v] = d[u] + weight;
							changes = true;
						}
					}
				}
				
				#pragma omp single
				std::swap(d, d_new);
			} while(changes);
		}
	}

private:
	std::vector<float> d;
	std::vector<float> d_new;
};

int main(int argc, char **argv) {
	if(argc != 3)
		throw std::runtime_error("Expected algorithm and instance as argument");

	std::mt19937 prng{42};
	std::uniform_real_distribution<float> weight_distrib{0.0f, 1.0f};

	// Load the graph.
	std::cout << "instance: " << argv[2] << std::endl;
	std::cout << "threads: " << omp_get_max_threads() << std::endl;

	std::ifstream ins(argv[2]);
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

	// Compute the transpose (for Bellman-Ford).
	auto tr = transpose(mat);

	std::uniform_int_distribution<unsigned int> s_distrib{0, mat.n - 1};

	auto algo_start = std::chrono::high_resolution_clock::now();
	if(!strcmp(argv[1], "dijkstra")) {
		dijkstra algo;
		algo.run(mat, s_distrib(prng));
	}else if(!strcmp(argv[1], "bf")) {
		bellman_ford algo;
		algo.run(tr, s_distrib(prng));
	}else if(!strcmp(argv[1], "delta")) {
		// TODO
	}else{
		throw std::runtime_error("Unexpected algorithm");
	}
	auto t_algo = std::chrono::high_resolution_clock::now() - algo_start;

	std::cout << "time_sssp: "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(t_algo).count() << std::endl;
}
