#pragma once

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "d_ary_addressable_int_heap.hpp"

template<typename F>
void read_graph_unweighted(std::istream &ins, F fn) {
	std::string line;
	bool seen_header = false;
	while (std::getline(ins, line)) {
		if(line.empty())
			continue;
		if(line.front() == '%')
			continue;

		std::istringstream ls(line);
		unsigned int u, v;
		if (!(ls >> u >> v))
			throw std::runtime_error("Parse error while reading input graph");

		if(!seen_header) {
			seen_header = true;
			continue;
		}

		fn(u, v);
	}
}

struct csr_matrix {
	unsigned int n;
	unsigned int nnz;
	std::vector<unsigned int> ind;
	std::vector<unsigned int> cols;
	std::vector<float> weights;
};

inline csr_matrix coordinates_to_csr(unsigned int n,
		std::vector<std::tuple<unsigned int, unsigned int, float>> cv) {
	unsigned int nnz = cv.size();

	csr_matrix mat;
	mat.n = n;
	mat.nnz = nnz;
	mat.ind.resize(n + 1);
	mat.cols.resize(nnz);
	mat.weights.resize(nnz);

	// Count the number of neighbors of each node.
	for(auto ct : cv) {
		auto u = std::get<0>(ct);
		++mat.ind[u];
	}

	// Form the prefix sum.
	for(unsigned int x = 1; x <= n; ++x)
		mat.ind[x] += mat.ind[x - 1];
	assert(mat.ind[n] == nnz);

	// Insert the entries of the matrix in reverse order.
	for(auto it = cv.rbegin(); it != cv.rend(); ++it) {
		auto u = std::get<0>(*it);
		auto v = std::get<1>(*it);
		auto weight = std::get<2>(*it);
		mat.cols[mat.ind[u] - 1] = v;
		mat.weights[mat.ind[u] - 1] = weight;
		--mat.ind[u];
	}

	return mat;
}

inline csr_matrix coordinates_to_csr(
		std::vector<std::tuple<unsigned int, unsigned int, float>> cv) {
	// Determine n as the number of node IDs = maximal node ID + 1.
	unsigned int n = 0;
	for(auto ct : cv) {
		auto u = std::get<0>(ct);
		auto v = std::get<1>(ct);
		if(u > n)
			n = u;
		if(v > n)
			n = v;
	}
	++n;

	return coordinates_to_csr(n, std::move(cv));
}

inline csr_matrix transpose(const csr_matrix &in) {
	auto n = in.n;
	auto nnz = in.nnz;

	csr_matrix out;
	out.n = n;
	out.nnz = nnz;
	out.ind.resize(n + 1);
	out.cols.resize(nnz);
	out.weights.resize(nnz);

	// Count the number of neighbors of each node.
	for(unsigned int i = 0; i < nnz; ++i) {
		auto v = in.cols[i];
		++out.ind[v];
	}

	// Form the prefix sum.
	for(unsigned int x = 1; x <= n; ++x)
		out.ind[x] += out.ind[x - 1];
	assert(out.ind[n] == nnz);

	// Insert the entries of the matrix in reverse order.
	#pragma omp parallel for
	for(unsigned int u = 0; u < n; ++u) {
		for(auto i = in.ind[u]; i < in.ind[u + 1]; ++i) {
			auto v = in.cols[i];
			auto weight = in.weights[i];

			unsigned int j;
			#pragma omp atomic capture
			j = --out.ind[v];

			out.cols[j] = v;
			out.weights[j] = weight;
		}
	}

	return out;
}
