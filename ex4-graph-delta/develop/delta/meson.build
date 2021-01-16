project('expalgo-delta', 'cpp')

omp = dependency('openmp')

executable('sssp',
	'main.cpp',
	dependencies: omp,
	override_options: ['cpp_std=c++17'],
	install: true)
