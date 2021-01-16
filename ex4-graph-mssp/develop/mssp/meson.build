project('expalgo-mssp', 'cpp', 'cuda')

omp = dependency('openmp')

executable('mssp-cpu',
	'mssp-cpu.cpp',
	dependencies: omp,
	override_options: ['cpp_std=c++17'],
	install: true)

executable('mssp-gpu',
	'mssp-gpu.cu',
	override_options: ['cpp_std=c++17'],
	install: true)
