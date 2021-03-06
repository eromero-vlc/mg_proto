
# this makes the executable out of my test_file and the 'gtest_all.cc' file

add_compile_options(${EXT_CXXFLAGS})


add_executable(test_lattice_parallel test_lattice_parallel.cpp)
target_link_libraries(test_lattice_parallel qdpxx_utils gtest_all mg_test mg ${EXT_LIBS})

add_executable(test_halo test_halo.cpp)
target_link_libraries(test_halo qdpxx_utils gtest_all mg_test mg ${EXT_LIBS})

add_executable(test_parallel_coarse_dslash test_parallel_coarse_dslash.cpp)
target_link_Libraries(test_parallel_coarse_dslash qdpxx_utils gtest_all mg_test mg ${EXT_LIBS})


add_test( NAME TestLatticeParallelT COMMAND mpirun -n 2 ./test_lattice_parallel -geom 1 1 1 2 ${DEFAULT_QPHIX_TEST_ARGS} )
add_test( NAME TestLatticeParallelZ COMMAND mpirun -n 2 ./test_lattice_parallel -geom 1 1 2 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME TestLatticeParallelZT COMMAND mpirun -n 4 ./test_lattice_parallel -geom 1 1 2 2 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME TestLatticeParallelY COMMAND mpirun -n 2 ./test_lattice_parallel -geom 1 2 1 1 ${DEFAULT_QPHIX_TEST_ARGS} )
add_test( NAME TestLatticeParallelYT COMMAND mpirun -n 4 ./test_lattice_parallel -geom 1 2 1 2 ${DEFAULT_QPHIX_TEST_ARGS} )
add_test( NAME TestLatticeParallelYZT COMMAND mpirun -n 8 ./test_lattice_parallel -geom 1 2 2 2 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME TestLatticeParallelX COMMAND mpirun -n 2 ./test_lattice_parallel -geom 2 1 1 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME TestLatticeParallelXT COMMAND mpirun -n 4 ./test_lattice_parallel -geom 2 1 1 2 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME TestLatticeParallelXZT COMMAND mpirun -n 8 ./test_lattice_parallel -geom 2 1 2 2 ${DEFAULT_QPHIX_TEST_ARGS} )
add_test( NAME TestLatticeParallelXYZT COMMAND mpirun -n 16 ./test_lattice_parallel -geom 2 2 2 2 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME TestLatticeParallelT4 COMMAND mpirun -n 4 ./test_lattice_parallel -geom 1 1 1 4 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME TestHaloT COMMAND mpirun -n 2 ./test_halo -geom 1 1 1 2 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME TestHaloZ COMMAND mpirun -n 2 ./test_halo -geom 1 1 2 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME TestHaloY COMMAND mpirun -n 2 ./test_halo -geom 1 2 1 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME TestHaloX COMMAND mpirun -n 2 ./test_halo -geom 2 1 1 1 ${DEFAULT_QPHIX_TEST_ARGS} )
add_test( NAME TestHaloZT COMMAND mpirun -n 4 ./test_halo -geom 1 1 2 2 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME TestHaloYT COMMAND mpirun -n 4 ./test_halo -geom 1 2 1 2 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME TestHaloXT COMMAND mpirun -n 4 ./test_halo -geom 2 1 1 2 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME TestHaloXZT COMMAND mpirun -n 8 ./test_halo -geom 2 1 2 2 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME TestHaloXYT COMMAND mpirun -n 8 ./test_halo -geom 2 2 1 2 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME TestHaloXYZT COMMAND mpirun -n 16 ./test_halo -geom 2 2 2 2 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME TestParallelDslashT COMMAND mpirun -n 2 ./test_parallel_coarse_dslash -geom 1 1 1 2 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME TestParallelDslashZ COMMAND mpirun -n 2 ./test_parallel_coarse_dslash -geom 1 1 2 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME TestParallelDslashY COMMAND mpirun -n 2 ./test_parallel_coarse_dslash -geom 1 2 1 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME TestParallelDslashX COMMAND mpirun -n 2 ./test_parallel_coarse_dslash -geom 2 1 1 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME TestParallelDslashZT COMMAND mpirun -n 4 ./test_parallel_coarse_dslash -geom 1 1 2 2 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME TestParallelDslashYT COMMAND mpirun -n 4 ./test_parallel_coarse_dslash -geom 1 2 1 2 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME TestParallelDslashXT COMMAND mpirun -n 4 ./test_parallel_coarse_dslash -geom 2 1 1 2 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME TestParallelDslashYZ COMMAND mpirun -n 4 ./test_parallel_coarse_dslash -geom 1 2 2 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME TestParallelDslashXZ COMMAND mpirun -n 4 ./test_parallel_coarse_dslash -geom 2 1 2 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME TestParallelDslashYZT COMMAND mpirun -n 8 ./test_parallel_coarse_dslash -geom 1 2 2 2 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME TestParallelDslashXZT COMMAND mpirun -n 8 ./test_parallel_coarse_dslash -geom 2 1 2 2 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME TestParallelDslashXYZT COMMAND mpirun -n 16 ./test_parallel_coarse_dslash -geom 2 2 2 2 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME VCycleParallelT COMMAND mpirun -n 2 ../test_vcycle -geom 1 1 1 2 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME VCycleParallelZ COMMAND mpirun -n 2 ../test_vcycle -geom 1 1 2 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME VCycleParallelY COMMAND mpirun -n 2 ../test_vcycle -geom 1 2 1 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME VCycleParallelX COMMAND mpirun -n 2 ../test_vcycle -geom 2 1 1 1 ${DEFAULT_QPHIX_TEST_ARGS})
add_test( NAME RecursiveVCycleParallel COMMAND mpirun -n 2 ../test_recursive_vcycle -geom 2 1 1 1 ${DEFAULT_QPHIX_TEST_ARGS})
#add_test( NAME VCycleParallelZT COMMAND mpirun -n 4 ../test_vcycle -geom 1 1 2 2 ${DEFAULT_QPHIX_TEST_ARGS})
#add_test( NAME VCycleParallelYT COMMAND mpirun -n 4  ../test_vcycle -geom 1 2 1 2 ${DEFAULT_QPHIX_TEST_ARGS} )
#add_test( NAME VCycleParallelXT COMMAND mpirun -n 4 ../test_vcycle  -geom 2 1 1 2 ${DEFAULT_QPHIX_TEST_ARGS})
#add_test( NAME VCycleParallelYZ COMMAND mpirun -n 4  ../test_vcycle  -geom 1 2 2 1 ${DEFAULT_QPHIX_TEST_ARGS})
#add_test( NAME VCycleParallelXZ COMMAND mpirun -n 4  ../test_vcycle  -geom 2 1 2 1 ${DEFAULT_QPHIX_TEST_ARGS})
#add_test( NAME VCycleParallelYZT COMMAND mpirun -n 8  ../test_vcycle  -geom 1 2 2 2 ${DEFAULT_QPHIX_TEST_ARGS})
#add_test( NAME VCycleParallelXZT COMMAND mpirun -n 8  ../test_vcycle  -geom 2 1 2 2 ${DEFAULT_QPHIX_TEST_ARGS})
#add_test( NAME VCycleParallelXYZT COMMAND mpirun -n 16  ../test_vcycle  -geom 2 2 2 2 ${DEFAULT_QPHIX_TEST_ARGS})



