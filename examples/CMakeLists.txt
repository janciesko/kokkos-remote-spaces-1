if (NOT Kokkos_ENABLE_CUDARDMASPACE)
add_subdirectory(cgsolve)
add_subdirectory(benchmarks)
endif()