#!/bin/bash

export Kokkos_DIR=/g/g92/ciesko1/Kokkos/kokkos/install
cmake . -DKokkos_DIR=/g/g92/ciesko1/Kokkos/kokkos/install \
  -DKokkos_ENABLE_NVSHMEMSPACE=ON \
  -DNVSHMEM_ROOT=/g/g92/ciesko1/software/nvshmem_src_2.0.3-0/
  -DKokkos_ENABLE_RACERLIB=ON \ 
  -DKokkos_ENABLE_DEBUG=ON
mpirun -np 1 ./unit_test/KokkosRemoteTest
