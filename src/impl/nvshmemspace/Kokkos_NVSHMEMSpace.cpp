//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Contact: Jan Ciesko (jciesko@sandia.gov)
//
//@HEADER

#include <Kokkos_NVSHMEMSpace.hpp>
#include <nvshmem.h>

namespace Kokkos {
namespace Experimental {

/* Default allocation mechanism */
NVSHMEMSpace::NVSHMEMSpace()
    : allocation_mode(Kokkos::Experimental::Symmetric) {}

void NVSHMEMSpace::impl_set_allocation_mode(const int allocation_mode_) {
  allocation_mode = allocation_mode_;
}

void NVSHMEMSpace::impl_set_extent(const int64_t extent_) { extent = extent_; }

void *NVSHMEMSpace::allocate(const size_t arg_alloc_size) const {
  static_assert(sizeof(void *) == sizeof(uintptr_t),
                "Error sizeof(void*) != sizeof(uintptr_t)");

  static_assert(
      Kokkos::Impl::is_integral_power_of_two(Kokkos::Impl::MEMORY_ALIGNMENT),
      "Memory alignment must be power of two");

  void *ptr = 0;
  if (arg_alloc_size) {
    if (allocation_mode == Kokkos::Experimental::Symmetric) {
      int num_pes = nvshmem_n_pes();
      int my_id   = nvshmem_my_pe();
      ptr         = nvshmem_malloc(arg_alloc_size);
    } else {
      Kokkos::abort("NVSHMEMSpace only supports symmetric allocation policy.");
    }
  }
  return ptr;
}

void NVSHMEMSpace::deallocate(void *const arg_alloc_ptr, const size_t) const {
  nvshmem_free(arg_alloc_ptr);
}

void NVSHMEMSpace::fence() {
  Kokkos::fence();
  nvshmem_barrier_all();
}

KOKKOS_FUNCTION
int get_num_pes() { return nvshmem_n_pes(); }

KOKKOS_FUNCTION
int get_my_pe() { return nvshmem_my_pe(); }

KOKKOS_FUNCTION
size_t get_indexing_block_size(size_t size) {
  int num_pes;
  size_t block;
  num_pes = get_num_pes();
  block   = (size + static_cast<size_t>(num_pes) - 1) / num_pes;
  return block;
}

template <typename T>
KOKKOS_FUNCTION Kokkos::pair<T, T> getRange(T size, int pe) {
  T start, end;
  T block = get_indexing_block_size(size);
  start   = static_cast<T>(pe) * block;
  end     = (static_cast<T>(pe) + 1) * block;

  T num_pes = get_num_pes();
  if (size < num_pes) {
    T diff = (num_pes * block) - size;
    if (pe > num_pes - 1 - diff) end--;
  } else {
    if (pe == num_pes - 1) {
      size_t diff = size - (num_pes - 1) * block;
      end         = start + diff;
    }
  }
  return Kokkos::pair<T, T>(start, end);
}

template <typename T>
KOKKOS_FUNCTION Kokkos::pair<T, T> get_range(T size, int pe) {
  return getRange(size, pe);
}

template <typename T>
KOKKOS_FUNCTION Kokkos::pair<T, T> get_local_range(T size) {
  auto pe = get_my_pe();
  return getRange(size, pe);
}

template KOKKOS_FUNCTION Kokkos::pair<size_t, size_t> get_range<size_t>(
    size_t size, int p);
template KOKKOS_FUNCTION Kokkos::pair<size_t, size_t> get_local_range<size_t>(
    size_t size);
template KOKKOS_FUNCTION Kokkos::pair<size_t, size_t> getRange<size_t>(
    size_t size, int pe);

template KOKKOS_FUNCTION Kokkos::pair<int, int> get_range<int>(int size,
                                                               int pe);
template KOKKOS_FUNCTION Kokkos::pair<int, int> get_local_range<int>(int size);
template KOKKOS_FUNCTION Kokkos::pair<int, int> getRange<int>(int size, int pe);

}  // namespace Experimental

namespace Impl {

Kokkos::Impl::DeepCopy<HostSpace, Kokkos::Experimental::NVSHMEMSpace>::DeepCopy(
    void *dst, const void *src, size_t n) {
  Kokkos::Experimental::NVSHMEMSpace().fence();
  cudaMemcpy(dst, src, n, cudaMemcpyDefault);
}

Kokkos::Impl::DeepCopy<Kokkos::Experimental::NVSHMEMSpace, HostSpace>::DeepCopy(
    void *dst, const void *src, size_t n) {
  Kokkos::Experimental::NVSHMEMSpace().fence();
  cudaMemcpy(dst, src, n, cudaMemcpyDefault);
}

template <typename ExecutionSpace>
Kokkos::Impl::DeepCopy<Kokkos::Experimental::NVSHMEMSpace,
                       Kokkos::Experimental::NVSHMEMSpace,
                       ExecutionSpace>::DeepCopy(void *dst, const void *src,
                                                 size_t n) {
  Kokkos::Experimental::NVSHMEMSpace().fence();
  cudaMemcpy(dst, src, n, cudaMemcpyDefault);
}

template <typename ExecutionSpace>
Kokkos::Impl::DeepCopy<Kokkos::Experimental::NVSHMEMSpace,
                       Kokkos::Experimental::NVSHMEMSpace,
                       ExecutionSpace>::DeepCopy(const ExecutionSpace &exec,
                                                 void *dst, const void *src,
                                                 size_t n) {
  Kokkos::Experimental::NVSHMEMSpace().fence();
  cudaMemcpy(dst, src, n, cudaMemcpyDefault);
}

// Currently not invoked. We need a better local_deep_copy overload that
// recognizes consecutive memory regions
void local_deep_copy_get(void *dst, const void *src, size_t pe, size_t n) {
  nvshmem_getmem(dst, src, pe, n);
}

// Currently not invoked. We need a better local_deep_copy overload that
// recognizes consecutive memory regions
void local_deep_copy_put(void *dst, const void *src, size_t pe, size_t n) {
  nvshmem_putmem(dst, src, pe, n);
}

}  // namespace Impl
}  // namespace Kokkos
