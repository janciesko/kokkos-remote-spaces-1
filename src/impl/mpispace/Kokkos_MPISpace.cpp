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

#include <Kokkos_Core.hpp>
#include <Kokkos_MPISpace.hpp>
#include <csignal>
#include <mpi.h>

namespace Kokkos {
namespace Experimental {

MPI_Win MPISpace::current_win;
std::vector<MPI_Win> MPISpace::mpi_windows;

/* Default allocation mechanism */
MPISpace::MPISpace() : allocation_mode(Kokkos::Experimental::Symmetric) {}

void MPISpace::impl_set_allocation_mode(const int allocation_mode_) {
  allocation_mode = allocation_mode_;
}

void MPISpace::impl_set_extent(const int64_t extent_) { extent = extent_; }

void *MPISpace::allocate(const size_t arg_alloc_size) const {
  static_assert(sizeof(void *) == sizeof(uintptr_t),
                "Error sizeof(void*) != sizeof(uintptr_t)");

  static_assert(
      Kokkos::Impl::is_integral_power_of_two(Kokkos::Impl::MEMORY_ALIGNMENT),
      "Memory alignment must be power of two");

  void *ptr = 0;
  if (arg_alloc_size) {
    if (allocation_mode == Kokkos::Experimental::Symmetric) {
      current_win = MPI_WIN_NULL;
      MPI_Win_allocate(arg_alloc_size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &ptr,
                       &current_win);

      assert(current_win != MPI_WIN_NULL);

      int ret = MPI_Win_lock_all(MPI_MODE_NOCHECK, current_win);
      if (ret != MPI_SUCCESS) {
        Kokkos::abort("MPI window lock all failed.");
      }
      int i;
      for (i = 0; i < mpi_windows.size(); ++i) {
        if (mpi_windows[i] == MPI_WIN_NULL) break;
      }

      if (i == mpi_windows.size())
        mpi_windows.push_back(current_win);
      else
        mpi_windows[i] = current_win;
    } else {
      Kokkos::abort("MPISpace only supports symmetric allocation policy.");
    }
  }
  return ptr;
}

void MPISpace::deallocate(void *const, const size_t) const {
  int last_valid;
  for (last_valid = 0; last_valid < mpi_windows.size(); ++last_valid) {
    if (mpi_windows[last_valid] == MPI_WIN_NULL) break;
  }

  last_valid--;
  for (int i = 0; i < mpi_windows.size(); ++i) {
    if (mpi_windows[i] == current_win) {
      mpi_windows[i]          = mpi_windows[last_valid];
      mpi_windows[last_valid] = MPI_WIN_NULL;
      break;
    }
  }

  assert(current_win != MPI_WIN_NULL);
  MPI_Win_unlock_all(current_win);
  MPI_Win_free(&current_win);

  // We pass a mempory space instance do multiple Views thus
  // setting "current_win = MPI_WIN_NULL;" will result in a wrong handle if
  // subsequent view runs out of scope
  // Fixme: The following only works when views are allocated sequentially
  // We need a thread-safe map to associate views and windows

  if (last_valid != 0)
    current_win = mpi_windows[last_valid - 1];
  else
    current_win = MPI_WIN_NULL;
}

void MPISpace::fence() {
  for (int i = 0; i < mpi_windows.size(); i++) {
    if (mpi_windows[i] != MPI_WIN_NULL) {
      MPI_Win_flush_all(mpi_windows[i]);
    } else {
      break;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

size_t get_num_pes() {
  int n_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
  return n_ranks;
}

size_t get_my_pe() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

}  // namespace Experimental

namespace Impl {

Kokkos::Impl::DeepCopy<HostSpace, Kokkos::Experimental::MPISpace>::DeepCopy(
    void *dst, const void *src, size_t n) {
  Kokkos::Experimental::MPISpace().fence();
  memcpy(dst, src, n);
}

Kokkos::Impl::DeepCopy<Kokkos::Experimental::MPISpace, HostSpace>::DeepCopy(
    void *dst, const void *src, size_t n) {
  Kokkos::Experimental::MPISpace().fence();
  memcpy((char *)dst, (char *)src, n);
}

Kokkos::Impl::DeepCopy<Kokkos::Experimental::MPISpace,
                       Kokkos::Experimental::MPISpace>::DeepCopy(void *dst,
                                                                 const void
                                                                     *src,
                                                                 size_t n) {
  Kokkos::Experimental::MPISpace().fence();
  memcpy(dst, src, n);
}

template <typename ExecutionSpace>
Kokkos::Impl::DeepCopy<Kokkos::Experimental::MPISpace,
                       Kokkos::Experimental::MPISpace,
                       ExecutionSpace>::DeepCopy(void *dst, const void *src,
                                                 size_t n) {
  Kokkos::Experimental::MPISpace().fence();
  memcpy(dst, src, n);
}

template <typename ExecutionSpace>
Kokkos::Impl::DeepCopy<Kokkos::Experimental::MPISpace,
                       Kokkos::Experimental::MPISpace,
                       ExecutionSpace>::DeepCopy(const ExecutionSpace &exec,
                                                 void *dst, const void *src,
                                                 size_t n) {
  Kokkos::Experimental::MPISpace().fence();
  memcpy(dst, src, n);
}

// Currently not invoked. We need a better local_deep_copy overload that
// recognizes consecutive memory regions
void local_deep_copy_get(void *dst, const void *src, size_t pe, size_t n) {
  // TBD
}

// Currently not invoked. We need a better local_deep_copy overload that
// recognizes consecutive memory regions
void local_deep_copy_put(void *dst, const void *src, size_t pe, size_t n) {
  // TBD
}

}  // namespace Impl
}  // namespace Kokkos
