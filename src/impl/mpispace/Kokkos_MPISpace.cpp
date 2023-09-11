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
  return allocate("[unlabeled]", arg_alloc_size);
}

void *MPISpace::allocate(const char *arg_label, const size_t arg_alloc_size,
                         const size_t

                             arg_logical_size) const {
  return impl_allocate(arg_label, arg_alloc_size, arg_logical_size);
}

void *MPISpace::impl_allocate(
    const char *arg_label, const size_t arg_alloc_size,
    const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  const size_t reported_size =
      (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
  static_assert(sizeof(void *) == sizeof(uintptr_t),
                "Error sizeof(void*) != sizeof(uintptr_t)");

  static_assert(
      Kokkos::Impl::is_integral_power_of_two(Kokkos::Impl::MEMORY_ALIGNMENT),
      "Memory alignment must be power of two");

  constexpr uintptr_t alignment      = Kokkos::Impl::MEMORY_ALIGNMENT;
  constexpr uintptr_t alignment_mask = alignment - 1;

  void *ptr = nullptr;

  if (arg_alloc_size) {
    // Over-allocate to and round up to guarantee proper alignment.
    size_t size_padded = arg_alloc_size + sizeof(void *) + alignment;

    if (allocation_mode == Kokkos::Experimental::Symmetric) {
      current_win = MPI_WIN_NULL;
      MPI_Win_allocate(size_padded, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &ptr,
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
  using MemAllocFailure =
      Kokkos::Impl::Experimental::RemoteSpacesMemoryAllocationFailure;
  using MemAllocFailureMode = Kokkos::Impl::Experimental::
      RemoteSpacesMemoryAllocationFailure::FailureMode;

  if ((ptr == nullptr) || (reinterpret_cast<uintptr_t>(ptr) == ~uintptr_t(0))
      // MPI_Win_allocate may allocate non-alligned to
      // Kokkos::Impl::MEMORY_ALIGNMENT
      // ||
      // (reinterpret_cast<uintptr_t>(ptr) & alignment_mask)*/
  ) {
    MemAllocFailureMode failure_mode =
        MemAllocFailureMode::AllocationNotAligned;
    if (ptr == nullptr) {
      failure_mode = MemAllocFailureMode::OutOfMemoryError;
    }
    MemAllocFailure::AllocationMechanism alloc_mec =
        MemAllocFailure::AllocationMechanism::MPIWINALLOC;
    throw MemAllocFailure(arg_alloc_size, alignment, failure_mode, alloc_mec);
  }

  if (Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::allocateData(arg_handle, arg_label, ptr, reported_size);
  }
  return ptr;
}

void MPISpace::deallocate(void *const arg_alloc_ptr,
                          const size_t arg_alloc_size) const {
  deallocate("[unlabeled]", arg_alloc_ptr, arg_alloc_size);
}

void MPISpace::deallocate(const char *arg_label, void *const arg_alloc_ptr,
                          const size_t arg_alloc_size,
                          const size_t

                              arg_logical_size) const {
  impl_deallocate(arg_label, arg_alloc_ptr, arg_alloc_size, arg_logical_size);
}

void MPISpace::impl_deallocate(
    const char *arg_label, void *const arg_alloc_ptr,
    const size_t arg_alloc_size, const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  if (arg_alloc_ptr) {
    Kokkos::fence("HostSpace::impl_deallocate before free");
    fence();
    size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    if (Kokkos::Profiling::profileLibraryLoaded()) {
      Kokkos::Profiling::deallocateData(arg_handle, arg_label, arg_alloc_ptr,
                                        reported_size);
    }

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
}

void MPISpace::fence() const {
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
