/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Jan Ciesko (jciesko@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_NVSHMEMSPACE_HPP
#define KOKKOS_NVSHMEMSPACE_HPP

#include <cstring>
#include <iosfwd>
#include <string>
#include <typeinfo>

#include <Kokkos_Core.hpp>

#include <Kokkos_RemoteSpaces.hpp>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

namespace Kokkos {
namespace Experimental {

class RemoteSpaceSpecializeTag {};

class NVSHMEMSpace {
public:
#if defined(KOKKOS_ENABLE_RACERLIB)

  // Refactor to use base class for default type
  // Move to AllocationRecord (likely)
 // Kokkos::Experimental::RACERlib::Engine<double> e;
#endif

#if defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_OPENMP)
  using execution_space = Kokkos::OpenMP;
#elif defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_THREADS)
  using execution_space = Kokkos::Threads;
#elif defined(KOKKOS_ENABLE_CUDA)
  using execution_space = Kokkos::Cuda;
#elif defined(KOKKOS_ENABLE_OPENMP)
  using execution_space = Kokkos::OpenMP;
#elif defined(KOKKOS_ENABLE_THREADS)
  using execution_space = Kokkos::Threads;
#elif defined(KOKKOS_ENABLE_SERIAL)
  using execution_space = Kokkos::Serial;
#else
#error                                                                         \
    "At least one of the following host execution spaces must be defined: Kokkos::OpenMP, Kokkos::Threads, Kokkos::Qthreads, or Kokkos::Serial.  \
        You might be seeing this message if you disabled the Kokkos::Serial device explicitly using the Kokkos_ENABLE_Serial:BOOL=OFF \
        CMake option, but did not enable any of the other host execution space devices."
#endif

  using memory_space = NVSHMEMSpace;
  using device_type = Kokkos::Device<execution_space, memory_space>;
  using size_type = size_t;

  NVSHMEMSpace();
  NVSHMEMSpace(NVSHMEMSpace &&rhs) = default;
  NVSHMEMSpace(const NVSHMEMSpace &rhs) = default;
  NVSHMEMSpace &operator=(NVSHMEMSpace &&) = default;
  NVSHMEMSpace &operator=(const NVSHMEMSpace &) = default;
  ~NVSHMEMSpace() = default;

  explicit NVSHMEMSpace(const MPI_Comm &);

  void *allocate(const size_t arg_alloc_size) const;

  void deallocate(void *const arg_alloc_ptr, const size_t arg_alloc_size) const;

  void *allocate(const int *gids, const int &arg_local_alloc_size) const;

  void deallocate(const int *gids, void *const arg_alloc_ptr,
                  const size_t arg_alloc_size) const;

  void set_allocation_mode(RemoteSpaces_MemoryAllocationMode mode) {
    allocation_mode = mode;
  }

  int my_rank;
  int num_ranks;

  /**\brief Return Name of the MemorySpace */
  static constexpr const char *name() { return m_name; }

  void fence();

  void mem_fence();

  int64_t extent;

  void impl_set_extent(int64_t N);

private:
  int allocation_mode;
  static constexpr const char *m_name = "NVSHMEM";
  friend class Kokkos::Impl::SharedAllocationRecord<
      Kokkos::Experimental::NVSHMEMSpace, void>;
};

size_t get_num_pes();
size_t get_my_pe();

} // namespace Experimental
} // namespace Kokkos

namespace Kokkos {
namespace Impl {

template <> struct DeepCopy<HostSpace, Kokkos::Experimental::NVSHMEMSpace> {
  DeepCopy(void *dst, const void *src, size_t);
};

template <> struct DeepCopy<Kokkos::Experimental::NVSHMEMSpace, HostSpace> {
  DeepCopy(void *dst, const void *src, size_t);
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::Experimental::NVSHMEMSpace,
                Kokkos::Experimental::NVSHMEMSpace, ExecutionSpace> {
  DeepCopy(void *dst, const void *src, size_t n);
  DeepCopy(const ExecutionSpace &exec, void *dst, const void *src, size_t n);
};

template <>
struct MemorySpaceAccess<Kokkos::Experimental::NVSHMEMSpace,
                         Kokkos::Experimental::NVSHMEMSpace> {
  enum { assignable = true };
  enum { accessible = true };
  enum { deepcopy = false };
};

template <>
struct MemorySpaceAccess<Kokkos::HostSpace,
                         Kokkos::Experimental::NVSHMEMSpace> {
  enum { assignable = false };
  enum { accessible = false };
  enum { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::CudaSpace,
                         Kokkos::Experimental::NVSHMEMSpace> {
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy = true };
};

} // namespace Impl
} // namespace Kokkos


template<class T>
KOKKOS_INLINE_FUNCTION
T volatile_load_2(T * ptr, const T val)
{
  return Kokkos::atomic_fetch_add(ptr, val);
}


template<class T>
KOKKOS_INLINE_FUNCTION
T volatile_load_2(T * ptr)
{
  return Kokkos::atomic_fetch_add(ptr, 0);
}

template<class T>
KOKKOS_INLINE_FUNCTION
void volatile_store_2(T * ptr, const T val)
{
  Kokkos::atomic_fetch_add(ptr, val);
}


#include <Kokkos_RemoteSpaces_DeepCopy.hpp>
#include <Kokkos_RemoteSpaces_LocalDeepCopy.hpp>
#include <Kokkos_RemoteSpaces_Options.hpp>
#include <Kokkos_RemoteSpaces_ViewLayout.hpp>
#include <Kokkos_RemoteSpaces_ViewOffset.hpp>
#include <Kokkos_RemoteSpaces_ViewMapping.hpp>
#include <Kokkos_NVSHMEMSpace_Ops.hpp>
#include <Kokkos_NVSHMEMSpace_AllocationRecord.hpp>
#include <Kokkos_NVSHMEMSpace_DataHandle.hpp>
#include <Kokkos_NVSHMEMSpace_ViewTraits.hpp>

#endif // #define KOKKOS_NVSHMEMSPACE_HPP
