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

#include <Kokkos_Core.hpp>
#include <Kokkos_CudaRDMASpace.hpp>

namespace Kokkos {
namespace Experimental {

/* Default allocation mechanism */
CudaRDMASpace::CudaRDMASpace() : allocation_mode(Kokkos::Experimental::Symmetric) {}

void CudaRDMASpace::impl_set_allocation_mode(const int allocation_mode_) {
  allocation_mode = allocation_mode_;
}

void CudaRDMASpace::impl_set_extent(const int64_t extent_) { extent = extent_; }

void *CudaRDMASpace::allocate(const size_t arg_alloc_size) const {
  static_assert(sizeof(void *) == sizeof(uintptr_t),
                "Error sizeof(void*) != sizeof(uintptr_t)");

  static_assert(
      Kokkos::Impl::is_integral_power_of_two(Kokkos::Impl::MEMORY_ALIGNMENT),
      "Memory alignment must be power of two");

  void *ptr = 0;
  if (arg_alloc_size) {
    if (allocation_mode == Kokkos::Experimental::Symmetric) {
      int num_pes = get_num_pes();
      int my_id = get_my_pe();
      cuda_safe(cuMemAlloc((CUdeviceptr *)&ptr, arg_alloc_size));
    } else {
      Kokkos::abort("CudaRDMASpace only supports symmetric allocation policy.");
    }
  }
  return ptr;
}

void CudaRDMASpace::deallocate(void *const arg_alloc_ptr, const size_t) const {
  cudaFree(arg_alloc_ptr);
}

void CudaRDMASpace::fence() {
  //Kokkos::fence(); //FIXME: check if the Kokkos fence is required
  MPI_Barrier(MPI_COMM_WORLD);  
}

size_t get_my_pe() {
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  return my_rank;
}

size_t get_num_pes() {
  int num_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  return num_ranks;
}

KOKKOS_FUNCTION
size_t get_block_round_up(size_t size) {
  size_t n_pe, block;
  n_pe = get_num_pes();
  block = (size % get_num_pes()) ? (size + n_pe) / n_pe : size / n_pe;
  return block;
}

KOKKOS_FUNCTION
size_t get_block_round_down(size_t size) {
  size_t n_pe, block;
  n_pe = get_num_pes();
  block = size / n_pe;
  block = block  == 0 ? 1 : block;
  return block;
}

KOKKOS_FUNCTION
size_t get_block(size_t size) { return get_block_round_up(size); }

} // namespace Experimental

namespace Impl 
{
  
Kokkos::Impl::DeepCopy<HostSpace, Kokkos::Experimental::CudaRDMASpace>
::DeepCopy(void *dst, const void *src, size_t n) {
    Kokkos::Experimental::CudaRDMASpace().fence();
    cudaMemcpy(dst, src, n, cudaMemcpyDefault);
}

Kokkos::Impl::DeepCopy<Kokkos::Experimental::CudaRDMASpace, HostSpace>
::DeepCopy(void *dst, const void *src, size_t n) {
    Kokkos::Experimental::CudaRDMASpace().fence();
    cudaMemcpy(dst, src, n, cudaMemcpyDefault);
}

template <typename ExecutionSpace>
Kokkos::Impl::DeepCopy<Kokkos::Experimental::CudaRDMASpace, Kokkos::Experimental::CudaRDMASpace,
              ExecutionSpace>:: 
DeepCopy(void *dst, const void *src, size_t n) { 
  Kokkos::Experimental::CudaRDMASpace().fence();
  cudaMemcpy(dst, src, n, cudaMemcpyDefault);
}

template <typename ExecutionSpace>
Kokkos::Impl::DeepCopy<Kokkos::Experimental::CudaRDMASpace, Kokkos::Experimental::CudaRDMASpace,
              ExecutionSpace>:: 
DeepCopy(const ExecutionSpace &exec, void *dst, const void *src, size_t n) {
  Kokkos::Experimental::CudaRDMASpace().fence();
  cudaMemcpy(dst, src, n, cudaMemcpyDefault);
}

// Currently not invoked. We need a better local_deep_copy overload that
// recognizes consecutive memory regions
void local_deep_copy_get(void *dst, const void *src, size_t pe, size_t n) {
  //tbd
}

// Currently not invoked. We need a better local_deep_copy overload that
// recognizes consecutive memory regions
void local_deep_copy_put(void *dst, const void *src, size_t pe, size_t n) {
  //tbd
}

} // namespace Impl
} // namespace Kokkos
