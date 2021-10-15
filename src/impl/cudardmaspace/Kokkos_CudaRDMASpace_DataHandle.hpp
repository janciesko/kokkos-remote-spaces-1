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

#ifndef KOKKOS_REMOTESPACES_CUDARDMA_DATAHANDLE_HPP
#define KOKKOS_REMOTESPACES_CUDARDMA_DATAHANDLE_HPP

namespace Kokkos {
namespace Impl {


template <class T, class Traits> struct DataHandle {
  using worker_type = Kokkos::Experimental::RACERlib::RdmaScatterGatherWorker<T>;
  using engine_type = Kokkos::Experimental::RACERlib::Engine<T>;

  T *ptr;
  engine_type *engine;
  worker_type *worker;
  KOKKOS_INLINE_FUNCTION
  DataHandle() : ptr(NULL), worker(NULL) {}

  KOKKOS_INLINE_FUNCTION
  DataHandle(void *ptr_, engine_type *e_, worker_type *w_)
      : ptr(reinterpret_cast<T*>(ptr_)), engine(e_), worker(w_) {}

  KOKKOS_INLINE_FUNCTION
  DataHandle(DataHandle<T, Traits> const &arg)
      : ptr(arg.ptr), engine(arg.engine), worker(arg.worker) {}

  template <typename iType>
  KOKKOS_INLINE_FUNCTION DataElement<T, Traits>
  operator()(const int &pe, const iType &i) const {
    DataElement<T, Traits> element(ptr, worker, pe, i);
    return element;
  }

  KOKKOS_INLINE_FUNCTION
  T *operator+(size_t &offset) const { return ptr + offset; }
};

template <class Traits>
struct ViewDataHandle<
    Traits,
    typename std::enable_if<
        std::is_same<typename Traits::specialize,
                     Kokkos::Experimental::RemoteSpaceSpecializeTag>::value /*&&
        RemoteSpaces_MemoryTraits<typename Traits::memory_traits>::is_cached*/>::
        type> {

  using value_type = typename Traits::value_type;
  using handle_type = DataHandle<value_type, Traits>;
  using return_type = DataElement<value_type, Traits>;
  using track_type = Kokkos::Impl::SharedAllocationTracker;

  KOKKOS_INLINE_FUNCTION
  static handle_type assign(value_type *arg_data_ptr,
                            track_type const &arg_tracker) {
    auto *record =
        arg_tracker.template get_record<Kokkos::Experimental::CudaRDMASpace>();
    return handle_type(arg_data_ptr, record->RACERlib_get_engine()->sgw);
  }
  template <class SrcHandleType>
  KOKKOS_INLINE_FUNCTION handle_type operator=(SrcHandleType const &rhs) {
    return handle_type(rhs);
  }
};

} // namespace Impl
} // namespace Kokkos

#endif // KOKKOS_REMOTESPACES_CUDARDMA_DATAHANDLE_HPP
