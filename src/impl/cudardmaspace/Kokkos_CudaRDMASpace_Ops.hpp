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

#ifndef KOKKOS_REMOTESPACES_CUDARDMA_OPS_HPP
#define KOKKOS_REMOTESPACES_CUDARDMA_OPS_HPP

#include <type_traits>

namespace Kokkos {
namespace Impl {

template <class T, class Traits, typename Enable = void>
struct DataElement {};

// Cached Operators (Requires RACERlib)
template <class T, class Traits>
struct DataElement<
    T, Traits,
    typename std::enable_if<RemoteSpaces_MemoryTraits<
        typename Traits::memory_traits>::is_cached>::type> {

  using Worker_t = Kokkos::Experimental::RACERlib::RdmaScatterGatherWorker<T>;
  Worker_t *worker;
  typedef const T const_value_type;
  typedef T non_const_value_type;
  uint32_t offset;
  T *ptr;
  int pe;

  KOKKOS_INLINE_FUNCTION
  DataElement(T *ptr_, Worker_t *w_, int pe_, int i_)
      : ptr(ptr_), worker(w_), pe(pe_), offset(i_) {}

  KOKKOS_INLINE_FUNCTION
  T request(int pe, uint32_t offset) const {
    bool nonlocal = pe != worker->my_rank;

    if (nonlocal) {
      void *shm_ptr = worker->direct_ptrs[pe];
      if (shm_ptr) {
        T *t = (T *)shm_ptr;
        return volatile_load_2(&t[offset]);
      }
      return worker->request(pe, offset);
    } else {
      return ptr[offset];
    }
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator+(const_value_type &val) const {
    return request(pe, offset) + val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator-(const_value_type &val) const {
    return request(pe, offset) - val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator*(const_value_type &val) const {
    return request(pe, offset) * val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator/(const_value_type &val) const {
    return request(pe, offset) / val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator%(const_value_type &val) const {
    return request(pe, offset) & val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator!() const { return !request(pe, offset); }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator&&(const_value_type &val) const {
    return request(pe, offset) && val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator||(const_value_type &val) const {
    return request(pe, offset) || val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator&(const_value_type &val) const {
    return request(pe, offset) & val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator|(const_value_type &val) const {
    return request(pe, offset) | val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator^(const_value_type &val) const {
    return request(pe, offset) & val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator~() const { return ~request(pe, offset); }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator<<(const unsigned int &val) const {
    return request(pe, offset) << val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator>>(const unsigned int &val) const {
    return request(pe, offset) >> val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator==(const_value_type &val) const {
    return request(pe, offset) == val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator!=(const_value_type &val) const {
    return request(pe, offset) != val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator>=(const_value_type &val) const {
    return request(pe, offset) >= val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator<=(const_value_type &val) const {
    return request(pe, offset) <= val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator<(const_value_type &val) const {
    return request(pe, offset) < val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator>(const_value_type &val) const {
    return request(pe, offset) > val;
  }

  KOKKOS_INLINE_FUNCTION
  operator const_value_type() const { return request(pe, offset); }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator=(const_value_type &val) const {
    if (worker->rank == pe) {
      ptr[offset] = val;
    }
    return val;
  }
};

} // namespace Impl
} // namespace Kokkos

#endif // KOKKOS_REMOTESPACES_CUDARDMA_OPS_HPP