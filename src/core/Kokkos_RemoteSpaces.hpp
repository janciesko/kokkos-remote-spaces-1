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
#ifndef KOKKOS_REMOTESPACES_HPP
#define KOKKOS_REMOTESPACES_HPP
#include <Kokkos_Core.hpp>

namespace Kokkos {
namespace Experimental {
enum RemoteSpaces_MemoryAllocationMode : int { Symmetric, Cached };
} // namespace Experimental
} // namespace Kokkos


#ifdef KOKKOS_ENABLE_CUDARDMASPACE
namespace Kokkos {
namespace Experimental {
class CudaRDMASpace;
} // namespace Experimental
} // namespace Kokkos
#include <RACERlib_Interface.hpp>
#include <Kokkos_CudaRDMASpace.hpp>
#endif

#ifdef KOKKOS_ENABLE_SHMEMSPACE
namespace Kokkos {
namespace Experimental {
class SHMEMSpace;
} // namespace Experimental
} // namespace Kokkos
#include <Kokkos_SHMEMSpace.hpp>
#endif

#ifdef KOKKOS_ENABLE_NVSHMEMSPACE
namespace Kokkos {
namespace Experimental {
class NVSHMEMSpace;
} // namespace Experimental
} // namespace Kokkos
#include <Kokkos_NVSHMEMSpace.hpp>
#endif

#ifdef KOKKOS_ENABLE_MPISPACE
namespace Kokkos {
namespace Experimental {
class MPISpace;
} // namespace Experimental
} // namespace Kokkos
#include <Kokkos_MPISpace.hpp>
#endif

namespace Kokkos {
namespace Experimental {

#ifdef KOKKOS_ENABLE_NVSHMEMSPACE
typedef NVSHMEMSpace DefaultRemoteMemorySpace;
#else
#ifdef KOKKOS_ENABLE_SHMEMSPACE
typedef SHMEMSpace DefaultRemoteMemorySpace;
#else
#ifdef KOKKOS_ENABLE_MPISPACE
typedef MPISpace DefaultRemoteMemorySpace;
#else
#ifdef KOKKOS_ENABLE_CUDARDMASPACE
typedef CudaRDMASpace DefaultRemoteMemorySpace;
#else
error "At least one remote space must be selected."
#endif
#endif
#endif
#endif

} // namespace Experimental=
} // namespace Kokkos

#endif // KOKKOS_RESMOTESPACES_HPP
