/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _qv_indexes_hpp_
#define _qv_indexes_hpp_

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace AER {
namespace QV {

// Type aliases
using uint_t = uint64_t;
using int_t = int64_t;
using reg_t = std::vector<uint_t>;
using indexes_t = std::unique_ptr<uint_t[]>;
template <size_t N>
using areg_t = std::array<uint_t, N>;

//============================================================================
// BIT MASKS and indexing
//============================================================================

const std::array<uint_t, 64> BITS{{1ULL,
                                   2ULL,
                                   4ULL,
                                   8ULL,
                                   16ULL,
                                   32ULL,
                                   64ULL,
                                   128ULL,
                                   256ULL,
                                   512ULL,
                                   1024ULL,
                                   2048ULL,
                                   4096ULL,
                                   8192ULL,
                                   16384ULL,
                                   32768ULL,
                                   65536ULL,
                                   131072ULL,
                                   262144ULL,
                                   524288ULL,
                                   1048576ULL,
                                   2097152ULL,
                                   4194304ULL,
                                   8388608ULL,
                                   16777216ULL,
                                   33554432ULL,
                                   67108864ULL,
                                   134217728ULL,
                                   268435456ULL,
                                   536870912ULL,
                                   1073741824ULL,
                                   2147483648ULL,
                                   4294967296ULL,
                                   8589934592ULL,
                                   17179869184ULL,
                                   34359738368ULL,
                                   68719476736ULL,
                                   137438953472ULL,
                                   274877906944ULL,
                                   549755813888ULL,
                                   1099511627776ULL,
                                   2199023255552ULL,
                                   4398046511104ULL,
                                   8796093022208ULL,
                                   17592186044416ULL,
                                   35184372088832ULL,
                                   70368744177664ULL,
                                   140737488355328ULL,
                                   281474976710656ULL,
                                   562949953421312ULL,
                                   1125899906842624ULL,
                                   2251799813685248ULL,
                                   4503599627370496ULL,
                                   9007199254740992ULL,
                                   18014398509481984ULL,
                                   36028797018963968ULL,
                                   72057594037927936ULL,
                                   144115188075855872ULL,
                                   288230376151711744ULL,
                                   576460752303423488ULL,
                                   1152921504606846976ULL,
                                   2305843009213693952ULL,
                                   4611686018427387904ULL,
                                   9223372036854775808ULL}};

const std::array<uint_t, 64> MASKS{{0ULL,
                                    1ULL,
                                    3ULL,
                                    7ULL,
                                    15ULL,
                                    31ULL,
                                    63ULL,
                                    127ULL,
                                    255ULL,
                                    511ULL,
                                    1023ULL,
                                    2047ULL,
                                    4095ULL,
                                    8191ULL,
                                    16383ULL,
                                    32767ULL,
                                    65535ULL,
                                    131071ULL,
                                    262143ULL,
                                    524287ULL,
                                    1048575ULL,
                                    2097151ULL,
                                    4194303ULL,
                                    8388607ULL,
                                    16777215ULL,
                                    33554431ULL,
                                    67108863ULL,
                                    134217727ULL,
                                    268435455ULL,
                                    536870911ULL,
                                    1073741823ULL,
                                    2147483647ULL,
                                    4294967295ULL,
                                    8589934591ULL,
                                    17179869183ULL,
                                    34359738367ULL,
                                    68719476735ULL,
                                    137438953471ULL,
                                    274877906943ULL,
                                    549755813887ULL,
                                    1099511627775ULL,
                                    2199023255551ULL,
                                    4398046511103ULL,
                                    8796093022207ULL,
                                    17592186044415ULL,
                                    35184372088831ULL,
                                    70368744177663ULL,
                                    140737488355327ULL,
                                    281474976710655ULL,
                                    562949953421311ULL,
                                    1125899906842623ULL,
                                    2251799813685247ULL,
                                    4503599627370495ULL,
                                    9007199254740991ULL,
                                    18014398509481983ULL,
                                    36028797018963967ULL,
                                    72057594037927935ULL,
                                    144115188075855871ULL,
                                    288230376151711743ULL,
                                    576460752303423487ULL,
                                    1152921504606846975ULL,
                                    2305843009213693951ULL,
                                    4611686018427387903ULL,
                                    9223372036854775807ULL}};

// index0 returns the integer representation of a number of bits set
// to zero inserted into an arbitrary bit string.
// Eg: for qubits 0,2 in a state k = ba ( ba = 00 => k=0, etc).
// indexes0([1], k) -> int(b0a)
// indexes0([1,3], k) -> int(0b0a)
// Example: k = 77  = 1001101 , qubits_sorted = [1,4]
// ==> output = 297 = 100101001 (with 0's put into places 1 and 4).
template <typename list_t>
uint_t index0(const list_t &qubits_sorted, const uint_t k);

// uint_t index0_MOSQ_CR(uint_t X_idx, uint_t Z_idx, const uint_t k);//SW
// uint_t index1_MOSQ_CR(uint_t X_idx, uint_t Z_idx, const uint_t k);//SW

// Return a std::unique_ptr to an array of of 2^N in ints
// each int corresponds to an N qubit bitstring for M-N qubit bits in state k,
// and the specified N qubits in states [0, ..., 2^N - 1]
// qubits_sorted must be sorted lowest to highest. Eg. {0, 1}.
// qubits specifies the location of the qubits in the returned strings.
// NOTE: since the return is a unique_ptr it cannot be copied.
// indexes returns the array of all bit values for the specified qubits
// (Eg: for qubits 0,2 in a state k = ba:
// indexes([1], [1], k) = [int(b0a), int(b1a)],
// if it were two qubits inserted say at 1,3 it would be:
// indexes([1,3], [1,3], k) -> [int(0b0a), int(0b1a), int(1b0a), (1b1a)]
// If the qubits were passed in reverse order it would swap qubit position in
// the list: indexes([3,1], [1,3], k) -> [int(0b0a), int(1b0a), int(0b1a),
// (1b1a)] Example: k=77, qubits=qubits_sorted=[1,4] ==>
// output=[297,299,313,315] input: k = 77  = 1001101 output[0]: 297 = 100101001
// (with 0's put into places 1 and 4). output[1]: 299 = 100101011 (with 0 put
// into place 1, and 1 put into place 4). output[2]: 313 = 100111001 (with 1 put
// into place 1, and 0 put into place 4). output[3]: 313 = 100111011 (with 1's
// put into places 1 and 4).
indexes_t indexes(const reg_t &qubits, const reg_t &qubits_sorted,
                  const uint_t k);
// indexes_t indexes_MOSQ_CR(size_t num_q, uint_t X_idx, uint_t Z_idx,
//                   const uint_t k);//SW

// As above but returns a fixed sized array of of 2^N in ints
template <size_t N>
areg_t<1ULL << N> indexes(const areg_t<N> &qs, const areg_t<N> &qubits_sorted,
                          const uint_t k);

/*******************************************************************************
 *
 * Implementations
 *
 ******************************************************************************/

template <typename list_t>
uint_t index0(const list_t &qubits_sorted, const uint_t k) {
  uint_t lowbits, retval = k;
  for (size_t j = 0; j < qubits_sorted.size(); j++) {
    lowbits = retval & MASKS[qubits_sorted[j]];
    retval >>= qubits_sorted[j];
    retval <<= qubits_sorted[j] + 1;
    retval |= lowbits;
  }
  return retval;
}

template <size_t N>
areg_t<1ULL << N> indexes(const areg_t<N> &qs, const areg_t<N> &qubits_sorted,
                          const uint_t k) {
  areg_t<1ULL << N> ret;
  ret[0] = index0(qubits_sorted, k);
  for (size_t i = 0; i < N; i++) {
    const auto n = BITS[i];
    const auto bit = BITS[qs[i]];
    for (size_t j = 0; j < n; j++)
      ret[n + j] = ret[j] | bit;
  }
  return ret;
}

inline indexes_t indexes(const reg_t &qubits, const reg_t &qubits_sorted,
                         const uint_t k) {
  const auto N = qubits_sorted.size();
  indexes_t ret(new uint_t[BITS[N]]);
  // Get index0
  ret[0] = index0(qubits_sorted, k);
  for (size_t i = 0; i < N; i++) {
    const auto n = BITS[i];
    const auto bit = BITS[qubits[i]];
    for (size_t j = 0; j < n; j++)
      ret[n + j] = ret[j] | bit;
  }
  return ret;
}

// template <typename list_t>
// uint_t index0_MOSQ_CR(uint_t X_idx, uint_t Z_idx, const uint_t k) {
//   uint_t lowbits, retval = k;
//   for (size_t j = 0; j < qubits_sorted.size(); j++) {
//     lowbits = retval & MASKS[qubits_sorted[j]];
//     retval >>= qubits_sorted[j];
//     retval <<= qubits_sorted[j] + 1;
//     retval |= lowbits;
//   }
//   return retval;
// }//SW

// inline indexes_t indexes_MOSQ_CR(size_t num_q, uint_t X_idx, uint_t Z_idx,
//                          const uint_t k) {
//   indexes_t ret(new uint_t[2]);
//   // Get index0
//   ret[0] = index0_MOSQ_CR(num_q, X_idx, k);
//   ret[1] = index1_MOSQ_CR(num_q, Z_idx, ret[0]);
//   return ret;
// }//SW

/*******************************************************************************
 *
 * LAMBDA FUNCTION TEMPLATES
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// State update
//------------------------------------------------------------------------------

template <typename Lambda>
inline void apply_lambda(const size_t start, const size_t stop,
                         const uint_t omp_threads, Lambda &&func) {

  if (omp_threads > 1) {
#pragma omp parallel num_threads(omp_threads)
    {
#pragma omp for
      for (int_t k = int_t(start); k < int_t(stop); k++) {
        std::forward<Lambda>(func)(k);
      }
    }
  } else {
    for (int_t k = int_t(start); k < int_t(stop); k++) {
      std::forward<Lambda>(func)(k);
    }
  }
}

template <typename Lambda, typename list_t>
inline void apply_lambda(const size_t start, const size_t stop,
                         const uint_t omp_threads, Lambda &&func,
                         const list_t &qubits) {

  const auto NUM_QUBITS = qubits.size();
  const int_t END = stop >> NUM_QUBITS;
  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());

  if (omp_threads > 1) {
#pragma omp parallel for num_threads(omp_threads)
    for (int_t k = int_t(start); k < END; k++) {
      // store entries touched by U
      const auto inds = indexes(qubits, qubits_sorted, k);
      std::forward<Lambda>(func)(inds);
    }
  } else {
    for (int_t k = int_t(start); k < END; k++) {
      // store entries touched by U
      const auto inds = indexes(qubits, qubits_sorted, k);
      std::forward<Lambda>(func)(inds);
    }
  }
}

template <typename Lambda, typename list_t>
inline void apply_lambda_MOSQ(const size_t start, const size_t stop,
                         const uint_t omp_threads, Lambda &&func,
                         const list_t &qubits) {
  if (omp_threads > 1) {
#pragma omp parallel for num_threads(omp_threads)
    for (int_t k = int_t(start); k < int_t(stop); k++) {
      //parity check
      bool parity = false;
      for (uint_t index : qubits) {
        parity ^= ((k >> index) & 1);
      }
      //forward lambda only for parity(ind) == 1
      if(parity) std::forward<Lambda>(func)(k);
    }
  } else {
    for (int_t k = int_t(start); k < int_t(stop); k++) {
      //parity check
      bool parity = false;
      for (uint_t index : qubits) {
        parity ^= ((k >> index) & 1);
      }
      //forward lambda only for parity(ind) == 1
      if(parity) std::forward<Lambda>(func)(k);
    }
  }
}//SW

template <typename Lambda, typename list_t, typename param_t>
inline void apply_lambda_MOSQ_CR(const size_t start, const size_t stop,
                         const uint_t omp_threads, Lambda &&func,
                         const list_t &qubits,
                         const param_t &mat0, const param_t &mat1, const param_t &mat2, const param_t &mat3,
                         uint_t X_idx, uint_t Y_idx, uint_t Z_idx, size_t num_qubits_) {

  // const auto NUM_QUBITS = qubits.size();
  // const int_t END = stop >> NUM_QUBITS;
  const int_t END = stop >> 1;
  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());

  //Y_idx 1 개수 세기
  uint_t num_Y = 0;
  uint_t temp = Y_idx;
  while (temp) {
    num_Y += (temp & 1);
    temp >>= 1;
  }
  // std::cout << "(X_idx): " << std::bitset<32>(X_idx) << std::endl;
  // std::cout << "(Y_idx): " << std::bitset<32>(Y_idx) << std::endl;
  // std::cout << "(Z_idx): " << std::bitset<32>(Z_idx) << std::endl;
  // std::cout << "(num_Y): " << num_Y << std::endl;

  if (omp_threads > 1) {
#pragma omp parallel num_threads(omp_threads)
    {
#pragma omp for
      for (int_t k = int_t(start); k < END; k += 1) {
        //k번째 2x2 matmul에 대한 index 2개를 찾는 과정
        //XY_idx의 가장 높은 1 위치를 찾아서 k 값의 그 위치에 0을 대입한 것이 idx0
        //idx0 값에서 XY_idx에 대해 XOR를 취해주면 idx1
        uint_t idx0, idx1, lowbits, XY_idx;
        XY_idx = X_idx ^ Y_idx;
        int highest_one_pos = -1;
        for (int i = sizeof(XY_idx) * 8 - 1; i >= 0; --i) {
            if ((XY_idx >> i) & 1) {
                highest_one_pos = i;
                break;
            }
        }
        idx0 = k;
        if(highest_one_pos > -1){
          lowbits = idx0 & MASKS[highest_one_pos];
          idx0 >>= highest_one_pos;
          idx0 <<= highest_one_pos + 1;
          idx0 |= lowbits;
        }
        idx1 = idx0 ^ XY_idx;
        indexes_t inds(new uint_t[2]);
        inds[0] = idx0;
        inds[1] = idx1;

        //matrix를 만드는 과정
        //idx0의 Y_idx와 Z_idx에 대한 1 개수를 세서 Y,Z에 저장하면 matrix의 각 항은
        //mat[0]=1+phase mat[1]=i^(num_Y-2*Y)*(-1)^(Z)*(1-phase)
        //mat[2]=i^(2*Y-num_Y)*(-1)^(Z)*(1-phase) mat[3]=1+phase
        uint_t Y_count = 0;
        uint_t Z_count = 0;
        uint_t Y_masked_value = Y_idx & idx0;
        uint_t Z_masked_value = Z_idx & idx0;
        for (size_t i = 0; i < sizeof(idx0) * 8; ++i) {
            Y_count += (Y_masked_value & 1);
            Y_masked_value >>= 1;
            Z_count += (Z_masked_value & 1);
            Z_masked_value >>= 1;
        }
        uint_t mat_sel = (num_Y - 2 * Y_count + 2 * Z_count) % 4;
        // if(k < 10){
        //   std::cout << "k: " << k << std::endl;
        //   std::cout << "(idx0): " << std::bitset<32>(idx0) << std::endl;
        //   std::cout << "(idx1): " << std::bitset<32>(idx1) << std::endl;
        //   std::cout << "(mat_sel): " << mat_sel << std::endl;
        // }
        if(mat_sel==0) std::forward<Lambda>(func)(inds, mat0);
        else if(mat_sel==1) std::forward<Lambda>(func)(inds, mat1);
        else if(mat_sel==2) std::forward<Lambda>(func)(inds, mat2);
        else if(mat_sel==3) std::forward<Lambda>(func)(inds, mat3);
      }
    }
  } else {
    for (int_t k = int_t(start); k < END; k += 1) {
        //k번째 2x2 matmul에 대한 index 2개를 찾는 과정
        //XY_idx의 가장 높은 1 위치를 찾아서 k 값의 그 위치에 0을 대입한 것이 idx0
        //idx0 값에서 XY_idx에 대해 XOR를 취해주면 idx1
        uint_t idx0, idx1, lowbits, XY_idx;
        XY_idx = X_idx ^ Y_idx;
        int highest_one_pos = -1;
        for (int i = sizeof(XY_idx) * 8 - 1; i >= 0; --i) {
            if ((XY_idx >> i) & 1) {
                highest_one_pos = i;
                break;
            }
        }
        idx0 = k;
        if(highest_one_pos > -1){
          lowbits = idx0 & MASKS[highest_one_pos];
          idx0 >>= highest_one_pos;
          idx0 <<= highest_one_pos + 1;
          idx0 |= lowbits;
        }
        idx1 = idx0 ^ XY_idx;
        indexes_t inds(new uint_t[2]);
        inds[0] = idx0;
        inds[1] = idx1;

        //matrix를 만드는 과정
        //idx0의 Y_idx와 Z_idx에 대한 1 개수를 세서 Y,Z에 저장하면 matrix의 각 항은
        //mat[0]=1+phase mat[1]=i^(num_Y-2*Y)*(-1)^(Z)*(1-phase)
        //mat[2]=i^(2*Y-num_Y)*(-1)^(Z)*(1-phase) mat[3]=1+phase
        uint_t Y_count = 0;
        uint_t Z_count = 0;
        uint_t Y_masked_value = Y_idx & idx0;
        uint_t Z_masked_value = Z_idx & idx0;
        for (size_t i = 0; i < sizeof(idx0) * 8; ++i) {
            Y_count += (Y_masked_value & 1);
            Y_masked_value >>= 1;
            Z_count += (Z_masked_value & 1);
            Z_masked_value >>= 1;
        }
        uint_t mat_sel = (num_Y - 2 * Y_count + 2 * Z_count) % 4;
        // if(k < 10){
        //   std::cout << "k: " << k << std::endl;
        //   std::cout << "(idx0): " << std::bitset<32>(idx0) << std::endl;
        //   std::cout << "(idx1): " << std::bitset<32>(idx1) << std::endl;
        //   std::cout << "(mat_sel): " << mat_sel << std::endl;
        // }
        if(mat_sel==0) std::forward<Lambda>(func)(inds, mat0);
        else if(mat_sel==1) std::forward<Lambda>(func)(inds, mat1);
        else if(mat_sel==2) std::forward<Lambda>(func)(inds, mat2);
        else if(mat_sel==3) std::forward<Lambda>(func)(inds, mat3);
    }
  }
}//SW

template <typename Lambda, typename list_t, typename param_t>
inline void apply_lambda(const size_t start, const size_t stop,
                         const size_t gap, const uint_t omp_threads,
                         Lambda &&func, const list_t &qubits,
                         const param_t &params) {

  const auto NUM_QUBITS = qubits.size();
  const int_t END = stop >> NUM_QUBITS;
  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());

  if (omp_threads > 1) {
#pragma omp parallel num_threads(omp_threads)
    {
#pragma omp for
      for (int_t k = int_t(start); k < END; k += gap) {
        const auto inds = indexes(qubits, qubits_sorted, k);
        std::forward<Lambda>(func)(inds, params);
      }
    }
  } else {
    for (int_t k = int_t(start); k < END; k += gap) {
      const auto inds = indexes(qubits, qubits_sorted, k);
      std::forward<Lambda>(func)(inds, params);
    }
  }
}

template <typename Lambda, typename list_t, typename param_t>
inline void apply_lambda(const size_t start, const size_t stop,
                         const uint_t omp_threads, Lambda &&func,
                         const list_t &qubits, const param_t &params) {
  apply_lambda(start, stop, 1, omp_threads, func, qubits, params);
}

//------------------------------------------------------------------------------
// Reduction Lambda
//------------------------------------------------------------------------------

template <typename Lambda>
inline std::complex<double>
apply_reduction_lambda(const size_t start, const size_t stop,
                       const uint_t omp_threads, Lambda &&func) {
  // Reduction variables
  double val_re = 0.;
  double val_im = 0.;
  if (omp_threads > 1) {
#pragma omp parallel reduction(+ : val_re, val_im) num_threads(omp_threads)
    {
#pragma omp for
      for (int_t k = int_t(start); k < int_t(stop); k++) {
        std::forward<Lambda>(func)(k, val_re, val_im);
      }
    } // end omp parallel
  } else {
    for (int_t k = int_t(start); k < int_t(stop); k++) {
      std::forward<Lambda>(func)(k, val_re, val_im);
    }
  }
  return std::complex<double>(val_re, val_im);
}

template <typename Lambda, typename list_t>
std::complex<double>
apply_reduction_lambda(const size_t start, const size_t stop,
                       const uint_t omp_threads, Lambda &&func,
                       const list_t &qubits) {

  const size_t NUM_QUBITS = qubits.size();
  const int_t END = stop >> NUM_QUBITS;
  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());

  // Reduction variables
  double val_re = 0.;
  double val_im = 0.;
  if (omp_threads > 1) {
#pragma omp parallel reduction(+ : val_re, val_im) num_threads(omp_threads)
    {
#pragma omp for
      for (int_t k = int_t(start); k < END; k++) {
        const auto inds = indexes(qubits, qubits_sorted, k);
        std::forward<Lambda>(func)(inds, val_re, val_im);
      }
    } // end omp parallel
  } else {
    for (int_t k = int_t(start); k < END; k++) {
      const auto inds = indexes(qubits, qubits_sorted, k);
      std::forward<Lambda>(func)(inds, val_re, val_im);
    }
  }
  return std::complex<double>(val_re, val_im);
}

template <typename Lambda, typename list_t, typename param_t>
std::complex<double>
apply_reduction_lambda(const size_t start, const size_t stop,
                       const uint_t omp_threads, Lambda &&func,
                       const list_t &qubits, const param_t &params) {

  const auto NUM_QUBITS = qubits.size();

  const int_t END = stop >> NUM_QUBITS;
  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());

  // Reduction variables
  double val_re = 0.;
  double val_im = 0.;
  if (omp_threads > 1) {
#pragma omp parallel reduction(+ : val_re, val_im) num_threads(omp_threads)
    {
#pragma omp for
      for (int_t k = int_t(start); k < END; k++) {
        const auto inds = indexes(qubits, qubits_sorted, k);
        std::forward<Lambda>(func)(inds, params, val_re, val_im);
      }
    } // end omp parallel
  } else {
    for (int_t k = int_t(start); k < END; k++) {
      const auto inds = indexes(qubits, qubits_sorted, k);
      std::forward<Lambda>(func)(inds, params, val_re, val_im);
    }
  }
  return std::complex<double>(val_re, val_im);
}

//------------------------------------------------------------------------------
} // end namespace QV
//------------------------------------------------------------------------------
} // namespace AER

//------------------------------------------------------------------------------
#endif // end module
