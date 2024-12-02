// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

// Debugging help
#define SIGMA_ASSERT(condition)                                                                                         \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            std::cerr << "ASSERT FAILED: " << #condition << " @ " << __FILE__ << " (" << __LINE__ << ")" << std::endl; \
        }                                                                                                              \
    }

// String expansion
#define _SIGMA_STRINGIZE(x) #x
#define SIGMA_STRINGIZE(x) _SIGMA_STRINGIZE(x)

// Join
#define _SIGMA_JOIN(M, N) M##N
#define SIGMA_JOIN(M, N) _SIGMA_JOIN(M, N)

// Check that double is 64 bits
static_assert(sizeof(double) == 8, "Require sizeof(double) == 8");

// Check that int is 32 bits
static_assert(sizeof(int) == 4, "Require sizeof(int) == 4");

// Check that unsigned long long is 64 bits
static_assert(sizeof(unsigned long long) == 8, "Require sizeof(unsigned long long) == 8");

// Bounds for bit-length of all coefficient moduli
#define SIGMA_MOD_BIT_COUNT_MAX 61
#define SIGMA_MOD_BIT_COUNT_MIN 2

// Bit-length of internally used coefficient moduli, e.g., auxiliary base in BFV
#define SIGMA_INTERNAL_MOD_BIT_COUNT 61

// Bounds for bit-length of user-defined coefficient moduli
#define SIGMA_USER_MOD_BIT_COUNT_MAX 60
#define SIGMA_USER_MOD_BIT_COUNT_MIN 2

// Bounds for bit-length of the plaintext modulus
#define SIGMA_PLAIN_MOD_BIT_COUNT_MAX SIGMA_USER_MOD_BIT_COUNT_MAX
#define SIGMA_PLAIN_MOD_BIT_COUNT_MIN SIGMA_USER_MOD_BIT_COUNT_MIN

// Bounds for number of coefficient moduli (no hard requirement)
#define SIGMA_COEFF_MOD_COUNT_MAX 256
#define SIGMA_COEFF_MOD_COUNT_MIN 1

// Bounds for polynomial modulus degree (no hard requirement)
#define SIGMA_POLY_MOD_DEGREE_MAX 131072
#define SIGMA_POLY_MOD_DEGREE_MIN 2

// Upper bound on the size of a ciphertext (cannot exceed 2^32 / poly_modulus_degree)
#define SIGMA_CIPHERTEXT_SIZE_MAX 16
#if SIGMA_CIPHERTEXT_SIZE_MAX > 0x100000000ULL / SIGMA_POLY_MOD_DEGREE_MAX
#error "SIGMA_CIPHERTEXT_SIZE_MAX is too large"
#endif
#define SIGMA_CIPHERTEXT_SIZE_MIN 2

// How many pairs of modular integers can we multiply and accumulate in a 128-bit data type
#if SIGMA_MOD_BIT_COUNT_MAX > 32
#define SIGMA_MULTIPLY_ACCUMULATE_MOD_MAX (1 << (128 - (SIGMA_MOD_BIT_COUNT_MAX << 1)))
#define SIGMA_MULTIPLY_ACCUMULATE_INTERNAL_MOD_MAX (1 << (128 - (SIGMA_INTERNAL_MOD_BIT_COUNT_MAX << 1)))
#define SIGMA_MULTIPLY_ACCUMULATE_USER_MOD_MAX (1 << (128 - (SIGMA_USER_MOD_BIT_COUNT_MAX << 1)))
#else
#define SIGMA_MULTIPLY_ACCUMULATE_MOD_MAX SIZE_MAX
#define SIGMA_MULTIPLY_ACCUMULATE_INTERNAL_MOD_MAX SIZE_MAX
#define SIGMA_MULTIPLY_ACCUMULATE_USER_MOD_MAX SIZE_MAX
#endif

// Detect system
#define SIGMA_SYSTEM_OTHER 1
#define SIGMA_SYSTEM_WINDOWS 2
#define SIGMA_SYSTEM_UNIX_LIKE 3

#if defined(_WIN32)
#define SIGMA_SYSTEM SIGMA_SYSTEM_WINDOWS
#elif defined(__linux__) || defined(__FreeBSD__) || defined(EMSCRIPTEN) || (defined(__APPLE__) && defined(__MACH__))
#define SIGMA_SYSTEM SIGMA_SYSTEM_UNIX_LIKE
#else
#define SIGMA_SYSTEM SIGMA_SYSTEM_OTHER
#error "Unsupported system"
#endif

// Detect compiler
#define SIGMA_COMPILER_MSVC 1
#define SIGMA_COMPILER_CLANG 2
#define SIGMA_COMPILER_GCC 3

#if defined(_MSC_VER)
#define SIGMA_COMPILER SIGMA_COMPILER_MSVC
#elif defined(__clang__)
#define SIGMA_COMPILER SIGMA_COMPILER_CLANG
#elif defined(__GNUC__) && !defined(__clang__)
#define SIGMA_COMPILER SIGMA_COMPILER_GCC
#else
#error "Unsupported compiler"
#endif

// MSVC support
#include "util/msvc.h"

// clang support
#include "util/clang.h"

// gcc support
#include "util/gcc.h"

// Create a true/false value for indicating debug mode
#ifdef SIGMA_DEBUG
#define SIGMA_DEBUG_V true
#else
#define SIGMA_DEBUG_V false
#endif

// Use std::byte as byte type
#ifdef SIGMA_USE_STD_BYTE
#include <cstddef>
namespace sigma
{
    using sigma_byte = std::byte;
} // namespace sigma
#else
namespace sigma
{
    enum class sigma_byte : unsigned char
    {
    };
} // namespace sigma
#endif

// Force inline
#ifndef SIGMA_FORCE_INLINE
#define SIGMA_FORCE_INLINE inline
#endif

// Use `if constexpr' from C++17
#ifdef SIGMA_USE_IF_CONSTEXPR
#define SIGMA_IF_CONSTEXPR if constexpr
#else
#define SIGMA_IF_CONSTEXPR if
#endif

// Use [[maybe_unused]] from C++17
#ifdef SIGMA_USE_MAYBE_UNUSED
#define SIGMA_MAYBE_UNUSED [[maybe_unused]]
#else
#define SIGMA_MAYBE_UNUSED
#endif

// Use [[nodiscard]] from C++17
#ifdef SIGMA_USE_NODISCARD
#define SIGMA_NODISCARD [[nodiscard]]
#else
#define SIGMA_NODISCARD
#endif

// C++14 does not have std::for_each_n so we use a custom implementation
#ifndef SIGMA_USE_STD_FOR_EACH_N
#define SIGMA_ITERATE sigma::util::sigma_for_each_n
#else
#define SIGMA_ITERATE std::for_each_n
#endif

// Allocate "size" bytes in memory and returns a sigma_byte pointer
// If SIGMA_USE_ALIGNED_ALLOC is defined, use _aligned_malloc and ::aligned_alloc (or std::malloc)
// Use `new sigma_byte[size]` as fallback
#ifndef SIGMA_MALLOC
#define SIGMA_MALLOC(size) (new sigma_byte[size])
#endif

// Deallocate a pointer in memory
// If SIGMA_USE_ALIGNED_ALLOC is defined, use _aligned_free or std::free
// Use `delete [] ptr` as fallback
#ifndef SIGMA_FREE
#define SIGMA_FREE(ptr) (delete[] ptr)
#endif

// Which random number generator to use by default
#define SIGMA_DEFAULT_PRNG_FACTORY SIGMA_JOIN(SIGMA_DEFAULT_PRNG, PRNGFactory)

// Which distribution to use for noise sampling: rounded Gaussian or Centered Binomial Distribution
#ifdef SIGMA_USE_GAUSSIAN_NOISE
#define SIGMA_NOISE_SAMPLER sample_poly_normal
#else
#define SIGMA_NOISE_SAMPLER sample_poly_cbd
#endif

// Use generic functions as (slower) fallback
#ifndef SIGMA_ADD_CARRY_UINT64
#define SIGMA_ADD_CARRY_UINT64(operand1, operand2, carry, result) add_uint64_generic(operand1, operand2, carry, result)
#endif

#ifndef SIGMA_SUB_BORROW_UINT64
#define SIGMA_SUB_BORROW_UINT64(operand1, operand2, borrow, result) \
    sub_uint64_generic(operand1, operand2, borrow, result)
#endif

#ifndef SIGMA_MULTIPLY_UINT64
#define SIGMA_MULTIPLY_UINT64(operand1, operand2, result128) multiply_uint64_generic(operand1, operand2, result128)
#endif

#ifndef SIGMA_DIVIDE_UINT128_UINT64
#define SIGMA_DIVIDE_UINT128_UINT64(numerator, denominator, result) \
    divide_uint128_uint64_inplace_generic(numerator, denominator, result);
#endif

#ifndef SIGMA_MULTIPLY_UINT64_HW64
#define SIGMA_MULTIPLY_UINT64_HW64(operand1, operand2, hw64) multiply_uint64_hw64_generic(operand1, operand2, hw64)
#endif

#ifndef SIGMA_MSB_INDEX_UINT64
#define SIGMA_MSB_INDEX_UINT64(result, value) get_msb_index_generic(result, value)
#endif

// Check whether an object is of expected type; this requires the type_traits header to be included
#define SIGMA_ASSERT_TYPE(obj, expected, message)                                                                    \
    do                                                                                                              \
    {                                                                                                               \
        static_assert(                                                                                              \
            std::is_same<decltype(obj), expected>::value,                                                           \
            "In " __FILE__ ":" SIGMA_STRINGIZE(__LINE__) " expected " SIGMA_STRINGIZE(expected) " (message: " message \
                                                                                              ")");                 \
    } while (false)

// This macro can be used to allocate a temporary buffer and create a PtrIter<T *> object pointing to it. This is
// convenient when the Pointer holding the buffer is not explicitly needed and the memory is only accessed through the
// iterator.
#define SIGMA_ALLOCATE_GET_PTR_ITER(name, type, size, pool)                               \
    auto SIGMA_JOIN(_sigma_temp_alloc_, __LINE__)(sigma::util::allocate<type>(size, pool)); \
    sigma::util::PtrIter<type *> name(SIGMA_JOIN(_sigma_temp_alloc_, __LINE__).get());

// This macro can be used to allocate a temporary buffer and create a StrideIter<T *> object pointing to it. This is
// convenient when the Pointer holding the buffer is not explicitly needed and the memory is only accessed through the
// iterator.
#define SIGMA_ALLOCATE_GET_STRIDE_ITER(name, type, size, stride, pool)                                                  \
    auto SIGMA_JOIN(_sigma_temp_alloc_, __LINE__)(sigma::util::allocate<type>(sigma::util::mul_safe(size, stride), pool)); \
    sigma::util::StrideIter<type *> name(SIGMA_JOIN(_sigma_temp_alloc_, __LINE__).get(), stride);

// This macro can be used to allocate a temporary buffer and create a PolyIter object pointing to it. This is convenient
// when the Pointer holding the buffer is not explicitly needed and the memory is only accessed through the iterator.
#define SIGMA_ALLOCATE_GET_POLY_ITER(name, poly_count, poly_modulus_degree, coeff_modulus_size, pool) \
    auto SIGMA_JOIN(_sigma_temp_alloc_, __LINE__)(                                                     \
        sigma::util::allocate_poly_array(poly_count, poly_modulus_degree, coeff_modulus_size, pool)); \
    sigma::util::PolyIter name(SIGMA_JOIN(_sigma_temp_alloc_, __LINE__).get(), poly_modulus_degree, coeff_modulus_size);

// This macro can be used to allocate a temporary buffer (set to zero) and create a PolyIter object pointing to it. This
// is convenient when the Pointer holding the buffer is not explicitly needed and the memory is only accessed through
// the iterator.
#define SIGMA_ALLOCATE_ZERO_GET_POLY_ITER(name, poly_count, poly_modulus_degree, coeff_modulus_size, pool) \
    auto SIGMA_JOIN(_sigma_temp_alloc_, __LINE__)(                                                          \
        sigma::util::allocate_zero_poly_array(poly_count, poly_modulus_degree, coeff_modulus_size, pool)); \
    sigma::util::PolyIter name(SIGMA_JOIN(_sigma_temp_alloc_, __LINE__).get(), poly_modulus_degree, coeff_modulus_size);

// This macro can be used to allocate a temporary buffer and create a RNSIter object pointing to it. This is convenient
// when the Pointer holding the buffer is not explicitly needed and the memory is only accessed through the iterator.
#define SIGMA_ALLOCATE_GET_RNS_ITER(name, poly_modulus_degree, coeff_modulus_size, pool) \
    auto SIGMA_JOIN(_sigma_temp_alloc_, __LINE__)(                                        \
        sigma::util::allocate_poly(poly_modulus_degree, coeff_modulus_size, pool));      \
    sigma::util::RNSIter name(SIGMA_JOIN(_sigma_temp_alloc_, __LINE__).get(), poly_modulus_degree);

// This macro can be used to allocate a temporary buffer (set to zero) and create a RNSIter object pointing to it. This
// is convenient when the Pointer holding the buffer is not explicitly needed and the memory is only accessed through
// the iterator.
#define SIGMA_ALLOCATE_ZERO_GET_RNS_ITER(name, poly_modulus_degree, coeff_modulus_size, pool) \
    auto SIGMA_JOIN(_sigma_temp_alloc_, __LINE__)(                                             \
        sigma::util::allocate_zero_poly(poly_modulus_degree, coeff_modulus_size, pool));      \
    sigma::util::RNSIter name(SIGMA_JOIN(_sigma_temp_alloc_, __LINE__).get(), poly_modulus_degree);

// This macro can be used to allocate a temporary buffer and create a CoeffIter object pointing to it. This is
// convenient when the Pointer holding the buffer is not explicitly needed and the memory is only accessed through the
// iterator.
#define SIGMA_ALLOCATE_GET_COEFF_ITER(name, poly_modulus_degree, pool)                                  \
    auto SIGMA_JOIN(_sigma_temp_alloc_, __LINE__)(sigma::util::allocate_uint(poly_modulus_degree, pool)); \
    sigma::util::CoeffIter name(SIGMA_JOIN(_sigma_temp_alloc_, __LINE__).get());

// This macro can be used to allocate a temporary buffer (set to zero) and create a CoeffIter object pointing to it.
// This is convenient when the Pointer holding the buffer is not explicitly needed and the memory is only accessed
// through the iterator.
#define SIGMA_ALLOCATE_ZERO_GET_COEFF_ITER(name, poly_modulus_degree, pool)                                  \
    auto SIGMA_JOIN(_sigma_temp_alloc_, __LINE__)(sigma::util::allocate_zero_uint(poly_modulus_degree, pool)); \
    sigma::util::CoeffIter name(SIGMA_JOIN(_sigma_temp_alloc_, __LINE__).get());

// Conditionally select the former if true and the latter if false
// This is a temporary solution that generates constant-time code with all compilers on all platforms.
#ifndef SIGMA_AVOID_BRANCHING
#define SIGMA_COND_SELECT(cond, if_true, if_false) (cond ? if_true : if_false)
#else
#define SIGMA_COND_SELECT(cond, if_true, if_false) \
    ((if_false) ^ ((~static_cast<uint64_t>(cond) + 1) & ((if_true) ^ (if_false))))
#endif
