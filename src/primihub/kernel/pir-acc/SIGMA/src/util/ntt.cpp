// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "util/ntt.h"
#include "util/uintarith.h"
#include "util/uintarithsmallmod.h"
#include <algorithm>
#include "kernelutils.cuh"

#ifdef SIGMA_USE_INTEL_HEXL
#include "memorymanager.h"
#include "util/iterator.h"
#include "util/locks.h"
#include "util/pointer.h"
#include <unordered_map>
#include "hexl/hexl.hpp"
#endif

using namespace std;
using namespace sigma::util;
using namespace sigma::kernel_util;

#ifdef SIGMA_USE_INTEL_HEXL
namespace intel
{
    namespace hexl
    {
        // Single threaded SIGMA allocator adapter
        template <>
        struct NTT::AllocatorAdapter<sigma::MemoryPoolHandle>
            : public AllocatorInterface<NTT::AllocatorAdapter<sigma::MemoryPoolHandle>>
        {
            AllocatorAdapter(sigma::MemoryPoolHandle handle) : handle_(std::move(handle))
            {}

            ~AllocatorAdapter()
            {}

            // interface implementations
            void *allocate_impl(std::size_t bytes_count)
            {
                cache_.push_back(static_cast<sigma::util::MemoryPool &>(handle_).get_for_byte_count(bytes_count));
                return cache_.back().get();
            }

            void deallocate_impl(void *p, SIGMA_MAYBE_UNUSED std::size_t n)
            {
                auto it = std::remove_if(
                    cache_.begin(), cache_.end(),
                    [p](const sigma::util::Pointer<sigma::sigma_byte> &sigma_pointer) { return p == sigma_pointer.get(); });

#ifdef SIGMA_DEBUG
                if (it == cache_.end())
                {
                    throw std::logic_error("Inconsistent single-threaded allocator cache");
                }
#endif
                cache_.erase(it, cache_.end());
            }

        private:
            sigma::MemoryPoolHandle handle_;
            std::vector<sigma::util::Pointer<sigma::sigma_byte>> cache_;
        };

        // Thread safe policy
        struct SimpleThreadSafePolicy
        {
            SimpleThreadSafePolicy() : m_ptr(std::make_unique<std::mutex>())
            {}

            std::unique_lock<std::mutex> locker()
            {
                if (!m_ptr)
                {
                    throw std::logic_error("accessing a moved object");
                }
                return std::unique_lock<std::mutex>{ *m_ptr };
            };

        private:
            std::unique_ptr<std::mutex> m_ptr;
        };

        // Multithreaded SIGMA allocator adapter
        template <>
        struct NTT::AllocatorAdapter<sigma::MemoryPoolHandle, SimpleThreadSafePolicy>
            : public AllocatorInterface<NTT::AllocatorAdapter<sigma::MemoryPoolHandle, SimpleThreadSafePolicy>>
        {
            AllocatorAdapter(sigma::MemoryPoolHandle handle, SimpleThreadSafePolicy &&policy)
                : handle_(std::move(handle)), policy_(std::move(policy))
            {}

            ~AllocatorAdapter()
            {}
            // interface implementations
            void *allocate_impl(std::size_t bytes_count)
            {
                {
                    // to prevent inline optimization with deadlock
                    auto accessor = policy_.locker();
                    cache_.push_back(static_cast<sigma::util::MemoryPool &>(handle_).get_for_byte_count(bytes_count));
                    return cache_.back().get();
                }
            }

            void deallocate_impl(void *p, SIGMA_MAYBE_UNUSED std::size_t n)
            {
                {
                    // to prevent inline optimization with deadlock
                    auto accessor = policy_.locker();
                    auto it = std::remove_if(
                        cache_.begin(), cache_.end(), [p](const sigma::util::Pointer<sigma::sigma_byte> &sigma_pointer) {
                            return p == sigma_pointer.get();
                        });

#ifdef SIGMA_DEBUG
                    if (it == cache_.end())
                    {
                        throw std::logic_error("Inconsistent multi-threaded allocator cache");
                    }
#endif
                    cache_.erase(it, cache_.end());
                }
            }

        private:
            sigma::MemoryPoolHandle handle_;
            SimpleThreadSafePolicy policy_;
            std::vector<sigma::util::Pointer<sigma::sigma_byte>> cache_;
        };
    } // namespace hexl

    namespace sigma_ext
    {
        struct HashPair
        {
            template <class T1, class T2>
            std::size_t operator()(const std::pair<T1, T2> &p) const
            {
                auto hash1 = std::hash<T1>{}(std::get<0>(p));
                auto hash2 = std::hash<T2>{}(std::get<1>(p));
                return hash_combine(hash1, hash2);
            }

            static std::size_t hash_combine(std::size_t lhs, std::size_t rhs)
            {
                lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
                return lhs;
            }
        };

        /**
        Returns a HEXL NTT object corresponding to the given parameters.

        @param[in] N The polynomial modulus degree
        @param[in] modulus The modulus
        @param[in] root The root of unity
        */
        static intel::hexl::NTT &get_ntt(size_t N, uint64_t modulus, uint64_t root)
        {
            static unordered_map<pair<uint64_t, uint64_t>, hexl::NTT, HashPair> ntt_cache_;

            static sigma::util::ReaderWriterLocker ntt_cache_locker_;

            pair<uint64_t, uint64_t> key{ N, modulus };

            // Enable shared access to NTT already present
            {
                sigma::util::ReaderLock reader_lock(ntt_cache_locker_.acquire_read());
                auto ntt_it = ntt_cache_.find(key);
                if (ntt_it != ntt_cache_.end())
                {
                    return ntt_it->second;
                }
            }

            // Deal with NTT not yet present
            sigma::util::WriterLock write_lock(ntt_cache_locker_.acquire_write());

            // Check ntt_cache for value (may be added by another thread)
            auto ntt_it = ntt_cache_.find(key);
            if (ntt_it == ntt_cache_.end())
            {
                hexl::NTT ntt(N, modulus, root, sigma::MemoryManager::GetPool(), hexl::SimpleThreadSafePolicy{});
                ntt_it = ntt_cache_.emplace(move(key), move(ntt)).first;
            }
            return ntt_it->second;
        }

        /**
        Computes the forward negacyclic NTT from the given parameters.

        @param[in,out] operand The data on which to compute the NTT.
        @param[in] N The polynomial modulus degree
        @param[in] modulus The modulus
        @param[in] root The root of unity
        @param[in] input_mod_factor Bounds the input data to the range [0, input_mod_factor * modulus)
        @param[in] output_mod_factor Bounds the output data to the range [0, output_mod_factor * modulus)
        */
        static void compute_forward_ntt(
            sigma::util::CoeffIter operand, std::size_t N, std::uint64_t modulus, std::uint64_t root,
            std::uint64_t input_mod_factor, std::uint64_t output_mod_factor)
        {
            get_ntt(N, modulus, root).ComputeForward(operand, operand, input_mod_factor, output_mod_factor);
        }

        /**
        Computes the inverse negacyclic NTT from the given parameters.

        @param[in,out] operand The data on which to compute the NTT.
        @param[in] N The polynomial modulus degree
        @param[in] modulus The modulus
        @param[in] root The root of unity
        @param[in] input_mod_factor Bounds the input data to the range [0, input_mod_factor * modulus)
        @param[in] output_mod_factor Bounds the output data to the range [0, output_mod_factor * modulus)
        */
        static void compute_inverse_ntt(
            sigma::util::CoeffIter operand, std::size_t N, std::uint64_t modulus, std::uint64_t root,
            std::uint64_t input_mod_factor, std::uint64_t output_mod_factor)
        {
            get_ntt(N, modulus, root).ComputeInverse(operand, operand, input_mod_factor, output_mod_factor);
        }

    } // namespace sigma_ext
} // namespace intel
#endif

namespace sigma
{
    namespace util
    {
        NTTTables::NTTTables(int coeff_count_power, const Modulus &modulus, MemoryPoolHandle pool) : pool_(move(pool))
        {
#ifdef SIGMA_DEBUG
            if (!pool_)
            {
                throw invalid_argument("pool is uninitialized");
            }
#endif
            initialize(coeff_count_power, modulus);
        }

        void NTTTables::initialize(int coeff_count_power, const Modulus &modulus)
        {
#ifdef SIGMA_DEBUG
            if ((coeff_count_power < get_power_of_two(SIGMA_POLY_MOD_DEGREE_MIN)) ||
                coeff_count_power > get_power_of_two(SIGMA_POLY_MOD_DEGREE_MAX))
            {
                throw invalid_argument("coeff_count_power out of range");
            }
#endif
            coeff_count_power_ = coeff_count_power;
            coeff_count_ = size_t(1) << coeff_count_power_;
            modulus_ = modulus;
            // We defer parameter checking to try_minimal_primitive_root(...)
            if (!try_minimal_primitive_root(2 * coeff_count_, modulus_, root_))
            {
                throw invalid_argument("invalid modulus");
            }
            if (!try_invert_uint_mod(root_, modulus_, inv_root_))
            {
                throw invalid_argument("invalid modulus");
            }

#ifdef SIGMA_USE_INTEL_HEXL
            // Pre-compute HEXL NTT object
            intel::sigma_ext::get_ntt(coeff_count_, modulus.value(), root_);
#endif

            // Populate tables with powers of root in specific orders.
            root_powers_ = HostArray<MultiplyUIntModOperand>(coeff_count_);
            MultiplyUIntModOperand root;
            root.set(root_, modulus_);
            uint64_t power = root_;
            for (size_t i = 1; i < coeff_count_; i++)
            {
                root_powers_[reverse_bits(i, coeff_count_power_)].set(power, modulus_);
                power = multiply_uint_mod(power, root, modulus_);
            }
            root_powers_[0].set(static_cast<uint64_t>(1), modulus_);
            device_root_powers_ = DeviceArray(root_powers_);

            inv_root_powers_ = HostArray<MultiplyUIntModOperand>(coeff_count_);
            root.set(inv_root_, modulus_);
            power = inv_root_;
            for (size_t i = 1; i < coeff_count_; i++)
            {
                inv_root_powers_[reverse_bits(i - 1, coeff_count_power_) + 1].set(power, modulus_);
                power = multiply_uint_mod(power, root, modulus_);
            }
            inv_root_powers_[0].set(static_cast<uint64_t>(1), modulus_);
            device_inv_root_powers_ = DeviceArray(inv_root_powers_);


            /*const MultiplyUIntModOperand *roots = device_root_powers_.get();
            for (size_t i = 1; i < coeff_count_; i++)
            {
                cout<<roots[i].operand<<" ";
            }*/


            // Compute n^(-1) modulo q.
            uint64_t degree_uint = static_cast<uint64_t>(coeff_count_);
            if (!try_invert_uint_mod(degree_uint, modulus_, inv_degree_modulo_.operand))
            {
                throw invalid_argument("invalid modulus");
            }
            inv_degree_modulo_.set_quotient(modulus_);

            mod_arith_lazy_ = ModArithLazy(modulus_);
            ntt_handler_ = NTTHandler(mod_arith_lazy_);
        }

        class NTTTablesCreateIter
        {
        public:
            using value_type = NTTTables;
            using pointer = void;
            using reference = value_type;
            using difference_type = ptrdiff_t;

            // LegacyInputIterator allows reference to be equal to value_type so we can construct
            // the return objects on the fly and return by value.
            using iterator_category = input_iterator_tag;

            // Require default constructor
            NTTTablesCreateIter()
            {}

            // Other constructors
            NTTTablesCreateIter(int coeff_count_power, vector<Modulus> modulus, MemoryPoolHandle pool)
                : coeff_count_power_(coeff_count_power), modulus_(modulus), pool_(move(pool))
            {}

            // Require copy and move constructors and assignments
            NTTTablesCreateIter(const NTTTablesCreateIter &copy) = default;

            NTTTablesCreateIter(NTTTablesCreateIter &&source) = default;

            NTTTablesCreateIter &operator=(const NTTTablesCreateIter &assign) = default;

            NTTTablesCreateIter &operator=(NTTTablesCreateIter &&assign) = default;

            // Dereferencing creates NTTTables and returns by value
            inline value_type operator*() const
            {
                return { coeff_count_power_, modulus_[index_], pool_ };
            }

            // Pre-increment
            inline NTTTablesCreateIter &operator++() noexcept
            {
                index_++;
                return *this;
            }

            // Post-increment
            inline NTTTablesCreateIter operator++(int) noexcept
            {
                NTTTablesCreateIter result(*this);
                index_++;
                return result;
            }

            // Must be EqualityComparable
            inline bool operator==(const NTTTablesCreateIter &compare) const noexcept
            {
                return (compare.index_ == index_) && (coeff_count_power_ == compare.coeff_count_power_);
            }

            inline bool operator!=(const NTTTablesCreateIter &compare) const noexcept
            {
                return !operator==(compare);
            }

            // Arrow operator must be defined
            value_type operator->() const
            {
                return **this;
            }

        private:
            size_t index_ = 0;
            int coeff_count_power_ = 0;
            vector<Modulus> modulus_;
            MemoryPoolHandle pool_;
        };

        void CreateNTTTables(//ntttablse有modulus.size个
            int coeff_count_power, const vector<Modulus> &modulus, Pointer<NTTTables> &tables, MemoryPoolHandle pool)
        {
            if (!pool)
            {
                throw invalid_argument("pool is uninitialized");
            }
            if (!modulus.size())
            {
                throw invalid_argument("invalid modulus");
            }
            // coeff_count_power and modulus will be validated by "allocate"

            NTTTablesCreateIter iter(coeff_count_power, modulus, pool);
            tables = allocate(iter, modulus.size(), pool);
        }

        void ntt_negacyclic_harvey_lazy(CoeffIter operand, const NTTTables &tables)
        {
#ifdef SIGMA_USE_INTEL_HEXL
            size_t N = size_t(1) << tables.coeff_count_power();
            uint64_t p = tables.modulus().value();
            uint64_t root = tables.get_root();

            intel::sigma_ext::compute_forward_ntt(operand, N, p, root, 4, 4);
#else
            //cout<<"a";
            tables.ntt_handler().transform_to_rev(
                operand.ptr(), tables.coeff_count_power(), tables.get_from_root_powers());
#endif
        }

        void ntt_negacyclic_harvey(CoeffIter operand, const NTTTables &tables)
        {
#ifdef SIGMA_USE_INTEL_HEXL
            size_t N = size_t(1) << tables.coeff_count_power();
            uint64_t p = tables.modulus().value();
            uint64_t root = tables.get_root();

            intel::sigma_ext::compute_forward_ntt(operand, N, p, root, 4, 1);
#else

            //std::size_t* n = (size_t*)malloc(sizeof(size_t));
            //KernelProvider::retrieve(n, (size_t*)tables.coeff_count_power(), size_t(1));
            //size_t temp  = size_t(1)<< *n;
            uint64_t* d_temp=KernelProvider::malloc<uint64_t>(4096);
            KernelProvider::copy(d_temp, operand.ptr(), 4096);
            sigma::kernel_util::g_ntt_negacyclic_harvey(d_temp, 4096 ,tables);
            KernelProvider::retrieve(operand.ptr(), d_temp, 4096);
            KernelProvider::free(d_temp);
            // Finally maybe we need to reduce every coefficient modulo q, but we
            // know that they are in the range [0, 4q).
            // Since word size is controlled this is fast.
            /*std::uint64_t modulus = tables.modulus().value();
             ntt_negacyclic_harvey_lazy(operand, tables);

            std::uint64_t two_times_modulus = modulus * 2;
            std::size_t n = std::size_t(1) << tables.coeff_count_power();

            SIGMA_ITERATE(operand, n, [&](auto &I) {
                // Note: I must be passed to the lambda by reference.
                if (I >= two_times_modulus)
                {
                    I -= two_times_modulus;
                }
                if (I >= modulus)
                {
                    I -= modulus;
                }
            });*/
#endif
        }

        void inverse_ntt_negacyclic_harvey_lazy(CoeffIter operand, const NTTTables &tables)
        {
#ifdef SIGMA_USE_INTEL_HEXL
            size_t N = size_t(1) << tables.coeff_count_power();
            uint64_t p = tables.modulus().value();
            uint64_t root = tables.get_root();
            intel::sigma_ext::compute_inverse_ntt(operand, N, p, root, 2, 2);
#else
            MultiplyUIntModOperand inv_degree_modulo = tables.inv_degree_modulo();
            tables.ntt_handler().transform_from_rev(
                operand.ptr(), tables.coeff_count_power(), tables.get_from_inv_root_powers(), &inv_degree_modulo);
#endif
        }

        void inverse_ntt_negacyclic_harvey(CoeffIter operand, const NTTTables &tables)
        {
#ifdef SIGMA_USE_INTEL_HEXL
            size_t N = size_t(1) << tables.coeff_count_power();
            uint64_t p = tables.modulus().value();
            uint64_t root = tables.get_root();
            intel::sigma_ext::compute_inverse_ntt(operand, N, p, root, 2, 1);
#else
           /* uint64_t* d_temp=KernelProvider::malloc<uint64_t>(4096);
            KernelProvider::copy(d_temp, operand.ptr(), 4096);
            sigma::kernel_util::g_inv_ntt_negacyclic_harvey(d_temp, 4096 ,tables);
            KernelProvider::retrieve(operand.ptr(), d_temp, 4096);
            KernelProvider::free(d_temp);*/

            inverse_ntt_negacyclic_harvey_lazy(operand, tables);
            std::uint64_t modulus = tables.modulus().value();
            std::size_t n = std::size_t(1) << tables.coeff_count_power();

            // Final adjustments; compute a[j] = a[j] * n^{-1} mod q.
            // We incorporated the final adjustment in the butterfly. Only need to reduce here.
            SIGMA_ITERATE(operand, n, [&](auto &I) {
                // Note: I must be passed to the lambda by reference.
                if (I >= modulus)
                {
                    I -= modulus;
                }
            });
#endif
        }
    } // namespace util
} // namespace sigma
