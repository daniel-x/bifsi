/*
 * bifsi.h
 * published: 2022-09-04
 * last change: 2022-09-04
 *
 * Copyright 2022 Daniel Strecker
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

/*
 * bifsi is a general purpose library for multi precision fixed size integers,
 * aka. big fixed size ints (= bifsi), where the size is determined at compile
 * time.
 * It's intended for use on devices using the CUDA programming language and
 * compute model, i.e. for GPUs, though it is completely compilable and usable
 * in standard C++ environments.
 * Most operations are branchless (= there are no if statements) to achieve
 * complete thread coherence. This is a design goal for best performance on
 * GPUs. The lack of branching and the resulting lack of ending an individual
 * operation early reduces performance on CPUs. Therefore this library, though
 * technically usable, is not well suited for use on CPUs. A good and also more
 * versatile alternative for CPUs is cpp_int of boost::multiprecision, which
 * you can find at
 * https://www.boost.org/doc/libs/1_80_0/libs/multiprecision/doc/html/boost_multiprecision/tut/ints/cpp_int.html .
 * (TOC: https://www.boost.org/doc/libs/1_80_0/libs/multiprecision/doc/html/index.html )
 */

#ifndef BIFSI_H_
#define BIFSI_H_

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <algorithm>

#ifndef __NVCC__
#define __host__
#define __device__
#define __global__
#endif

namespace {

/*
 * Template for selecting an integer type twice the size of some other integer
 * type.
 */
template<class T> struct type_tag {
	using type = T;
};
template<class > struct twice_size;
template<class T> using twice_size_t = typename twice_size<T>::type;

template<> struct twice_size<uint8_t> : type_tag<uint16_t> {
};
template<> struct twice_size<uint16_t> : type_tag<uint32_t> {
};
template<> struct twice_size<uint32_t> : type_tag<uint64_t> {
};
#ifdef __GNUC__
template<> struct twice_size<uint64_t> : type_tag<unsigned __int128> {
};
#endif

template<typename UINT_T>
twice_size_t<UINT_T> f(UINT_T x) {
	twice_size_t<UINT_T> t;

	// ... do something with x and t

	return t;
}

} /* anonymous namespace */

namespace bifsi {

/*
 * Integer type of the elements (limbs) that a big int uses to store its
 * magnitude. Usually, the best choice for this type is the second largest type
 * that is provided by the C++ environment. It's not the largest type, because
 * some operation implementations, e.g. multiplication, need a type that is
 * twice the size of this element type.
 */
//typedef uint8_t el_t;
//typedef uint16_t el_t;
typedef uint32_t el_t;
//typedef uint64_t el_t;

/*
 * Integer type twice the size of the element type. Many C++ environments
 * provide an emulated type which is twice the size of the biggest type
 * supported natively by the hardware. This is usually the right choice.
 */
typedef twice_size_t<el_t> tw_t;

const size_t EL_SIZE_IN_BITS = sizeof(el_t) * 8;

const size_t TW_SIZE_IN_BITS = sizeof(tw_t) * 8;

///**
// * A portable constexpr popcount (number of 1 bits).
// */
//template<typename INT_T>
//inline constexpr uint32_t popcount(INT_T value) {
//	uint32_t result = 0;
//
//	for (size_t i = 0; i < sizeof(INT_T) * 8; i++) {
//		result += ((value >> i) & 1);
//	}
//
//	return result;
//}

__host__ __device__
inline constexpr int32_t ceil_constexpr(double d) {
	int32_t i = static_cast<int32_t>(d);

	i = (static_cast<double>(i) == d) ? i : i + ((d > 0.0) ? 1 : 0);

	return i;
}

/*
 * clz function for various types. Result is arbitrary for x = 0.
 */
template<typename INT_T>
__host__ __device__
inline int wrapped_builtin_clz(const INT_T &x) {
	if (sizeof(INT_T) == sizeof(unsigned char)) {
		return __builtin_clz((unsigned int) x) - 24;

	} else if (sizeof(INT_T) == sizeof(unsigned short)) {
		return __builtin_clz((unsigned int) x) - 16;

	} else if (sizeof(INT_T) == sizeof(unsigned int)) {
		return __builtin_clz(x);

	} else if (sizeof(INT_T) == sizeof(unsigned long)) {
		return __builtin_clzl(x);

	} else if (sizeof(INT_T) == sizeof(unsigned long long)) {
		return __builtin_clzll(x);

	} else {
		static_assert(sizeof(INT_T) <= sizeof(unsigned long long), "no suitable __builtin_clz intrinsic for INT_T");
		return 0; // never reached
	}
}

template<typename INT_T>
__host__ __device__
inline int number_of_leading_0_bits(const INT_T &x) {
	return (x == 0) ? (sizeof(INT_T) * 8) : wrapped_builtin_clz(x);
}

template<typename INT_T>
__host__ __device__
inline int bitlen(const INT_T &x) {
	return (x == 0) ? sizeof(INT_T) * 8 : sizeof(INT_T) * 8 - wrapped_builtin_clz(x);
}

#include <cxxabi.h>
template<typename T>
std::string type_name() {
	std::string result = typeid(T).name();

	int status;
	char *demangled_name = abi::__cxa_demangle(result.c_str(), NULL, NULL, &status);

	if (status == 0) {
		result = demangled_name;
		std::free(demangled_name);
	}

	return result;
}

/**
 * Fails if INT_T is not an integer or if it is neither signed nor
 * unsigned. If INT_T passes the checks, then this function translates to
 * 0 instructions, because these are compile time checks.
 */
template<typename INT_T>
inline static constexpr void static_assert_singed_or_unsigned_int_type() {
	static_assert(std::is_integral_v<INT_T>, "INT_T is not an integer type");
	static_assert(std::is_unsigned_v<INT_T> || std::is_signed_v<INT_T>, "INT_T is neither signed nor unsigned");
}

template<typename UINT_T>
inline static constexpr void assert_unsigned_int_type() {
	static_assert(std::is_integral_v<UINT_T>, "UINT_T is not an integer type");

	// making the check for unsignedness at runtime allows for easier code
	// using this function and it is completely optimized away when not
	// failing, because it's already known at compile time if the check
	// fails or succeeds.
	assert(std::is_unsigned_v<UINT_T>);
}

template<typename UINT_T>
inline std::string uint_to_string(const UINT_T &x) {
	constexpr double LOG_BASE10_2 = 0.301029995663981195; // constexpr std::log10(2.0);

	constexpr size_t UINT_SIZE_IN_BITS = sizeof(x) * 8;

	constexpr int32_t MAX_DIGITS = ceil_constexpr(UINT_SIZE_IN_BITS * LOG_BASE10_2);

	char result[MAX_DIGITS + 1];

	UINT_T tmp = x;
	size_t i = 0;

	do {
		result[i] = '0' + (char) (tmp % 10);
		tmp /= 10;

		i++;
	} while (tmp != 0);

	result[i] = '\0';

	std::reverse(result, result + i);

	return std::string(result);
}

/**
 * This class represents an unsigned big fixed size int with the specified size
 * in number of bits. bui means simply big unsigned int. This is the central
 * class of this library.
 */
template<size_t SIZE_IN_BITS>
class bui {
private:
	static_assert(sizeof(tw_t) == 2 * sizeof(el_t), "constraint not fulfilled: sizeof(tw_t) == 2 * sizeof(el_t)");
	// static_assert(sizeof(el_t) == 2 * sizeof(ha_t), "constraint not fulfilled: sizeof(el_t) = 2 * sizeof(ha_t)");
	static_assert(EL_SIZE_IN_BITS % 8 == 0, "constraint not fulfilled: EL_SIZE_IN_BITS % 8 == 0");
	static_assert(SIZE_IN_BITS > 0, "constraint not fulfilled: SIZE_IN_BITS > 0");
	static_assert(SIZE_IN_BITS % EL_SIZE_IN_BITS == 0, "constraint not fulfilled: SIZE_IN_BITS % EL_SIZE_IN_BITS == 0");

public:
	/*
	 * Number of elements that this big int uses to store its magnitude.
	 */
	static const size_t SIZE_IN_ELS = SIZE_IN_BITS / EL_SIZE_IN_BITS;

	/*
	 * Array of data elements which store the magnitude of this big int.
	 */
	el_t el[SIZE_IN_ELS];

	/*
	 * Constructs a new object of this type. The value is not initialized, so
	 * it's arbitrary. This constructor is empty and if fully optimized by the
	 * compiler, it translates to zero instructions.
	 */
	__host__ __device__
	inline bui() {
	}

	template<typename INT_T>
	__host__ __device__
	inline constexpr bui(const INT_T &value) :
			bui() {
		static_assert_singed_or_unsigned_int_type<INT_T>();

		if (std::is_unsigned_v<INT_T>) {
			this->set_uint(value);
		} else {
			this->set_uint(static_cast<std::make_unsigned_t<INT_T>>(value));
		}
	}

	__host__ __device__
	inline constexpr bui(const char *value) :
			bui() {
		this->set(value);
	}

	__host__ __device__
	inline constexpr el_t to_el_t() const {
		return el[0];
	}

	__host__ __device__
	inline constexpr tw_t to_tw_t() const {
		if (SIZE_IN_ELS == 1) {
			return el[0];
		} else {
			return el[0] | (((tw_t) el[1]) << EL_SIZE_IN_BITS);
		}
	}

	__host__ __device__
	inline explicit constexpr operator el_t() const {
		return to_el_t();
	}

	__host__ __device__
	inline explicit constexpr operator tw_t() const {
		return to_tw_t();
	}

private:
	template<typename UINT_T>
	__host__ __device__
	inline constexpr void set_uint(const UINT_T &value) {
		assert_unsigned_int_type<UINT_T>();

		if (sizeof(el_t) >= sizeof(UINT_T) || SIZE_IN_ELS == 1) {
			el[0] = value;
			set_zero_starting_at_el<1>();

		} else if (2 * sizeof(el_t) >= sizeof(UINT_T) || SIZE_IN_ELS == 2) {
			el[0] = value;
			el[1] = value >> 1 * EL_SIZE_IN_BITS;
			set_zero_starting_at_el<2>();

		} else if (4 * sizeof(el_t) >= sizeof(UINT_T) || SIZE_IN_ELS <= 4) {
			el[0] = value;
			el[1] = value >> 1 * EL_SIZE_IN_BITS;
			el[2] = value >> 2 * EL_SIZE_IN_BITS;

			if (SIZE_IN_ELS == 3) {
				set_zero_starting_at_el<3>();
			} else {
				el[3] = value >> 3 * EL_SIZE_IN_BITS;
				set_zero_starting_at_el<4>();
			}

		} else if (8 * sizeof(el_t) >= sizeof(UINT_T) || SIZE_IN_ELS <= 8) {
			el[0] = value;
			el[1] = value >> 1 * EL_SIZE_IN_BITS;
			el[2] = value >> 2 * EL_SIZE_IN_BITS;
			el[3] = value >> 3 * EL_SIZE_IN_BITS;
			el[4] = value >> 4 * EL_SIZE_IN_BITS;

			if (SIZE_IN_ELS == 5) {
				set_zero_starting_at_el<5>();

			} else if (SIZE_IN_ELS == 6) {
				el[5] = value >> 5 * EL_SIZE_IN_BITS;
				set_zero_starting_at_el<6>();

			} else if (SIZE_IN_ELS == 7) {
				el[5] = value >> 5 * EL_SIZE_IN_BITS;
				el[6] = value >> 6 * EL_SIZE_IN_BITS;
				set_zero_starting_at_el<7>();

			} else {
				el[5] = value >> 5 * EL_SIZE_IN_BITS;
				el[6] = value >> 6 * EL_SIZE_IN_BITS;
				el[7] = value >> 7 * EL_SIZE_IN_BITS;
				set_zero_starting_at_el<8>();

			}

		} else {
			static_assert(8 * sizeof(el_t) >= sizeof(uint64_t), "no suitable way to assign from UINT_T");

		}
	}

	template<typename UINT_T>
	__host__ __device__
	inline constexpr UINT_T as_uint() const {
		assert_unsigned_int_type<UINT_T>();

		if (1 * sizeof(el_t) >= sizeof(UINT_T) || SIZE_IN_ELS == 1) {
			return el[0];

		} else if (2 * sizeof(el_t) >= sizeof(UINT_T) || SIZE_IN_ELS == 2) {
			return el[0] | //
					(((UINT_T) el[1]) << 1 * EL_SIZE_IN_BITS);

		} else if (4 * sizeof(el_t) >= sizeof(UINT_T) || SIZE_IN_ELS <= 4) {
			if (SIZE_IN_ELS == 3) {
				return el[0] | //
						(((UINT_T) el[1]) << 1 * EL_SIZE_IN_BITS) | //
						(((UINT_T) el[2]) << 2 * EL_SIZE_IN_BITS);

			} else {
				return el[0] | //
						(((UINT_T) el[1]) << 1 * EL_SIZE_IN_BITS) | //
						(((UINT_T) el[2]) << 2 * EL_SIZE_IN_BITS) | //
						(((UINT_T) el[3]) << 3 * EL_SIZE_IN_BITS);

			}

		} else if (8 * sizeof(el_t) >= sizeof(UINT_T) || SIZE_IN_ELS <= 8) {
			if (SIZE_IN_ELS == 5) {
				return el[0] | //
						(((UINT_T) el[1]) << 1 * EL_SIZE_IN_BITS) | //
						(((UINT_T) el[2]) << 2 * EL_SIZE_IN_BITS) | //
						(((UINT_T) el[3]) << 3 * EL_SIZE_IN_BITS) | //
						(((UINT_T) el[4]) << 4 * EL_SIZE_IN_BITS);

			} else if (SIZE_IN_ELS == 6) {
				return el[0] | //
						(((UINT_T) el[1]) << 1 * EL_SIZE_IN_BITS) | //
						(((UINT_T) el[2]) << 2 * EL_SIZE_IN_BITS) | //
						(((UINT_T) el[3]) << 3 * EL_SIZE_IN_BITS) | //
						(((UINT_T) el[4]) << 4 * EL_SIZE_IN_BITS) | //
						(((UINT_T) el[5]) << 5 * EL_SIZE_IN_BITS);

			} else if (SIZE_IN_ELS == 7) {
				return el[0] | //
						(((UINT_T) el[1]) << 1 * EL_SIZE_IN_BITS) | //
						(((UINT_T) el[2]) << 2 * EL_SIZE_IN_BITS) | //
						(((UINT_T) el[3]) << 3 * EL_SIZE_IN_BITS) | //
						(((UINT_T) el[4]) << 4 * EL_SIZE_IN_BITS) | //
						(((UINT_T) el[5]) << 5 * EL_SIZE_IN_BITS) | //
						(((UINT_T) el[6]) << 6 * EL_SIZE_IN_BITS);

			} else {
				return el[0] | //
						(((UINT_T) el[1]) << 1 * EL_SIZE_IN_BITS) | //
						(((UINT_T) el[2]) << 2 * EL_SIZE_IN_BITS) | //
						(((UINT_T) el[3]) << 3 * EL_SIZE_IN_BITS) | //
						(((UINT_T) el[4]) << 4 * EL_SIZE_IN_BITS) | //
						(((UINT_T) el[5]) << 5 * EL_SIZE_IN_BITS) | //
						(((UINT_T) el[6]) << 6 * EL_SIZE_IN_BITS) | //
						(((UINT_T) el[7]) << 7 * EL_SIZE_IN_BITS);

			}

		} else {
			// the condition in the static assert is to make the compiler happy
			// in all cases when the static assert isn't even reached.
			static_assert(8 * sizeof(el_t) >= sizeof(UINT_T), "no suitable way to convert to UINT_T");
			return 0; // will never be reached

		}
	}

	__host__ __device__
	inline constexpr bui& set_from_str(const char *str) {
		size_t len = std::strlen(str);

		set_zero();

#ifdef __NVCC__
		#pragma unroll
		#endif
		for (size_t i = 0; i < len; i++) {
			char c = str[i];

			assert(c >= '0');
			assert(c <= '9');

			*this *= 10;

			*this += (el_t) (c - '0');
		}

		return *this;
	}

	/*
	 * Returns true if, and only if, b represents a different value than this
	 * object.
	 *
	 * This function calls
	 *
	 * assert(
	 *   sizeof(el_t) % sizeof(UINT_T) == 0 ||
	 *   sizeof(UINT_T) % sizeof(el_t) == 0
	 * ),
	 *
	 * which fails if the condition is not met, because the implementation is
	 * not ready for such cases.
	 */
	template<typename UINT_T>
	__host__ __device__
	inline constexpr bool operator_neq_uint(const UINT_T &b) const {
		assert_unsigned_int_type<UINT_T>();

		bool result;

		if (sizeof(el_t) >= sizeof(UINT_T)) {
			assert(sizeof(el_t) % sizeof(UINT_T) == 0);

			result = (to_el_t() != b);
			result |= is_nonzero_starting_at_el<1>();

		} else {
			assert(sizeof(UINT_T) % sizeof(el_t) == 0);

			result = (as_uint<UINT_T>() != b);
			result |= is_nonzero_starting_at_el<sizeof(UINT_T) / sizeof(el_t)>();
		}

		return result;
	}

	/*
	 * Returns true if, and only if, b represents a the same value as this
	 * object.
	 *
	 * This function calls
	 *
	 * assert(
	 *   sizeof(el_t) % sizeof(UINT_T) == 0 ||
	 *   sizeof(UINT_T) % sizeof(el_t) == 0
	 * ),
	 *
	 * which fails if the condition is not met, because the implementation is
	 * not ready for such cases.
	 */
	template<typename UINT_T>
	__host__ __device__
	inline constexpr bool operator_eq_uint(const UINT_T &b) const {
		assert_unsigned_int_type<UINT_T>();

		bool result;

		if (sizeof(el_t) >= sizeof(UINT_T)) {
			assert(sizeof(el_t) % sizeof(UINT_T) == 0);

			result = (to_el_t() == b);
			result &= is_zero_starting_at_el<1>();

		} else {
			assert(sizeof(UINT_T) % sizeof(el_t) == 0);

			result = (as_uint<UINT_T>() == b);
			result &= is_zero_starting_at_el<sizeof(UINT_T) / sizeof(el_t)>();
		}

		return result;
	}

	/*
	 * Returns true if, and only if, b represents a value less than this
	 * object.
	 *
	 * This function calls
	 *
	 * assert(
	 *   sizeof(el_t) % sizeof(UINT_T) == 0 ||
	 *   sizeof(UINT_T) % sizeof(el_t) == 0
	 * ),
	 *
	 * which fails if the condition is not met, because the implementation is
	 * not ready for such cases.
	 */
	template<typename UINT_T>
	__host__ __device__
	inline constexpr bool operator_lt_uint(const UINT_T &b) const {
		assert_unsigned_int_type<UINT_T>();

		bool result;

		if (sizeof(el_t) >= sizeof(UINT_T)) {
			assert(sizeof(el_t) % sizeof(UINT_T) == 0);

			result = (to_el_t() < b);
			result &= is_zero_starting_at_el<1>();

		} else {
			assert(sizeof(UINT_T) % sizeof(el_t) == 0);

			result = (as_uint<UINT_T>() < b);
			result &= is_zero_starting_at_el<sizeof(UINT_T) / sizeof(el_t)>();
		}

		return result;
	}

	/*
	 * Returns true if, and only if, b represents a value greater than this
	 * object.
	 *
	 * This function calls
	 *
	 * assert(
	 *   sizeof(el_t) % sizeof(UINT_T) == 0 ||
	 *   sizeof(UINT_T) % sizeof(el_t) == 0
	 * ),
	 *
	 * which fails if the condition is not met, because the implementation is
	 * not ready for such cases.
	 */
	template<typename UINT_T>
	__host__ __device__
	inline constexpr bool operator_gt_uint(const UINT_T &b) const {
		assert_unsigned_int_type<UINT_T>();

		bool result;

		if (sizeof(el_t) >= sizeof(UINT_T)) {
			assert(sizeof(el_t) % sizeof(UINT_T) == 0);

			result = (to_el_t() > b);
			result |= is_nonzero_starting_at_el<1>();

		} else {
			assert(sizeof(UINT_T) % sizeof(el_t) == 0);

			result = (as_uint<UINT_T>() > b);
			result |= is_nonzero_starting_at_el<sizeof(UINT_T) / sizeof(el_t)>();
		}

		return result;
	}

	/*
	 * Returns true if, and only if, b represents a value less than or equal to
	 * this object.
	 *
	 * This function calls
	 *
	 * assert(
	 *   sizeof(el_t) % sizeof(UINT_T) == 0 ||
	 *   sizeof(UINT_T) % sizeof(el_t) == 0
	 * ),
	 *
	 * which fails if the condition is not met, because the implementation is
	 * not ready for such cases.
	 */
	template<typename UINT_T>
	__host__ __device__
	inline constexpr bool operator_leq_uint(const UINT_T &b) const {
		assert_unsigned_int_type<UINT_T>();

		bool result;

		if (sizeof(el_t) >= sizeof(UINT_T)) {
			assert(sizeof(el_t) % sizeof(UINT_T) == 0);

			result = (to_el_t() <= b);
			result &= is_zero_starting_at_el<1>();

		} else {
			assert(sizeof(UINT_T) % sizeof(el_t) == 0);

			result = (as_uint<UINT_T>() <= b);
			result &= is_zero_starting_at_el<sizeof(UINT_T) / sizeof(el_t)>();
		}

		return result;
	}

	/*
	 * Returns true if, and only if, b represents a value greater than or equal
	 * to this object.
	 *
	 * This function calls
	 *
	 * assert(
	 *   sizeof(el_t) % sizeof(UINT_T) == 0 ||
	 *   sizeof(UINT_T) % sizeof(el_t) == 0
	 * ),
	 *
	 * which fails if the condition is not met, because the implementation is
	 * not ready for such cases.
	 */
	template<typename UINT_T>
	__host__ __device__
	inline constexpr bool operator_geq_uint(const UINT_T &b) const {
		assert_unsigned_int_type<UINT_T>();

		bool result;

		if (sizeof(el_t) >= sizeof(UINT_T)) {
			assert(sizeof(el_t) % sizeof(UINT_T) == 0);

			result = (to_el_t() >= b);
			result |= is_nonzero_starting_at_el<1>();

		} else {
			assert(sizeof(UINT_T) % sizeof(el_t) == 0);

			result = (as_uint<UINT_T>() >= b);
			result |= is_nonzero_starting_at_el<sizeof(UINT_T) / sizeof(el_t)>();
		}

		return result;
	}

	template<typename UINT_T>
	__host__ __device__
	inline constexpr bui& operator_pluseq_uint(const UINT_T &b) {
		assert_unsigned_int_type<UINT_T>();

		if (sizeof(UINT_T) < sizeof(el_t)) {
			return operator_pluseq_uint((el_t) b);
		}

		assert(sizeof(UINT_T) % sizeof(el_t) == 0); // compiled to 0 instructions on success

		size_t B_EL_COUNT = std::min(sizeof(UINT_T) / sizeof(el_t), SIZE_IN_ELS);

		el_t carry = 0;

#ifdef __NVCC__
		#pragma unroll
		#endif
		for (size_t i = 0; i < B_EL_COUNT; i++) {
			el[i] += carry;
			carry = (el[i] < carry);

			el_t b_i = (el_t) (b >> i * EL_SIZE_IN_BITS);
			el[i] += b_i;
			carry |= (el[i] < b_i);
		}

#ifdef __NVCC__
		#pragma unroll
		#endif
		for (size_t i = B_EL_COUNT; i < SIZE_IN_ELS; i++) {
			el[i] += carry;
			carry = (el[i] < carry);
		}

		return *this;
	}

	template<typename UINT_T>
	__host__ __device__
	inline constexpr bui& operator_minuseq_uint(const UINT_T &b) {
		assert_unsigned_int_type<UINT_T>();

		if (sizeof(UINT_T) < sizeof(el_t)) {
			return operator_minuseq_uint((el_t) b);
		}

		assert(sizeof(UINT_T) % sizeof(el_t) == 0); // compiled to 0 instructions on success

		size_t B_EL_COUNT = std::min(sizeof(UINT_T) / sizeof(el_t), SIZE_IN_ELS);

		el_t carry = 0;

#ifdef __NVCC__
		#pragma unroll
		#endif
		for (size_t i = 0; i < B_EL_COUNT; i++) {
			el_t el_i_before;

			el_i_before = el[i];

			el[i] -= carry;
			carry = (el[i] > el_i_before);

			el_t b_i = (el_t) (b >> i * EL_SIZE_IN_BITS);
			el_i_before = el[i];
			el[i] -= b_i;
			carry |= (el[i] > el_i_before);
		}

#ifdef __NVCC__
		#pragma unroll
		#endif
		for (size_t i = B_EL_COUNT; i < SIZE_IN_ELS; i++) {
			el_t el_i_before = el[i];
			el[i] -= carry;
			carry = (el[i] > el_i_before);
		}

		return *this;
	}

	template<typename UINT_T>
	__host__ __device__
	inline constexpr bui& operator_muleq_uint(const UINT_T &b) {
		assert_unsigned_int_type<UINT_T>();

		if (sizeof(UINT_T) < sizeof(el_t)) {
			return operator_muleq_uint((el_t) b);
		}

		assert(sizeof(UINT_T) % sizeof(el_t) == 0); // compiled to 0 instructions on success

		if (sizeof(UINT_T) == sizeof(el_t)) {
			tw_t tw = 0;

#ifdef __NVCC__
			#pragma unroll
			#endif
			for (size_t i = 0; i < SIZE_IN_ELS; i++) {
				tw += ((tw_t) el[i]) * b;

				el[i] = (el_t) tw;

				tw >>= EL_SIZE_IN_BITS;
			}

		} else /* if (sizeof(UINT_T) > sizeof(el_t)) */{
			el_t b_0 = (el_t) b;
			const UINT_T b_upper = b >> EL_SIZE_IN_BITS;

			UINT_T ui = 0;
			tw_t tw = 0;

#ifdef __NVCC__
			#pragma unroll
			#endif
			for (size_t i = 0; i < SIZE_IN_ELS; i++) {
				tw += ((tw_t) b_0) * el[i];

				el[i] = (el_t) tw;
				tw >>= EL_SIZE_IN_BITS;

				ui += b_upper * el[i];

				tw += (el_t) ui;
				ui >>= EL_SIZE_IN_BITS;
			}
		}

		return *this;
	}

	template<typename UINT_T>
	__host__ __device__
	inline constexpr bui& operator_diveq_uint(const UINT_T &b) {
		assert_unsigned_int_type<UINT_T>();

		if (sizeof(UINT_T) < sizeof(el_t)) {
			return operator_diveq_uint((el_t) b);
		}

		assert(sizeof(UINT_T) % sizeof(el_t) == 0); // compiled to 0 instructions on success

		if (sizeof(UINT_T) == sizeof(el_t)) {
			tw_t tw = 0;

#ifdef __NVCC__
	#pragma unroll
	#endif
			for (size_t i = SIZE_IN_ELS - 1; i != (size_t) -1; i--) {
				tw <<= EL_SIZE_IN_BITS;
				tw |= el[i];
				el[i] = (el_t) (tw / b);
				tw %= b;
			}

			return (el_t) tw;
		} else /* if (sizeof(UINT_T) > sizeof(el_t)) */{


		}
	}

public:
	template<size_t START_EL_IDX>
	__host__ __device__
	inline void set_zero_starting_at_el() {
#ifdef __NVCC__
#pragma unroll
#endif
		for (size_t i = START_EL_IDX; i < SIZE_IN_ELS; i++) {
			el[i] = 0;
		}
	}

	__host__ __device__
	inline constexpr void set_zero() {
		set_zero_starting_at_el<0>();
	}

	template<typename INT_T>
	__host__ __device__
	inline constexpr INT_T as() const {
		static_assert_singed_or_unsigned_int_type<INT_T>();

		if (std::is_unsigned_v<INT_T>) {
			return as_uint<INT_T>();
		} else {
			return as_uint<std::make_unsigned_t<INT_T>>();
		}
	}

	template<typename INT_T>
	__host__ __device__
	inline bui& set(const INT_T &value) {
		if (std::is_unsigned_v<INT_T>) {
			return set_uint(value);
		} else {
			return set_uint(static_cast<std::make_unsigned_t<INT_T>>(value));
		}
	}

	template<typename INT_T>
	__host__ __device__
	inline constexpr bui& operator=(const INT_T &value) {
		return set(value);
	}

	__host__ __device__
	inline constexpr bui& operator=(const char *str) {
		set_from_str(str);
	}

	__host__ __device__
	inline constexpr size_t bitlen() const {
#ifdef __NVCC__
#pragma unroll
#endif
		for (size_t i = SIZE_IN_ELS - 1; i < SIZE_IN_ELS; i--) {
			if (el[i] != 0) {
				size_t sub_el_bitlen = bitlen(el[i]);
				return i * EL_SIZE_IN_BITS + sub_el_bitlen;
			}
		}

		return 0;
	}

	template<size_t START_EL_IDX>
	__host__ __device__
	inline constexpr bool is_zero_starting_at_el() const {
		bool result = true;

#ifdef __NVCC__
#pragma unroll
#endif
		for (size_t i = START_EL_IDX; i < SIZE_IN_ELS; i++) {
			result &= (el[i] == 0);
		}

		return result;
	}

	template<size_t START_EL_IDX>
	__host__ __device__
	inline constexpr bool is_nonzero_starting_at_el() const {
		bool result = false;

#ifdef __NVCC__
#pragma unroll
#endif
		for (size_t i = START_EL_IDX; i < SIZE_IN_ELS; i++) {
			result |= (el[i] != 0);
		}

		return result;
	}

	__host__ __device__
	inline constexpr bool is_zero() const {
		return is_zero_starting_at_el<0>();
	}

	__host__ __device__
	inline constexpr bool is_nonzero() const {
		return is_nonzero_starting_at_el<0>();
	}

	template<typename INT_T>
	__host__ __device__
	inline constexpr bool operator!=(const INT_T &b) const {
		static_assert_singed_or_unsigned_int_type<INT_T>();

		if (std::is_unsigned_v<INT_T>) {
			const auto bu = static_cast<std::make_unsigned_t<INT_T>>(b);
			return operator_neq_uint(bu);

		} else {
			auto bu = static_cast<std::make_unsigned_t<INT_T>>(b);
			return (b < 0) | operator_neq_uint(bu);
		}
	}

	template<typename INT_T>
	__host__ __device__
	inline constexpr bool operator==(const INT_T &b) const {
		if (std::is_unsigned_v<INT_T>) {
			return operator_eq_uint(b);
		} else {
			const auto bu = static_cast<std::make_unsigned_t<INT_T>>(b);
			return (b >= 0) & operator_eq_uint(bu);
		}
	}

	template<typename INT_T>
	__host__ __device__
	inline constexpr bool operator>(const INT_T &b) const {
		if (std::is_unsigned_v<INT_T>) {
			return operator_gt_uint(b);
		} else {
			return (b < 0) | operator_gt_uint(b);
		}
	}

	template<typename INT_T>
	__host__ __device__
	inline constexpr bool operator<(const INT_T &b) const {
		if (std::is_unsigned_v<INT_T>) {
			return operator_lt_uint(b);
		} else {
			return (b >= 0) & operator_lt_uint(b);
		}
	}

	template<typename INT_T>
	__host__ __device__
	inline constexpr bool operator>=(const INT_T &b) const {
		if (std::is_unsigned_v<INT_T>) {
			return operator_geq_uint(b);
		} else {
			return (b < 0) | operator_geq_uint(b);
		}
	}

	template<typename INT_T>
	__host__ __device__
	inline constexpr bool operator<=(const INT_T &b) const {
		if (std::is_unsigned_v<INT_T>) {
			return operator_leq_uint(b);
		} else {
			return (b >= 0) & operator_leq_uint(b);
		}
	}

	template<typename INT_T>
	__host__ __device__
	inline constexpr bui& operator+=(const INT_T &b) {
		if (std::is_unsigned_v<INT_T>) {
			return operator_pluseq_uint(b);

		} else if (b >= 0) {
			const auto bu = static_cast<std::make_unsigned_t<INT_T>>(b);
			return operator_pluseq_uint(bu);

		} else {
			const auto bu = static_cast<std::make_unsigned_t<INT_T>>(-b);
			return operator_minuseq_uint(bu);
		}
	}

	template<typename INT_T>
	__host__ __device__
	inline constexpr bui& operator-=(const INT_T &b) {
		if (std::is_unsigned_v<INT_T>) {
			return operator_minuseq_uint(b);

		} else if (b >= 0) {
			const auto bu = static_cast<std::make_unsigned_t<INT_T>>(b);
			return operator_minuseq_uint(bu);

		} else {
			const auto bu = static_cast<std::make_unsigned_t<INT_T>>(-b);
			return operator_pluseq_uint(bu);
		}
	}

	template<typename INT_T>
	__host__ __device__
	inline constexpr bui& operator*=(const INT_T &b) {
		if (std::is_unsigned_v<INT_T>) {
			return operator_muleq_uint(b);

		} else if (b >= 0) {
			const auto bu = static_cast<std::make_unsigned_t<INT_T>>(b);
			return operator_muleq_uint(bu);

		} else {
			assert(b >= 0); // fails
			return *this; // never reached
		}
	}

	__host__ __device__
	inline constexpr el_t operator/=(const el_t &b) {
		tw_t tw = 0;

#ifdef __NVCC__
#pragma unroll
#endif
		for (size_t i = SIZE_IN_ELS - 1; i != (size_t) -1; i--) {
			tw <<= EL_SIZE_IN_BITS;
			tw |= el[i];
			el[i] = (el_t) (tw / b);
			tw %= b;
		}

		return (el_t) tw;
	}

	__host__ __device__
	inline constexpr el_t operator%(const el_t &b) const {
		tw_t tw = 0;

#ifdef __NVCC__
#pragma unroll
#endif
		for (size_t i = SIZE_IN_ELS - 1; i != (size_t) -1; i--) {
			tw <<= EL_SIZE_IN_BITS;
			tw |= el[i];
			tw %= b;
		}

		return (el_t) tw;
	}

	__host__ __device__
	inline constexpr el_t operator&(const el_t &b) const {
		return el[0] & b;
	}

	__host__ __device__
	inline constexpr el_t operator&(const tw_t &b) const {
		tw_t tw = (tw_t) *this;
		return tw & b;
	}

	__host__ __device__
	inline constexpr el_t operator&(const int &b) const {
		if (sizeof(el_t) >= sizeof(int)) {
			return ((el_t) *this) & (el_t) b;
		}

		assert(false);
		return 0; // will never be reached
	}

	/*
	 * Shift the bits of this number by WIDTH many bit positions towards lower bit indices (right shift).
	 */
	template<uint32_t WIDTH>
	__host__ __device__
	inline constexpr bui& loshift_bits() {
		static_assert(WIDTH < EL_SIZE_IN_BITS);

		el[0] >>= WIDTH;

#ifdef __NVCC__
#pragma unroll
#endif
		for (size_t i = 1; i < SIZE_IN_ELS; i++) {
			el[i - 1] |= el[i] << (EL_SIZE_IN_BITS - WIDTH);
			el[i] >>= WIDTH;
		}

		return *this;
	}

	/*
	 * Shift the bits of this number by WIDTH many bit positions towards higher bit indices (left shift).
	 */
	template<uint32_t WIDTH>
	__host__ __device__
	inline constexpr bui& hishift_bits() {
		static_assert(WIDTH < EL_SIZE_IN_BITS);

		el[SIZE_IN_ELS - 1] <<= WIDTH;

#ifdef __NVCC__
#pragma unroll
#endif
		for (size_t i = SIZE_IN_ELS - 2; i != (size_t) -1; i--) {
			el[i + 1] |= el[i] >> (EL_SIZE_IN_BITS - WIDTH);
			el[i] <<= WIDTH;
		}

		return *this;
	}

	/*
	 * Shift the elements of this number by one position towards lower bit indices (right shift).
	 */
	__host__ __device__
	inline constexpr bui& loshift_1el() {
#ifdef __NVCC__
#pragma unroll
#endif
		for (size_t i = 0; i < SIZE_IN_ELS - 1; i++) {
			el[i] = el[i + 1];
		}

		el[SIZE_IN_ELS - 1] = 0;

		return *this;
	}

	/*
	 * Shift the elements of this number by one position towards higher bit indices (right shift).
	 */
	__host__ __device__
	inline constexpr bui& hishift_1el() {
#ifdef __NVCC__
#pragma unroll
#endif
		for (size_t i = SIZE_IN_ELS - 1; i < SIZE_IN_ELS; i--) {
			el[i] = el[i - 1];
		}

		el[0] = 0;

		return *this;
	}

	inline std::string str() const {
		return uint_to_string(*this);
	}
}
;

template<size_t SIZE_IN_BITS>
inline std::string to_string(const bui<SIZE_IN_BITS> &x) {
	return x.str();
}

template<size_t SIZE_IN_BITS>
inline std::ostream& operator<<(std::ostream &os, const bui<SIZE_IN_BITS> &x) {
	return os << x.str();
}

} /* namespace bifsi */

#endif /* BIFSI_H_ */
