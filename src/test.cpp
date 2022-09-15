/*
 * test.cpp
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
 * Stand alone program for running tests on the bifsi library. You don't need
 * this file for using the library. For using the library, you only need to
 * include the file bifsi.h. It's a one header file only library.
 */

#include <bits/stdint-intn.h>
#include <bits/stdint-uintn.h>
#include <stddef.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <typeinfo>

#include "bifsi.h"

using std::cout;
using std::endl;
using std::flush;
using std::string;
using bifsi::bui;
using bifsi::to_string;

typedef unsigned __int128 uint128_t;

inline std::string to_string(uint128_t x) {
	constexpr double LOG_BASE10_2 = 0.301029995663981195; //std::log10((double) 2);

	constexpr int32_t INT128_SIZE_IN_BITS = sizeof(uint128_t) * 8;

	constexpr int32_t MAX_DIGITS = bifsi::ceil_constexpr(INT128_SIZE_IN_BITS * LOG_BASE10_2);

	char result[MAX_DIGITS + 1];

	uint32_t i = 0;

	while (x != 0) {
		result[i] = '0' + (char) (x % 10);
		x /= 10;

		i++;
	}

	if (i == 0) {
		return "0";
	}

	result[i] = '\0';

	std::reverse(result, result + i);

	return std::string(result);
}

inline std::ostream& operator<<(std::ostream &os, const uint128_t &x) {
	return os << to_string(x);
}

int main() {
	uint128_t before = 0;
	uint128_t expected = 0;
	bifsi::bui<128> actual = 0;

//	if (actual != 0) {
//		cout << "actual != 0" << endl;
//	}
//	if (actual == 0) {
//		cout << "actual == 0" << endl;
//	}
//	cout << ((uint32_t) actual) << endl;

	std::srand(12345);

	size_t i = 0;

	const size_t TEST_COUNT = 20000000;

	int curr_percent = 0;
	int logged_percent = 0;

	cout << "running tests on " << bifsi::type_name<decltype(actual)>() << endl;

	for (; i < TEST_COUNT; i++) {
		int op_idx = std::rand() % 5;
		bifsi::el_t op_val = std::rand();

		bifsi::el_t r_expected = 0;
		bifsi::el_t r_actual = 0;

		before = expected;

		switch (op_idx) {
		case 0:
			expected += op_val;
			actual += op_val;
			break;
		case 1:
			expected -= op_val;
			actual -= op_val;
			break;
		case 2:
			expected *= op_val;
			actual *= op_val;
			break;
		case 3:
			op_val += (op_val == 0);

			expected /= op_val;
			actual /= op_val;
			break;
		case 4:
			op_val += (op_val == 0);

			r_expected = expected % op_val;
			r_actual = actual % op_val;

			if (r_actual != r_expected) {
				cout << "test failed: r_actual != r_expected:" << endl;
				cout << "i         : " << i << endl;
				cout << "before    : " << before << endl;
				cout << "op_idx    : " << op_idx << endl;
				cout << "op_val    : " << op_val << endl;
				cout << "r_expected: " << r_expected << endl;
				cout << "r_actual  : " << r_actual << endl;
				return 1;
			}

			break;
		default:
			throw std::runtime_error("unknown op_idx");
		}

		if (to_string(actual) != to_string(expected)) {
			cout << "test failed: actual != expected:" << endl;
			cout << "i       : " << i << endl;
			cout << "before  : " << before << endl;
			cout << "op_idx  : " << op_idx << endl;
			cout << "op_val  : " << op_val << endl;
			cout << "expected: " << expected << endl;
			cout << "actual  : " << actual << endl;

			return 1;
		}

		curr_percent = i * 100 / (TEST_COUNT - 1);
		if (curr_percent > logged_percent) {
			logged_percent = curr_percent;

			cout << logged_percent << "%" << endl;
		}
	}

	cout << TEST_COUNT << " tests completed successfully." << endl;

	return 0;
}
