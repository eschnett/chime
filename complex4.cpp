#include <immintrin.h>

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <utility>

using std::array;
using std::complex;
using std::cout;
using std::exp;
using std::get;
using std::make_pair;
using std::make_tuple;
using std::pair;
using std::rint;
using std::tuple;

// Split 4-bit offset-encoded complex numbers into separate 8-bit real and
// imaginary parts, assuming the real part lives in the low bits
pair<__m256i, __m256i> split_complex(__m256i z) {
  __m256i lomask = _mm256_set1_epi8(0x0f);
  __m256i x = _mm256_and_si256(z, lomask);
  __m256i y = _mm256_and_si256(_mm256_srli_epi16(z, 4), lomask);
  __m256i const8 = _mm256_set1_epi8(8);
  x = _mm256_sub_epi8(x, const8);
  y = _mm256_sub_epi8(y, const8);
  return make_pair(x, y);
}

// Combine 8-bit real and imaginary parts into a single complex number,
// discarding the high bits
__m256i combine_complex(__m256i x, __m256i y) {
  __m256i const8 = _mm256_set1_epi8(8);
  x = _mm256_add_epi8(x, const8);
  y = _mm256_add_epi8(y, const8);
  __m256i lomask = _mm256_set1_epi8(0x0f);
  x = _mm256_and_si256(x, lomask);
  y = _mm256_slli_epi16(_mm256_and_si256(y, lomask), 4);
  return _mm256_or_si256(x, y);
}

// Convert an 8-bit integer vector to two 16-bit integer vectors
pair<__m256i, __m256i> cvt8to16(__m256i x) {
  __m128i xlo = _mm256_castsi256_si128(x);
  __m128i xhi = _mm256_castsi256_si128(_mm256_permute2x128_si256(x, x, 1));
  __m256i ylo = _mm256_cvtepi8_epi16(xlo);
  __m256i yhi = _mm256_cvtepi8_epi16(xhi);
  return make_pair(ylo, yhi);
}

// Convert an 8-bit integer vector to two fixed-point 16-bit integer vectors
pair<__m256i, __m256i> cvt8to16fp(__m256i x) {
  __m256i evenmask = _mm256_set1_epi16(0xff00);
  __m256i xeven = _mm256_and_si256(x, evenmask);
  __m256i xodd = _mm256_slli_epi16(x, 8);
  xodd = _mm256_and_si256(xodd, evenmask);
  return make_pair(xodd, xeven);
}

// Convert two fixed-point 16-bit integer vectors to an 8-bit integer vector,
// discarding the fractional part
__m256i cvt16fpto8(__m256i xodd, __m256i xeven) {
  __m256i evenmask = _mm256_set1_epi16(0xff00);
  xeven = _mm256_and_si256(xeven, evenmask);
  xodd = _mm256_and_si256(xodd, evenmask);
  xodd = _mm256_srli_epi16(xodd, 8);
  return _mm256_or_si256(xodd, xeven);
}

// Classes for integer arithmetic, to provide a shorter and more readable
// notation

// Vector of integers
class int16x16 {
  __m256i val;

public:
  int16x16() = default;
  int16x16(__m256i val) : val(val) {}
  operator __m256i() { return val; }
  explicit int16x16(int16_t x) : val(_mm256_set1_epi16(x)) {}
  int16x16 operator+() const { return *this; }
  int16x16 operator-() const { return int16x16(0) - *this; }
  int16x16 operator+(int16x16 x) const { return _mm256_add_epi16(val, x.val); }
  int16x16 &operator+=(int16x16 x) { return *this = *this + x; }
  int16x16 operator-(int16x16 x) const { return _mm256_sub_epi16(val, x.val); }
  int16x16 &operator-=(int16x16 x) { return *this = *this - x; }
  // This expects 15 fractional bits in x
  int16x16 operator*(int16x16 x) const {
    return _mm256_mulhrs_epi16(val, x.val);
  }
  int16x16 &operator*=(int16x16 x) { return *this = *this * x; }
};

// Complex vector of integers (stored as two vectors)
class cpint16x16 {
  int16x16 re, im;

  // q = 1^(1/4); q^2 = i
  static const int16_t qval = 0.7071067811865476 * (1 << 15) + 0.5;
  // q = qval + i qval

public:
  cpint16x16() = default;
  cpint16x16(int16x16 re, int16x16 im = int16x16(0)) : re(re), im(im) {}
  friend int16x16 real(cpint16x16 x) { return x.re; }
  friend int16x16 imag(cpint16x16 x) { return x.im; }
  cpint16x16 operator+() const { return *this; }
  cpint16x16 operator-() const { return {-re, -im}; }
  friend cpint16x16 conj(cpint16x16 x) { return {x.re, -x.im}; }
  cpint16x16 operator+(cpint16x16 x) const { return {re + x.re, im + x.im}; }
  cpint16x16 &operator+=(cpint16x16 x) { return *this = *this + x; }
  cpint16x16 operator-(cpint16x16 x) const { return {re - x.re, im - x.im}; }
  cpint16x16 &operator-=(cpint16x16 x) { return *this = *this - x; }
  friend cpint16x16 times_i(cpint16x16 x) { return {-x.im, x.re}; }
  friend cpint16x16 times_q(cpint16x16 x) {
    return {(x.re - x.im) * int16x16(qval), (x.re + x.im) * int16x16(qval)};
  }
  friend cpint16x16 times_iq(cpint16x16 x) { return times_q(times_i(x)); }
};

// Fourier transform
// Input: z[0:15]
// Output: c[0:15]
//    c[k] = Sum_j exp(-2πi k j / 16) x[j]

// Attempt 1: direct implementation, with pointer to data

void fft1(const int16x16 *__restrict__ zre, const int16x16 *__restrict__ zim,
          int16x16 *__restrict__ cre, int16x16 *__restrict__ cim) {
  zre = (const int16x16 *)__builtin_assume_aligned(zre, 32);
  zim = (const int16x16 *)__builtin_assume_aligned(zim, 32);
  cre = (int16x16 *)__builtin_assume_aligned(cre, 32);
  cim = (int16x16 *)__builtin_assume_aligned(cim, 32);
  cre[0] = zre[0];
  cim[0] = zim[0];
}

void fft2(const int16x16 *__restrict__ zre, const int16x16 *__restrict__ zim,
          int16x16 *__restrict__ cre, int16x16 *__restrict__ cim) {
  zre = (const int16x16 *)__builtin_assume_aligned(zre, 32);
  zim = (const int16x16 *)__builtin_assume_aligned(zim, 32);
  cre = (int16x16 *)__builtin_assume_aligned(cre, 32);
  cim = (int16x16 *)__builtin_assume_aligned(cim, 32);
  // cre[0] = _mm256_add_epi16(zre[0], zre[1]);
  // cim[0] = _mm256_add_epi16(zim[0], zim[1]);
  // cre[1] = _mm256_sub_epi16(zre[0], zre[1]);
  // cim[1] = _mm256_sub_epi16(zim[0], zim[1]);
  cre[0] = zre[0] + zre[1];
  cim[0] = zim[0] + zim[1];
  cre[1] = zre[0] - zre[1];
  cim[1] = zim[0] - zim[1];
}

void fft4(const int16x16 *__restrict__ zre, const int16x16 *__restrict__ zim,
          int16x16 *__restrict__ cre, int16x16 *__restrict__ cim) {
  zre = (const int16x16 *)__builtin_assume_aligned(zre, 32);
  zim = (const int16x16 *)__builtin_assume_aligned(zim, 32);
  cre = (int16x16 *)__builtin_assume_aligned(cre, 32);
  cim = (int16x16 *)__builtin_assume_aligned(cim, 32);
  // c[0] = z[0] +   z[1] + z[2] +   z[3]
  // c[1] = z[0] + i z[1] - z[2] - i z[3]
  // c[2] = z[0] -   z[1] + z[2] -   z[3]
  // c[3] = z[0] - i z[1] - z[2] + i z[3]
  cre[0] = (zre[0] + zre[2]) + (zre[1] + zre[3]);
  cim[0] = (zim[0] + zim[2]) + (zim[1] + zim[3]);
  cre[1] = (zim[1] - zim[3]) - (zre[0] - zre[2]);
  cim[1] = (zim[0] - zim[2]) - (zre[1] - zre[3]);
  cre[2] = (zre[0] + zre[2]) - (zre[1] + zre[3]);
  cim[2] = (zim[0] + zim[2]) - (zim[1] + zim[3]);
  cre[3] = (zre[0] - zre[2]) + (zim[1] - zim[3]);
  cim[3] = (zim[0] - zim[2]) - (zre[1] - zre[3]);
}

// Attempt 2: Passing data as value arrays

auto fft1(array<int16x16, 1> zre, array<int16x16, 1> zim)
    -> pair<array<int16x16, 1>, array<int16x16, 1>> {
  array<int16x16, 1> cre, cim;
  cre[0] = zre[0];
  cim[0] = zim[0];
  return {cre, cim};
}

auto fft2(array<int16x16, 2> zre, array<int16x16, 2> zim)
    -> pair<array<int16x16, 2>, array<int16x16, 2>> {
  array<int16x16, 2> cre, cim;
  cre[0] = zre[0] + zre[1];
  cim[0] = zim[0] + zim[1];
  cre[1] = zre[0] - zre[1];
  cim[1] = zim[0] - zim[1];
  return {cre, cim};
}

auto fft4(array<int16x16, 4> zre, array<int16x16, 4> zim)
    -> pair<array<int16x16, 4>, array<int16x16, 4>> {
  // c[0] = z[0] +   z[1] + z[2] +   z[3]
  // c[1] = z[0] + i z[1] - z[2] - i z[3]
  // c[2] = z[0] -   z[1] + z[2] -   z[3]
  // c[3] = z[0] - i z[1] - z[2] + i z[3]
  array<int16x16, 4> cre, cim;
  cre[0] = (zre[0] + zre[2]) + (zre[1] + zre[3]);
  cim[0] = (zim[0] + zim[2]) + (zim[1] + zim[3]);
  cre[1] = (zim[1] - zim[3]) - (zre[0] - zre[2]);
  cim[1] = (zim[0] - zim[2]) - (zre[1] - zre[3]);
  cre[2] = (zre[0] + zre[2]) - (zre[1] + zre[3]);
  cim[2] = (zim[0] + zim[2]) - (zim[1] + zim[3]);
  cre[3] = (zre[0] - zre[2]) + (zim[1] - zim[3]);
  cim[3] = (zim[0] - zim[2]) - (zre[1] - zre[3]);
  return {cre, cim};
}

// Attempt 3: Using complex arithmetic

auto fft1(array<cpint16x16, 1> z) -> array<cpint16x16, 1> {
  array<cpint16x16, 1> c;
  c[0] = z[0];
  return c;
}

auto fft2(array<cpint16x16, 2> z) -> array<cpint16x16, 2> {
  array<cpint16x16, 2> c;
  c[0] = z[0] + z[1];
  c[1] = z[0] - z[1];
  return c;
}

auto fft4(array<cpint16x16, 4> z) -> array<cpint16x16, 4> {
  // c[0] = z[0] +   z[1] + z[2] +   z[3]
  // c[1] = z[0] + i z[1] - z[2] - i z[3]
  // c[2] = z[0] -   z[1] + z[2] -   z[3]
  // c[3] = z[0] - i z[1] - z[2] + i z[3]
  array<cpint16x16, 4> c;
  c[0] = (z[0] + z[2]) + (z[1] + z[3]);
  c[1] = (z[0] - z[2]) + times_i(z[1] - z[3]);
  c[2] = (z[0] + z[2]) - (z[1] + z[3]);
  c[3] = (z[0] - z[2]) - times_i(z[1] - z[3]);
  return c;
}

auto fft8(array<cpint16x16, 8> z) -> array<cpint16x16, 8> {
  // c_0 = z_0 +    z_1 +   z_2 +    z_3 + z_4 +    z_5 +   z_6 +    z_7
  // c_1 = z_0 +  q z_1 + i z_2 + iq z_3 - z_4 -  q z_5 - i z_6 - iq z_7
  // c_2 = z_0 + i  z_1 -   z_2 - i  z_3 + z_4 + i  z_5 -   z_6 - i  z_7
  // c_3 = z_0 + iq z_1 - i z_2 +  q z_3 - z_4 - iq z_5 + i z_6 -  q z_7
  // c_4 = z_0 -    z_1 +   z_2 -    z_3 + z_4 -    z_5 +   z_6 -    z_7
  // c_5 = z_0 -  q z_1 + i z_2 - iq z_3 - z_4 +  q z_5 - i z_6 + iq z_7
  // c_6 = z_0 - i  z_1 -   z_2 + i  z_3 + z_4 - i  z_5 -   z_6 + i  z_7
  // c_7 = z_0 - iq z_1 - i z_2 -  q z_3 - z_4 + iq z_5 + i z_6 +  q z_7
  array<cpint16x16, 8> c;
  c[0] = ((z[0] + z[4]) + (z[2] + z[6])) + ((z[1] + z[5]) + (z[3] + z[7]));
  c[1] = ((z[0] - z[4]) + times_i(z[2] - z[6])) +
         times_q((z[1] - z[5]) + times_i(z[3] - z[7]));
  c[2] =
      ((z[0] + z[4]) - (z[2] + z[6])) + times_i((z[1] + z[5]) - (z[3] + z[7]));
  c[3] = ((z[0] - z[4]) - times_i(z[2] - z[6])) +
         times_iq((z[1] - z[5]) - times_i(z[3] - z[7]));
  c[4] = ((z[0] + z[4]) + (z[2] + z[6])) - ((z[1] + z[5]) + (z[3] + z[7]));
  c[5] = ((z[0] - z[4]) + times_i(z[2] - z[6])) -
         times_q((z[1] - z[5]) + times_i(z[3] - z[7]));
  c[6] =
      ((z[0] + z[4]) - (z[2] + z[6])) - times_i((z[1] + z[5]) - (z[3] + z[7]));
  c[0] = ((z[0] - z[4]) - times_i(z[2] - z[6])) -
         times_iq((z[1] - z[5]) - times_i(z[3] - z[7]));
  return c;
}
