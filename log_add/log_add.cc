// AVX intrinsics
#include <immintrin.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sys/time.h>
#include <vector>
using namespace std;

////////////////////////////////////////////////////////////////////////////////

// A template class for SIMD vectors
template <typename T, int N> struct vec;

// Basic functions
template <typename T, int N>
typename vec<T, N>::floatvec_type as_float(vec<T, N> x) {
  return vec<T, N>::as_float(x);
}
template <typename T, int N>
typename vec<T, N>::intvec_type as_int(vec<T, N> x) {
  return vec<T, N>::as_int(x);
}
template <typename T, int N> vec<T, N> ceil(vec<T, N> x) {
  return vec<T, N>::ceil(x);
}
template <typename T, int N>
typename vec<T, N>::floatvec_type convert_float(vec<T, N> x) {
  return vec<T, N>::convert_float(x);
}
template <typename T, int N>
typename vec<T, N>::intvec_type convert_int(vec<T, N> x) {
  return vec<T, N>::convert_int(x);
}
template <typename T, int N> vec<T, N> fabs(vec<T, N> x) {
  return vec<T, N>::fabs(x);
}
template <typename T, int N> vec<T, N> fdim(vec<T, N> x, vec<T, N> y) {
  return vec<T, N>::fdim(x, y);
}
template <typename T, int N> vec<T, N> floor(vec<T, N> x) {
  return vec<T, N>::floor(x);
}
template <typename T, int N> vec<T, N> fmax(vec<T, N> x, vec<T, N> y) {
  return vec<T, N>::fmax(x, y);
}
template <typename T, int N> vec<T, N> fmin(vec<T, N> x, vec<T, N> y) {
  return vec<T, N>::fmin(x, y);
}
template <typename T, int N>
typename vec<T, N>::intvec_type ilogb(vec<T, N> x) {
  return vec<T, N>::ilogb(x);
}
template <typename T, int N>
vec<T, N> ldexp(vec<T, N> x, typename vec<T, N>::intvec_type n) {
  return vec<T, N>::ldexp(x, n);
}
template <typename T, int N>
vec<T, N> muladd(vec<T, N> x, vec<T, N> y, vec<T, N> z) {
  return vec<T, N>::muladd(x, y, z);
}
template <typename T, int N> vec<T, N> rint(vec<T, N> x) {
  return vec<T, N>::rint(x);
}
template <typename T, int N> vec<T, N> trunc(vec<T, N> x) {
  return vec<T, N>::trunc(x);
}

////////////////////////////////////////////////////////////////////////////////

// Template specializations for single precision
template <> struct vec<int, 8>;
template <> struct vec<float, 8>;

template <> struct vec<int, 8> {
  // Meta information
  typedef int value_type;
  static const int size = 8;

  typedef vec<float, 8> floatvec_type;

  // Implementation
  __m256i elts;

  // Straightforward constructors
  vec() {}
  vec(const vec &x) : elts(x.elts) {}
  vec &operator=(const vec &x) {
    elts = x.elts;
    return *this;
  }
  vec(__m256i elts) : elts(elts) {}
  operator __m256i() const { return elts; }

  // Create from scalar and scalar access
  vec(int a) : elts(_mm256_set1_epi32(a)) {}
  int operator[](int d) const {
    int a;
    std::memcpy(&a, (const int *)&elts + d, sizeof a);
    return a;
  }

  // Load and store (aligned)
  static vec load(const int *p); // defined later
  void store(int *p) const;      // defined later

  vec operator+() const { return *this; }
  vec operator-() const { return vec(0) - *this; }

  vec operator+(vec y) const {
    vec x = *this;
    __m128i xlo = _mm256_castsi256_si128(x);
    __m128i xhi = _mm256_extractf128_si256(x, 1);
    __m128i ylo = _mm256_castsi256_si128(y);
    __m128i yhi = _mm256_extractf128_si256(y, 1);
    __m128i rlo = _mm_add_epi32(xlo, ylo);
    __m128i rhi = _mm_add_epi32(xhi, yhi);
    return _mm256_insertf128_si256(_mm256_castsi128_si256(rlo), rhi, 1);
  }
  vec operator-(vec y) const {
    vec x = *this;
    __m128i xlo = _mm256_castsi256_si128(x);
    __m128i xhi = _mm256_extractf128_si256(x, 1);
    __m128i ylo = _mm256_castsi256_si128(y);
    __m128i yhi = _mm256_extractf128_si256(y, 1);
    __m128i rlo = _mm_sub_epi32(xlo, ylo);
    __m128i rhi = _mm_sub_epi32(xhi, yhi);
    return _mm256_insertf128_si256(_mm256_castsi128_si256(rlo), rhi, 1);
  }

  vec &operator+=(vec const &x) { return *this = *this + x; }
  vec &operator-=(vec const &x) { return *this = *this - x; }

  vec operator~() const { return vec(0) ^ *this; }

  vec operator&(vec x) const; // defined below
  vec operator|(vec x) const; // defined below
  vec operator^(vec x) const; // defined below

  vec &operator&=(vec const &x) { return *this = *this & x; }
  vec &operator|=(vec const &x) { return *this = *this | x; }
  vec &operator^=(vec const &x) { return *this = *this ^ x; }

  vec operator<<(int n) const {
    vec x = *this;
    __m128i xlo = _mm256_castsi256_si128(x);
    __m128i xhi = _mm256_extractf128_si256(x, 1);
    __m128i rlo = _mm_slli_epi32(xlo, n);
    __m128i rhi = _mm_slli_epi32(xhi, n);
    return _mm256_insertf128_si256(_mm256_castsi128_si256(rlo), rhi, 1);
  }
  vec operator>>(int n) const {
    vec x = *this;
    __m128i xlo = _mm256_castsi256_si128(x);
    __m128i xhi = _mm256_extractf128_si256(x, 1);
    __m128i rlo = _mm_srai_epi32(xlo, n);
    __m128i rhi = _mm_srai_epi32(xhi, n);
    return _mm256_insertf128_si256(_mm256_castsi128_si256(rlo), rhi, 1);
  }
  vec lsr(int n) const {
    vec x = *this;
    __m128i xlo = _mm256_castsi256_si128(x);
    __m128i xhi = _mm256_extractf128_si256(x, 1);
    __m128i rlo = _mm_srli_epi32(xlo, n);
    __m128i rhi = _mm_srli_epi32(xhi, n);
    return _mm256_insertf128_si256(_mm256_castsi128_si256(rlo), rhi, 1);
  }

  static floatvec_type as_float(vec x); // defined below
  static vec as_int(vec x) { return x; }
  static floatvec_type convert_float(vec x); // defined below
  static vec convert_int(vec x) { return x; }
};

template <> struct vec<float, 8> {
  // Meta information
  typedef float value_type;
  static const int size = 8;

  typedef vec<int, 8> intvec_type;
  static const int mantissa_bits = std::numeric_limits<float>::digits - 1;
  static const int signbit_bits = 1;
  static const int exponent_bits =
      8 * sizeof(float) - mantissa_bits - signbit_bits;
  static const int exponent_offset =
      2 - std::numeric_limits<float>::min_exponent;
  static const unsigned int mantissa_mask = ~(~0U << mantissa_bits);
  static const unsigned int exponent_mask = ~(~0U << exponent_bits)
                                            << mantissa_bits;
  static const unsigned int signbit_mask = ~(~0U << signbit_bits)
                                           << (mantissa_bits + exponent_bits);

  // Implementation
  __m256 elts;

  // Straightforward constructors
  vec() {}
  vec(const vec &x) : elts(x.elts) {}
  vec &operator=(const vec &x) {
    elts = x.elts;
    return *this;
  }
  vec(__m256 elts) : elts(elts) {}
  operator __m256() const { return elts; }

  // Create from scalar and scalar access
  vec(float a) : elts(_mm256_set1_ps(a)) {}
  float operator[](int d) const {
    float a;
    std::memcpy(&a, (const float *)&elts + d, sizeof a);
    return a;
  }

  // Load and store (aligned)
  static vec load(const float *p) { return _mm256_load_ps(p); }
  void store(float *p) const { _mm256_store_ps(p, elts); }

  vec operator+() const { return *this; }
  vec operator-() const { return vec(0) - *this; }

  vec operator+(vec x) const { return _mm256_add_ps(elts, x); }
  vec operator-(vec x) const { return _mm256_sub_ps(elts, x); }
  vec operator*(vec x) const { return _mm256_mul_ps(elts, x); }
  vec operator/(vec x) const { return _mm256_div_ps(elts, x); }

  vec &operator+=(vec const &x) { return *this = *this + x; }
  vec &operator-=(vec const &x) { return *this = *this - x; }
  vec &operator*=(vec const &x) { return *this = *this * x; }
  vec &operator/=(vec const &x) { return *this = *this / x; }

  vec operator~() const { return vec(0) ^ *this; }

  vec operator&(vec x) const { return _mm256_and_ps(elts, x); }
  vec operator|(vec x) const { return _mm256_or_ps(elts, x); }
  vec operator^(vec x) const { return _mm256_xor_ps(elts, x); }

  vec &operator&=(vec const &x) { return *this = *this & x; }
  vec &operator|=(vec const &x) { return *this = *this | x; }
  vec &operator^=(vec const &x) { return *this = *this ^ x; }

  static vec as_float(vec x) { return x; }
  static intvec_type as_int(vec x) { return _mm256_castps_si256(x); }
  static vec ceil(vec x) { return _mm256_ceil_ps(x); }
  static vec convert_float(vec x) { return x; }
  static intvec_type convert_int(vec x) { return _mm256_cvttps_epi32(x); }
  static vec fabs(vec x) { return x & as_float(~signbit_mask); }
  static vec fdim(vec x, vec y) { return fmax(x - y, vec(0)); }
  static vec floor(vec x) { return _mm256_floor_ps(x); }
  static vec fmax(vec x, vec y) { return _mm256_max_ps(x, y); }
  static vec fmin(vec x, vec y) { return _mm256_min_ps(x, y); }
  static intvec_type ilogb(vec x) {
    return (as_int(x) & intvec_type(exponent_mask)).lsr(mantissa_bits) -
           intvec_type(exponent_offset);
  }
  static vec ldexp(vec x, intvec_type n) {
    return ::as_float(as_int(x) + (n << mantissa_bits));
  }
  static vec muladd(vec x, vec y, vec z) { return x * y + z; }
  static vec rint(vec x) {
    return _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT);
  }
  static vec trunc(vec x) { return _mm256_round_ps(x, _MM_FROUND_TO_ZERO); }
};

vec<int, 8> vec<int, 8>::load(const int *p) {
  return ::as_int(floatvec_type::load((const float *)p));
}
void vec<int, 8>::store(int *p) const { as_float(elts).store((float *)p); }

vec<int, 8> vec<int, 8>::operator&(vec x) const {
  return ::as_int(as_float(*this) & as_float(x));
}
vec<int, 8> vec<int, 8>::operator|(vec x) const {
  return ::as_int(as_float(*this) | as_float(x));
}
vec<int, 8> vec<int, 8>::operator^(vec x) const {
  return ::as_int(as_float(*this) ^ as_float(x));
}

vec<int, 8>::floatvec_type vec<int, 8>::as_float(vec x) {
  return _mm256_castsi256_ps(x);
}
vec<int, 8>::floatvec_type vec<int, 8>::convert_float(vec x) {
  return _mm256_cvtepi32_ps(x);
}

////////////////////////////////////////////////////////////////////////////////

// Elementary functions

template <typename T, int N> vec<T, N> exp2(vec<T, N> x) {
  // TODO: Check SLEEF 2.80 algorithm

  // Rescale
  vec<T, N> ix = rint(x);
  x -= ix;

  // Polynomial expansion
  vec<T, N> r;
  switch (sizeof(typename vec<T, N>::value_type)) {
  case 4:
    // float, error=1.62772721960621336664735896836e-7
    r = vec<T, N>(0.00133952915439234389712105060319);
    r = muladd(r, x, vec<T, N>(0.009670773148229417605024318985));
    r = muladd(r, x, vec<T, N>(0.055503406540531310853149866446));
    r = muladd(r, x, vec<T, N>(0.240222115700585316818177639177));
    r = muladd(r, x, vec<T, N>(0.69314720007380208630542805293));
    r = muladd(r, x, vec<T, N>(1.00000005230745711373079206024));
    break;
  case 8:
    // double, error=3.74939899823302048807873981077e-14
    r = vec<T, N>(1.02072375599725694063203809188e-7);
    r = muladd(r, x, vec<T, N>(1.32573274434801314145133004073e-6));
    r = muladd(r, x, vec<T, N>(0.0000152526647170731944840736190013));
    r = muladd(r, x, vec<T, N>(0.000154034441925859828261898614555));
    r = muladd(r, x, vec<T, N>(0.00133335582175770747495287552557));
    r = muladd(r, x, vec<T, N>(0.0096181291794939392517233403183));
    r = muladd(r, x, vec<T, N>(0.055504108664525029438908798685));
    r = muladd(r, x, vec<T, N>(0.240226506957026959772247598695));
    r = muladd(r, x, vec<T, N>(0.6931471805599487321347668143));
    r = muladd(r, x, vec<T, N>(1.00000000000000942892870993489));
    break;
  default:
    assert(0);
  }

  // Undo rescaling
  r = ldexp(r, convert_int(ix));

  // TODO: Handle small exponents correctly
  // r = ifthen(x0 < vec<T, N>(vec<T, N>::min_exponent), vec<T, N>(0), r);

  return r;
}

template <typename T, int N> vec<T, N> log2(vec<T, N> x) {
  // Algorithm inspired by SLEEF 2.80

  // Rescale
  typedef typename vec<T, N>::intvec_type ivec;
  ivec ilogb_x = ilogb(x * vec<T, N>(M_SQRT2));
  x = ldexp(x, -ilogb_x);

  vec<T, N> y = (x - vec<T, N>(1)) / (x + vec<T, N>(1));
  vec<T, N> y2 = y * y;

  vec<T, N> r;
  switch (sizeof(typename vec<T, N>::value_type)) {
  case 4:
    // float, error=7.09807175879142775648452461821e-8
    r = vec<T, N>(0.59723611417135718739797302426);
    r = muladd(r, y2, vec<T, N>(0.961524413175528426101613434));
    r = muladd(r, y2, vec<T, N>(2.88539097665498228703236701));
    break;
  case 8:
    // double, error=2.1410114030383689267772704676e-14
    r = vec<T, N>(0.283751646449323373643963474845);
    r = muladd(r, y2, vec<T, N>(0.31983138095551191299118812));
    r = muladd(r, y2, vec<T, N>(0.412211603844146279666022));
    r = muladd(r, y2, vec<T, N>(0.5770779098948940070516));
    r = muladd(r, y2, vec<T, N>(0.961796694295973716912));
    r = muladd(r, y2, vec<T, N>(2.885390081777562819196));
    break;
  default:
    assert(0);
  }
  r *= y;

  // Undo rescaling
  r += convert_float(ilogb_x);

  return r;
}

////////////////////////////////////////////////////////////////////////////////

// Calculate x ++ y = log2(exp2(x) + exp2(y))

// Assume x >= y
// Define r := x - y >= 0

// x ++ y = log2(exp2(x) + exp2(y))
//        = log2(exp2(x) [1 + exp2(y) / exp2(x)])
//        = x + log2(1 + exp2(-r))

// Neutral element
template <typename T> T log_add_neutral() {
  return -numeric_limits<T>::infinity();
}

// Original definition
template <typename T> T log_add(T x, T y) { return log2(exp2(x) + exp2(y)); }

// Modified implementation
template <typename T> T log_add_2(T x, T y) {
  T x1 = fmax(x, y);
  T y1 = fmin(x, y);
  T r = x1 - y1;
  return x1 + log2(T(1) + exp2(-r));
}

// Instantiate templates to examine the generated assembler code
float flog_add(float x, float y) { return log_add(x, y); }
float flog_add_2(float x, float y) { return log_add_2(x, y); }

typedef vec<int, 8> int8;
typedef vec<float, 8> float8;
float8 flog_add(float8 x, float8 y) { return log_add(x, y); }
float8 flog_add_2(float8 x, float8 y) { return log_add_2(x, y); }

int8 filogb(float8 x) { return ilogb(x); }

////////////////////////////////////////////////////////////////////////////////

// Log-sum two arrays xs and ys element-wise
template <typename T, int N>
void vlog_add(T *__restrict__ rs, const T *__restrict__ xs,
              const T *__restrict__ ys, ptrdiff_t n) {
  assert(n % N == 0);
  for (ptrdiff_t i = 0; i < n; i += N) {
    float8 x = vec<T, N>::load(&xs[i]);
    float8 y = vec<T, N>::load(&ys[i]);
    float8 r = log_add_2(x, y);
    r.store(&rs[i]);
  }
}

// Reduce one array via log-sum
template <typename T, int N>
float vlog_sum(const T *__restrict__ xs, ptrdiff_t n) {
  assert(n % N == 0);
  float8 r(log_add_neutral<float>());
  for (ptrdiff_t i = 0; i < n; i += N) {
    float8 x = vec<T, N>::load(&xs[i]);
    r = log_add_2(r, x);
  }
  float r1(log_add_neutral<float>());
  for (int d = 0; d < N; ++d)
    r1 = log_add_2(r1, r[d]);
  return r1;
}

// Instantiate templates to examine the generated assembler code
void fvlog_add(float *__restrict__ rs, const float *__restrict__ xs,
               const float *__restrict__ ys, ptrdiff_t n) {
  vlog_add<float, 8>(rs, xs, ys, n);
}

float fvlog_sum(const float *__restrict__ xs, ptrdiff_t n) {
  return vlog_sum<float, 8>(xs, n);
}

////////////////////////////////////////////////////////////////////////////////

// Get current time
double gettime() {
  timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + 1.0e-6 * tv.tv_usec;
}

// Calculate the relative error
float relerr(float x, float y) { return fabs(x - y) / fmax(fabs(x), fabs(y)); }

int main(int argc, char **argv) {

// Various tests

#if 0
  float x = 19.1;

  int r1 = ilogb(x);
  int r2 = ilogb(float8(x))[0];

  printf("x=%.1f   r1=%d r2=%d   e2=%d\n", x, r1, r2, relerr(r1,r2));
#endif

#if 0
  for (int ix = -100; ix <= +100; ++ix) {
    float x = ix / 10.0;

    float r1 = exp2(x);
    float r2 = exp2(float8(x))[0];

    printf("x=%.1f   r1=%.7f r2=%.7f   e2=%.7g\n", x, r1, r2, relerr(r1, r2));
  }
#endif

#if 0
  for (int ix = -100; ix <= +100; ++ix) {
    float x = ix / 10.0;

    float r1 = log2(fabs(x));
    float r2 = log2(float8(fabs(x)))[0];

    printf("x=%.1f   r1=%.7f r2=%.7f   e2=%.7g\n", x, r1, r2, relerr(r1, r2));
  }
#endif

#if 0
  static const float c[] = {1, 2, 3, 4, 5, 6, 7, 8};
  float8 x = float8::load(c);
  for (int d = 0; d < 8; ++d)
    assert(x[d] == d + 1);
  static float r[8];
  x.store(r);
  for (int d = 0; d < 8; ++d)
    assert(r[d] == d + 1);
#endif

#if 0
  static const float cx[] = {1, 2, 3, 4, 5, 6, 7, 8};
  static const float cy[] = {2.1, 3.2, 4.3, 5.4, 6.5, 7.6, 8.7, 9.8};
  float8 x = float8::load(cx);
  float8 y = float8::load(cy);
  float8 r = log_add_2(x, y);
  static float br[8];
  r.store(br);
  for (int d = 0; d < 8; ++d)
    printf("d=%d x=%.7f y=%.7f r1=%.7f r2=%.7f e2=%.7g\n", d, cx[d], cy[d],
           log_add(cx[d], cy[d]), br[d], relerr(log_add(cx[d], cy[d]), br[d]));
#endif

#if 0
  for (int ix = -100; ix <= +100; ++ix) {
    float x = ix / 10.0;
    for (int iy = -100; iy <= +100; ++iy) {
      float y = iy / 10.0;

      float r1 = log_add(x, y);
      float r2 = log_add_2(x, y);
      float r3 = log_add_2(float8(x), float8(y))[0];

      printf("x=%.1f y=%.1f   r1=%.7f r2=%.7f r3=%.7f   e2=%.7g e3=%.7g\n", x,
             y, r1, r2, r3, relerr(r1, r2), relerr(r1, r3));
    }
  }
#endif

#if 0
  const ptrdiff_t n = 1000;
  vector<float> xs(n), ys(n), rs1(n), rs2(n);
  for (ptrdiff_t i = 0; i < n; ++i) {
    xs[i] = rand() / double(RAND_MAX) * 20 - 10;
    ys[i] = rand() / double(RAND_MAX) * 20 - 10;
    rs1[i] = log_add(xs[i], ys[i]);
  }
  vlog_add<float, 8>(&rs2[0], &xs[0], &ys[0], n);
  for (ptrdiff_t i = 0; i < n; ++i) {
    if (relerr(rs1[i], rs2[i]) > 2.0e-7)
      printf("i=%td x=%.7f y=%.7f r1=%.7f r2=%.7f e2=%.7g\n", i, xs[i], ys[i],
             rs1[i], rs2[i], relerr(rs1[i], rs2[i]));
  }
#endif

#if 0
  const ptrdiff_t n = 1000;
  vector<float> xs(n);
  float rs1 = -std::numeric_limits<float>::max();
  for (ptrdiff_t i = 0; i < n; ++i) {
    xs[i] = rand() / double(RAND_MAX) * 20 - 10;
    rs1 = log_add(rs1, xs[i]);
  }
  float rs2 = vlog_add<float, 8>(&xs[0], n);
  if (relerr(rs1, rs2) > 2.0e-7)
    printf("r1=%.7f r2=%.7f e2=%.7g\n", rs1, rs2, relerr(rs1, rs2));
#endif

#if 0
  const ptrdiff_t n = 4000; // use 128 kByte
  vector<float> xs(n), ys(n), rs1(n), rs2(n);
  for (ptrdiff_t i = 0; i < n; ++i) {
    xs[i] = rand() / double(RAND_MAX) * 20 - 10;
    ys[i] = rand() / double(RAND_MAX) * 20 - 10;
    rs1[i] = 0;
    rs2[i] = 0;
  }
  const int niters = 1000;
  double t0 = gettime();
  for (int iter = 0; iter < niters; ++iter) {
    for (ptrdiff_t i = 0; i < n; ++i) {
      rs1[i] = log_add_2(xs[i], ys[i]);
    }
  }
  double t1 = gettime();
  for (int iter = 0; iter < niters; ++iter) {
    vlog_add<float, 8>(&rs2[0], &xs[0], &ys[0], n);
  }
  double t2 = gettime();
  printf("standard evaluation:   %g ns / call\n",
         1e9 * (t1 - t0) / (niters * n));
  printf("vectorized evaluation: %g ns / call\n",
         1e9 * (t2 - t1) / (niters * n));
#endif

#if 1
  const ptrdiff_t n = 16000;
  vector<float> xs(n);
  for (ptrdiff_t i = 0; i < n; ++i) {
    xs[i] = rand() / double(RAND_MAX) * 20 - 10;
  }
  const int niters = 1000;
  double t0 = gettime();
  float rs1 = -std::numeric_limits<float>::max();
  for (int iter = 0; iter < niters; ++iter) {
    for (ptrdiff_t i = 0; i < n; ++i) {
      rs1 = log_add(rs1, xs[i]);
    }
  }
  double t1 = gettime();
  volatile float use_rs1 = rs1;
  float rs2 = -std::numeric_limits<float>::max();
  for (int iter = 0; iter < niters; ++iter) {
    rs2 = log_add_2(rs2, vlog_sum<float, 8>(&xs[0], n));
  }
  double t2 = gettime();
  volatile float use_rs2 = rs2;
  printf("standard evaluation:   %g ns / call\n",
         1e9 * (t1 - t0) / (niters * n));
  printf("vectorized evaluation: %g ns / call\n",
         1e9 * (t2 - t1) / (niters * n));
#endif

  return 0;
}
