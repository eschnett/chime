#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
using namespace std;

template <ptrdiff_t p, ptrdiff_t q>
void resample1(float *__restrict__ const r, const float *__restrict__ const x,
               const ptrdiff_t nr) {
  assert(p > 0 && q > 0);
  assert(nr % p == 0);
  const ptrdiff_t nx = nr * q / p;

  for (ptrdiff_t i0 = 0; i0 < nr; i0 += p) {
    ptrdiff_t j0 = i0 / p * q;
    for (ptrdiff_t i1 = 0; i1 < p; ++i1) {
      ptrdiff_t j1 = i1 * q / p;
      ptrdiff_t i = i0 + i1;
      ptrdiff_t j = j0 + j1;
      float delta = float(i1) * q / p - j1;
      float x0 = -1.0f;
      float x1 = 0.0f;
      float x2 = +1.0f;
      float x3 = +2.0f;
      float c0 = (delta - x1) * (delta - x2) * (delta - x3) /
                 ((x0 - x1) * (x0 - x2) * (x0 - x3));
      float c1 = (delta - x0) * (delta - x2) * (delta - x3) /
                 ((x1 - x0) * (x1 - x2) * (x1 - x3));
      float c2 = (delta - x0) * (delta - x1) * (delta - x3) /
                 ((x2 - x0) * (x2 - x1) * (x2 - x3));
      float c3 = (delta - x0) * (delta - x1) * (delta - x2) /
                 ((x3 - x0) * (x3 - x1) * (x3 - x2));
      r[i] = c0 * x[j - 1] + c1 * x[j] + c2 * x[j + 1] + c3 * x[j + 2];
    }
  }
}

template <ptrdiff_t p, ptrdiff_t q>
void resample2(float *__restrict__ const r, const float *__restrict__ const x,
               const ptrdiff_t nr) {
  assert(p > 0 && q > 0);
  assert(nr % p == 0);
  const ptrdiff_t nx = nr * q / p;

  for (ptrdiff_t i0 = 0; i0 < nr; i0 += p) {
    ptrdiff_t j0 = i0 / p * q;

    float s[p];
    for (ptrdiff_t i1 = 0; i1 < p; ++i1)
      s[i1] = 0.0f;

    for (ptrdiff_t j1 = -1; j1 < q + 3; ++j1) {

      for (ptrdiff_t i1 = 0; i1 < p; ++i1) {

        float delta = float(i1) * q / p;
        float x0 = i1 * q / p - 1.0f;
        float x1 = i1 * q / p + 0.0f;
        float x2 = i1 * q / p + 1.0f;
        float x3 = i1 * q / p + 2.0f;

        float c;
        ptrdiff_t j1min = i1 * q / p - 1;
        ptrdiff_t j1max = j1min + 4;
        if (j1 == j1min + 0)
          c = (delta - x1) * (delta - x2) * (delta - x3) /
              (-1.0f * -2.0f * -3.0f);
        else if (j1 == j1min + 1)
          c = (delta - x0) * (delta - x2) * (delta - x3) /
              (+1.0f * -1.0f * -2.0f);
        else if (j1 == j1min + 2)
          c = (delta - x0) * (delta - x1) * (delta - x3) /
              (+2.0f * +1.0f * -1.0f);
        else if (j1 == j1min + 3)
          c = (delta - x0) * (delta - x1) * (delta - x2) /
              (+3.0f * +2.0f * +1.0f);
        else
          c = 0.0f;

        s[i1] += c * x[j0 + j1];
      }
    }

    for (ptrdiff_t i1 = 0; i1 < p; ++i1)
      r[i0 + i1] = s[i1];
  }
}

template <ptrdiff_t p, ptrdiff_t q>
void resample3(float *__restrict__ const r, const float *__restrict__ const x,
               const ptrdiff_t nr) {
  assert(p > 0 && q > 0);
  assert(nr % p == 0);
  const ptrdiff_t nx = nr * q / p;

  typedef float vfloat __attribute__((__ext_vector_type__(p)));

  vfloat c[q + 4];
  for (ptrdiff_t j1 = -1; j1 < q + 3; ++j1) {
    for (ptrdiff_t i1 = 0; i1 < p; ++i1) {

      float delta = float(i1) * q / p;
      float x0 = i1 * q / p - 1.0f;
      float x1 = i1 * q / p + 0.0f;
      float x2 = i1 * q / p + 1.0f;
      float x3 = i1 * q / p + 2.0f;

      ptrdiff_t j1min = i1 * q / p - 1;
      ptrdiff_t j1max = j1min + 4;
      if (j1 == j1min + 0)
        c[j1 + 1][i1] = (delta - x1) * (delta - x2) * (delta - x3) /
                        (-1.0f * -2.0f * -3.0f);
      else if (j1 == j1min + 1)
        c[j1 + 1][i1] = (delta - x0) * (delta - x2) * (delta - x3) /
                        (+1.0f * -1.0f * -2.0f);
      else if (j1 == j1min + 2)
        c[j1 + 1][i1] = (delta - x0) * (delta - x1) * (delta - x3) /
                        (+2.0f * +1.0f * -1.0f);
      else if (j1 == j1min + 3)
        c[j1 + 1][i1] = (delta - x0) * (delta - x1) * (delta - x2) /
                        (+3.0f * +2.0f * +1.0f);
      else
        c[j1 + 1][i1] = 0.0f;
    }
  }

  for (ptrdiff_t i0 = 0; i0 < nr; i0 += p) {
    ptrdiff_t j0 = i0 / p * q;

    vfloat s;
    for (ptrdiff_t i1 = 0; i1 < p; ++i1)
      s[i1] = 0.0f;

    for (ptrdiff_t j1 = -1; j1 < q + 3; ++j1)
      for (ptrdiff_t i1 = 0; i1 < p; ++i1)
        s[i1] += c[j1 + 1][i1] * x[j0 + j1];

    for (ptrdiff_t i1 = 0; i1 < p; ++i1)
      r[i0 + i1] = s[i1];
  }
}

template void resample1<3, 5>(float *__restrict__ r,
                              const float *__restrict__ x, ptrdiff_t nr);
template void resample2<3, 5>(float *__restrict__ r,
                              const float *__restrict__ x, ptrdiff_t nr);
template void resample3<3, 5>(float *__restrict__ r,
                              const float *__restrict__ x, ptrdiff_t nr);

int main(int argc, char **argv) {
  const ptrdiff_t p = 3, q = 5;
  const ptrdiff_t nr = p * 10;
  const ptrdiff_t nx = nr * q / p;
  float x0[nx + 16];
  float *const x = &x0[8];
  float r[nr];
  for (ptrdiff_t i = -8; i < nx + 8; ++i)
    x[i] = pow(float(i) / nx, 3);

  resample3<p, q>(r, x, nr);

  for (ptrdiff_t i = 0; i < nr; ++i)
    cout << "i " << i << " " << r[i] - pow(float(i) / nr, 3) << "\n";

  return 0;
}
