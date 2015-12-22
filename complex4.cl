#include <stdint.h>

// OpenCL C notation

// The notation is shorter, but the generated code is less efficient

typedef int8_t int8x32 __attribute__((__ext_vector_type__(32)));

typedef struct {
  int8x32 s0, s1;
} int8x64;

int8x64 split_complex(int8x32 z) {
  int8x32 lomask = 0x0f;
  int8x32 x = (z & lomask) - (int8_t)8;
  int8x32 y = ((z >> 4) & lomask) - (int8_t)8;
  int8x64 r = {x, y};
  return r;
}
