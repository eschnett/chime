# chime
Coding experiments inspired by [CHIME](http://chime.phas.ubc.ca)

## Instructions

Compile e.g. with
```sh
clang++': clang++ -std=c++11 -march=native -O3 -Wall -S complex4.cpp
```
or
```sh
g++': clang++ -std=c++11 -march=native -O3 -Wall -S complex4.cpp
```

I am using g++ 5.3.0 and clang++ 3.7.0.

It seems that `clang++` generates sligthly worse code -- I did not time the code
yet, but it introduces superfluous spills.

Both `clang++` and `g++` handle abstraction very well; one can introduce e.g. a
wrapper class for complex (integer) arithmetic that leads to zero run-time
overhead.

## Observations

- A size-8 Fourier transform already leads to register spilling; a size-16 Fourier transform is definitively too large for 16 registers
- I don't think that using "horizontal vectorization" would be efficient
- Similarly, I don't think that interleaving real and imaginary parts into the same vector would be efficient
- I am currently using 16-bit integers (with 8 fractional bits). This is way too many fractional bits for a size-8 transform. It may make sense to use 8-bit integers instead, handling twice as many elements at once. If one uses 1 fractional bit, then this leaves 3 bits to prevent overflow.
