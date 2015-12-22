# chime
Coding experiments inspired by [CHIME](http://chime.phas.ubc.ca)

Compile e.g. with
```sh
clang++ -std=c++11 -march=native -O3 -Wall -S complex4.cpp
```
or
```sh
clang++ -std=c++11 -march=native -O3 -Wall -S complex4.cpp
```

I am using g++ 5.3.0 and clang++ 3.7.0, and an Intel Haswell CPU providing AVX2
instructions.

It seems that `clang++` generates sligthly worse code -- I did not time the code
yet, but it introduces superfluous spills.

Both `clang++` and `g++` handle abstraction very well; one can introduce e.g. a
wrapper class for complex (integer) arithmetic that leads to zero run-time
overhead.
