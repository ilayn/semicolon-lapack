#ifndef SEMICOLON_TYPES_H
#define SEMICOLON_TYPES_H

#include <stdint.h>

typedef int32_t         i32;
typedef int64_t         i64;
typedef float           f32;
typedef double          f64;

#ifdef _MSC_VER
    #include <complex.h>
    typedef _Fcomplex       c64;
    typedef _Dcomplex       c128;
#else
    typedef float _Complex  c64;
    typedef double _Complex c128;
#endif

#endif /* SEMICOLON_TYPES_H */
