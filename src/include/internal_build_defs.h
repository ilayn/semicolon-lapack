#ifndef SEMICOLON_INTERNAL_BUILD_DEFS_H
#define SEMICOLON_INTERNAL_BUILD_DEFS_H

#include <stdint.h>

typedef int32_t i32;
typedef int64_t i64;

/* Integer type for all LAPACK integer parameters, local variables, and arrays.
   LP64: resolves to i32. ILP64: resolves to i64. */
#ifdef SEMICOLON_ILP64
    #define INT i64
#else
    #define INT i32
#endif

#endif
