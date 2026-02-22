#ifndef SEMICOLON_LAPACK_H
#define SEMICOLON_LAPACK_H

#if defined(_WIN32)
    #if defined(SEMICOLON_BUILD_SHARED)
        #define SEMICOLON_API __declspec(dllexport)
    #elif defined(SEMICOLON_USE_SHARED)
        #define SEMICOLON_API __declspec(dllimport)
    #endif
#else
    #if defined(SEMICOLON_BUILD_SHARED)
        #define SEMICOLON_API __attribute__((visibility("default")))
    #endif
#endif

#ifndef SEMICOLON_API
    #define SEMICOLON_API
#endif

#include "types.h"

#ifdef SEMICOLON_ILP64
    #define INT i64
    #ifndef LAPACK_NAME
        #define LAPACK_NAME(name) name##_64
    #endif
    #include "lapack_name_map.h"
#else
    #define INT i32
#endif
#include "semicolon_lapack_auxiliary.h"
#include "semicolon_lapack_double.h"
#include "semicolon_lapack_single.h"
#include "semicolon_lapack_complex_double.h"
#include "semicolon_lapack_complex_single.h"

#endif /* SEMICOLON_LAPACK_H */
