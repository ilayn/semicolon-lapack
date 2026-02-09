#ifndef SEMICOLON_LAPACK_H
#define SEMICOLON_LAPACK_H

#if defined(_WIN32)
    #if defined(__TINYC__)
        #define __declspec(x) __attribute__((x))
    #endif
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

#endif /* SEMICOLON_LAPACK_H */
