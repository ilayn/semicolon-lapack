#ifndef SEMICOLON_LAPACK_AUXILIARY_H
#define SEMICOLON_LAPACK_AUXILIARY_H

#include "semicolon_lapack/semicolon_lapack.h"

SEMICOLON_API void   xerbla(const char* srname, int info);
SEMICOLON_API int    ieeeck(const int ispec, const float zero, const float one);
SEMICOLON_API int    ilaenv2stage(const int ispec, const char* name, const char* opts, const int n1, const int n2, const int n3, const int n4);
SEMICOLON_API int    iparam2stage(const int ispec, const char* name, const char* opts, const int ni, const int nbi, const int ibi, const int nxi);
SEMICOLON_API int    iparmq(const int ispec, const char* name, const char* opts, const int n, const int ilo, const int ihi, const int lwork);

#endif /* SEMICOLON_LAPACK_AUXILIARY_H */
