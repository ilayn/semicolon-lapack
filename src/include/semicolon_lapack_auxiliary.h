#ifndef SEMICOLON_LAPACK_AUXILIARY_H
#define SEMICOLON_LAPACK_AUXILIARY_H

#include "semicolon_lapack/semicolon_lapack.h"

SEMICOLON_API void   xerbla(const char* srname, INT info);

typedef void (*xerbla_handler_t)(const char* srname, INT info);
SEMICOLON_API extern xerbla_handler_t xerbla_override;

SEMICOLON_API INT    ieeeck(const INT ispec, const float zero, const float one);
SEMICOLON_API INT    ilaenv2stage(const INT ispec, const char* name, const char* opts, const INT n1, const INT n2, const INT n3, const INT n4);
SEMICOLON_API INT    iparam2stage(const INT ispec, const char* name, const char* opts, const INT ni, const INT nbi, const INT ibi, const INT nxi);
SEMICOLON_API INT    iparmq(const INT ispec, const char* name, const char* opts, const INT n, const INT ilo, const INT ihi, const INT lwork);

#endif /* SEMICOLON_LAPACK_AUXILIARY_H */
