#ifndef SEMICOLON_LAPACK_SINGLE_H
#define SEMICOLON_LAPACK_SINGLE_H

#include "semicolon_lapack/semicolon_lapack.h"

/* xerbla: common error handler */
#ifndef XERBLA_DECLARED
#define XERBLA_DECLARED
SEMICOLON_API void xerbla(const char* srname, int info);
#endif
void sgetf2(const int m, const int n, float* const restrict A, const int lda, int* const restrict ipiv, int* info);
void sgetrf(const int m, const int n, float* const restrict A, const int lda, int* const restrict ipiv, int* info);
void sgetrs(const char* trans, const int n, const int nrhs, const float* const restrict A, const int lda, const int* const restrict ipiv, float* const restrict B, const int ldb, int* info);
void slaswp(const int n, float* const restrict A, const int lda, const int k1, const int k2, const int* const restrict ipiv, const int incx);
void slag2d(const int m, const int n, const float* const restrict SA, const int ldsa, double* const restrict A, const int lda, int* info);
int  sisnan(const float sin);
void spotf2(const char* uplo, const int n, float* const restrict A, const int lda, int* info);
void spotrf(const char* uplo, const int n, float* const restrict A, const int lda, int* info);
void spotrs(const char* uplo, const int n, const int nrhs, const float* const restrict A, const int lda, float* const restrict B, const int ldb, int* info);

#endif /* SEMICOLON_LAPACK_SINGLE_H */
