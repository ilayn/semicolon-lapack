#ifndef SEMICOLON_LAPACK_AUXILIARY_H
#define SEMICOLON_LAPACK_AUXILIARY_H

#include "semicolon_lapack/semicolon_lapack.h"

SEMICOLON_API void   xerbla(const char* srname, INT info);

typedef void (*xerbla_handler_t)(const char* srname, INT info);
SEMICOLON_API extern xerbla_handler_t xerbla_override;

/* The tuning queries of lapack_tuning.h. When lapack_tuning_override is set,
 * the library answers every tuning query through it; the test suite sets it so
 * that xlaenv() reaches the library where weak symbols cannot (Windows DLLs). */
typedef struct {
    int (*get_nb)(const char* routine);
    int (*get_nbmin)(const char* routine);
    int (*get_nx)(const char* routine);
    int (*get_mnthr)(const char* routine, int m, int n);
    int (*get_geqr_mb)(int m, int n);
    int (*get_geqr_nb)(int m, int n);
    int (*get_gelq_mb)(int m, int n);
    int (*get_gelq_nb)(int m, int n);
    int (*get_trsyl_nb)(int n1, int n2, int is_complex);
    int (*get_gbtrf_nb)(int ku);
    int (*get_pbtrf_nb)(int kd);
    int (*get_tzrzf_nb)(int is_complex);
    int (*get_nbcol)(const char* routine);
    int (*get_iparmq)(int ispec, const char* name, const char* opts,
                      int64_t n, int64_t ilo, int64_t ihi, int64_t lwork);
    int (*get_iparam2stage)(int ispec, const char* name, const char* opts,
                            int64_t ni, int64_t nbi, int64_t ibi, int64_t nxi);
} lapack_tuning_override_t;
SEMICOLON_API extern const lapack_tuning_override_t* lapack_tuning_override;

SEMICOLON_API char   chla_transtype(const INT trans);
SEMICOLON_API INT    ieeeck(const INT ispec, const float zero, const float one);
SEMICOLON_API INT    iladiag(const char* diag);
SEMICOLON_API INT    ilaenv2stage(const INT ispec, const char* name, const char* opts, const INT n1, const INT n2, const INT n3, const INT n4);
SEMICOLON_API INT    ilaprec(const char* prec);
SEMICOLON_API INT    ilatrans(const char* trans);
SEMICOLON_API INT    ilauplo(const char* uplo);
SEMICOLON_API void   ilaver(INT* major, INT* minor, INT* patch);
SEMICOLON_API INT    iparam2stage(const INT ispec, const char* name, const char* opts, const INT ni, const INT nbi, const INT ibi, const INT nxi);
SEMICOLON_API INT    iparmq(const INT ispec, const char* name, const char* opts, const INT n, const INT ilo, const INT ihi, const INT lwork);

#endif /* SEMICOLON_LAPACK_AUXILIARY_H */
