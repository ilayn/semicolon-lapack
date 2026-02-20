# tests/s/meson.build
# Single precision test suite

subdir('testutils')

test_inc_s = include_directories('..', 'testutils')

# =============================================================================
# LAPACK Test Driver Ports (schk*.f and sdrv*.f)
# =============================================================================

lapack_tests_s = {
  # Comprehensive test drivers
  'schkge': ['test_schkge.c', 'lu'],
  'schkgb': ['test_schkgb.c', 'band'],
  'schkpb': ['test_schkpb.c', 'band'],
  'schkpo': ['test_schkpo.c', 'cholesky'],
  'schkpp': ['test_schkpp.c', 'cholesky'],
  'schkps': ['test_schkps.c', 'cholesky'],
  'schkgt': ['test_schkgt.c', 'tridiag'],
  'schkpt': ['test_schkpt.c', 'tridiag'],
  'schksy': ['test_schksy.c', 'symmetric'],
  'schksy_aa': ['test_schksy_aa.c', 'symmetric'],
  'schksy_aa_2stage': ['test_schksy_aa_2stage.c', 'symmetric'],
  'schksy_rk': ['test_schksy_rk.c', 'symmetric'],
  'schksy_rook': ['test_schksy_rook.c', 'symmetric'],
  'schksp': ['test_schksp.c', 'symmetric'],
  'schkqr': ['test_schkqr.c', 'qr'],
  'schklq': ['test_schklq.c', 'qr'],
  'schkql': ['test_schkql.c', 'qr'],
  'schkrq': ['test_schkrq.c', 'qr'],
  'schkq3': ['test_schkq3.c', 'qr'],
  'schkqp3rk': ['test_schkqp3rk.c', 'qr'],
  'schktz': ['test_schktz.c', 'qr'],
  'schklqt': ['test_schklqt.c', 'qr'],
  'schklqtp': ['test_schklqtp.c', 'qr'],
  'schkqrt': ['test_schkqrt.c', 'qr'],
  'schkqrtp': ['test_schkqrtp.c', 'qr'],
  'schkorhr_col': ['test_schkorhr_col.c', 'qr'],
  'schktsqr': ['test_schktsqr.c', 'qr'],
  'schktr': ['test_schktr.c', 'triangular'],
  'schktp': ['test_schktp.c', 'triangular'],
  'schktb': ['test_schktb.c', 'triangular'],
  'schkeq': ['test_schkeq.c', 'condition'],
  'schkrfp': ['test_schkrfp.c', 'rfp'],
  'schkbl': ['test_schkbl.c', 'eigen'],
  'schkbk': ['test_schkbk.c', 'eigen'],
  'schkgl': ['test_schkgl.c', 'eigen'],
  'schkgk': ['test_schkgk.c', 'eigen'],
  'schkec': ['test_schkec.c', 'eigen'],
  'schkgg': ['test_schkgg.c', 'eigen'],
  'schkhs': ['test_schkhs.c', 'eigen'],
  'schkbb': ['test_schkbb.c', 'eigen'],
  'schksb': ['test_schksb.c', 'eigen'],
  'schksb2stg': ['test_schksb2stg.c', 'eigen'],
  'schkbd': ['test_schkbd.c', 'svd'],
  'sckcsd': ['test_sckcsd.c', 'eigen'],

  # Expert driver tests
  'sdrvgb': ['test_sdrvgb.c', 'band'],
  'sdrvpb': ['test_sdrvpb.c', 'band'],
  'sdrvge': ['test_sdrvge.c', 'lu'],
  'sdrvpo': ['test_sdrvpo.c', 'cholesky'],
  'sdrvpp': ['test_sdrvpp.c', 'cholesky'],
  'sdrvsy': ['test_sdrvsy.c', 'symmetric'],
  'sdrvsy_aa': ['test_sdrvsy_aa.c', 'symmetric'],
  'sdrvsy_aa_2stage': ['test_sdrvsy_aa_2stage.c', 'symmetric'],
  'sdrvsy_rk': ['test_sdrvsy_rk.c', 'symmetric'],
  'sdrvsy_rook': ['test_sdrvsy_rook.c', 'symmetric'],
  'sdrvsp': ['test_sdrvsp.c', 'symmetric'],
  'sdrvgt': ['test_sdrvgt.c', 'tridiag'],
  'sdrvpt': ['test_sdrvpt.c', 'tridiag'],
  'sdrvls': ['test_sdrvls.c', 'ls'],
  'sdrvbd': ['test_sdrvbd.c', 'svd'],
  'sdrves': ['test_sdrves.c', 'eigen'],
  'sdrvev': ['test_sdrvev.c', 'eigen'],
  'sdrvsx': ['test_sdrvsx.c', 'eigen'],
  'sdrvvx': ['test_sdrvvx.c', 'eigen'],
  'sdrvst': ['test_sdrvst.c', 'eigen'],
  'sdrvst2stg': ['test_sdrvst2stg.c', 'eigen'],
  'schkst': ['test_schkst.c', 'eigen'],
  'schkst2stg': ['test_schkst2stg.c', 'eigen'],
  'sckglm': ['test_sckglm.c', 'glm'],
  'scklse': ['test_scklse.c', 'lse'],
  'sckgqr': ['test_sckgqr.c', 'gqr'],
  'sckgsv': ['test_sckgsv.c', 'gsv'],
  'sdrvrf1': ['test_sdrvrf1.c', 'rfp'],
  'sdrvrf2': ['test_sdrvrf2.c', 'rfp'],
  'sdrvrf3': ['test_sdrvrf3.c', 'rfp'],
  'sdrvrf4': ['test_sdrvrf4.c', 'rfp'],
  'sdrgev': ['test_sdrgev.c', 'eigen'],
  'sdrgev3': ['test_sdrgev3.c', 'eigen'],
  'sdrgvx': ['test_sdrgvx.c', 'eigen'],
  'sdrges': ['test_sdrges.c', 'eigen'],
  'sdrges3': ['test_sdrges3.c', 'eigen'],
  'sdrgsx': ['test_sdrgsx.c', 'eigen'],
  'sdrvsg': ['test_sdrvsg.c', 'eigen'],
  'sdrvsg2stg': ['test_sdrvsg2stg.c', 'eigen'],
}

foreach name, info : lapack_tests_s
  src = info[0]
  suite = info[1]

  exe = executable('test_' + name,
    src,
    include_directories: test_inc_s,
    dependencies: [verify_dep_s, cmocka_dep],
  )

  test(name, exe,
    suite: suite,
    timeout: 120,
    is_parallel: true,
  )
endforeach

subdir('smoke')
