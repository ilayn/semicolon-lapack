============
Contributing
============

Documenting C Code
-------------------

semicolon-lapack uses Doxygen-style comments that are processed by Breathe
into Sphinx documentation.

Comment Style
^^^^^^^^^^^^^

Use Javadoc-style comments with ``@param``:

.. code-block:: c

   /**
    * Brief description (first sentence ends with period).
    *
    * Detailed description can span multiple paragraphs.
    *
    * @param[in]     m     Number of rows of matrix A (m >= 0)
    * @param[in]     n     Number of columns of matrix A (n >= 0)
    * @param[in,out] A     On entry, the M-by-N matrix A.
    *                      On exit, the factors L and U from A = P*L*U.
    * @param[in]     lda   Leading dimension of A (lda >= max(1,m))
    * @param[out]    ipiv  Pivot indices, dimension min(m,n). Row i was
    *                      interchanged with row ipiv[i].
    * @param[out]    info  = 0: successful exit
    *                      < 0: if -i, the i-th argument had an illegal value
    *                      > 0: if i, U(i,i) is exactly zero
    */
   void dgetrf(const int m, const int n, double* const restrict A,
               const int lda, int* const restrict ipiv, int* info);

Parameter Direction
^^^^^^^^^^^^^^^^^^^

Always specify parameter direction:

- ``@param[in]`` — Input only, not modified
- ``@param[out]`` — Output only, input value ignored
- ``@param[in,out]`` — Both input and output

Building Documentation
----------------------

The documentation pipeline uses Doxygen (C) and Sphinx (Python), connected by
the Breathe extension. You will need:

- `Doxygen <https://www.doxygen.nl/>`_ for parsing C source files
- Python 3 with the following packages:

  - `Sphinx <https://www.sphinx-doc.org/>`_ >= 7.2
  - `Breathe <https://breathe.readthedocs.io/>`_ >= 4.35 (Doxygen-to-Sphinx bridge)
  - `pydata-sphinx-theme <https://pydata-sphinx-theme.readthedocs.io/>`_ >= 0.15
  - `sphinx-copybutton <https://sphinx-copybutton.readthedocs.io/>`_ >= 0.5

These are also listed in ``doc/requirements.txt`` for convenience:

.. code-block:: bash

   pip install -r doc/requirements.txt

(or ``conda``, ``uv`` or whichever tool you use)

Then build and serve locally:

.. code-block:: bash

   cd doc
   make html
   cd build/html && python -m http.server 8000

The output will be in ``doc/build/html/`` and served at
``http://localhost:8000``.

.. note ::
   If you prefer Doxygen's native HTML output (no judgement), you can run
   ``doxygen doc/source/Doxyfile`` directly and skip the Python toolchain
   entirely. We specifically avoid the caller diagrams and source browser
   that Doxygen generates. There is no technical reason, we just don't
   like them.
