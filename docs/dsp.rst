Digital Signal Processing
==========================

The ``PyDSP`` class (``pySAR/pyDSP.py``) transforms numerically-encoded protein sequences
into frequency-domain spectral features using the Fast Fourier Transform (FFT). These spectral
features can then be used directly as inputs to a regression model, enabling the 
AAI + DSP encoding strategy in pySAR.

.. note::
   ``PyDSP`` operates on **numerically encoded** sequences — not raw amino acid strings.
   Each sequence must first be encoded via an AAIndexEncoding step before being passed
   to ``PyDSP``.

----

Pipeline Overview
-----------------

The DSP encoding pipeline follows these steps:

1. **Numerical encoding** — protein sequences are encoded with physico-chemical AAI indices,
   producing a 2-D array of shape ``(num_sequences, signal_len)``.
2. **Pre-processing** — sequences are zero-padded to a uniform length; ``inf`` and ``NaN``
   values are replaced with ``0``.
3. **Window function** (optional) — a window is element-wise multiplied with each encoded
   sequence before the FFT to reduce spectral leakage.
4. **FFT** — ``scipy.fft.fft`` is applied to each sequence independently.
5. **Filter function** (optional) — a smoothing or analytic filter is applied to the FFT
   output of each sequence.
6. **Spectrum extraction** — the desired spectral representation (power, absolute, real,
   or imaginary) is extracted from the complex FFT output.

The resulting ``spectrum_encoding`` array (shape ``(num_sequences, signal_len)``) is the
feature matrix used for model training.

----

Instantiation
-------------

``PyDSP.__init__(config_file, protein_seqs, **kwargs)``

.. code-block:: python

    from pySAR.pyDSP import PyDSP

    # from a config file path
    dsp = PyDSP(config_file="config/thermostability.json", protein_seqs=encoded_seqs)

    # from a config dict
    dsp = PyDSP(config_file={"pyDSP": {"spectrum": "power", "window_type": "hamming"}},
                protein_seqs=encoded_seqs)

    # via keyword arguments (override or replace config values)
    dsp = PyDSP(config_file=None, protein_seqs=encoded_seqs,
                spectrum="absolute", window_type="blackman")

All ``[pyDSP]`` config keys are also accepted as keyword arguments and will override
any value read from the config file.

----

Pre-processing
--------------

Calling ``pre_processing()`` prepares the encoded sequences for the FFT:

- Zero-pads all sequences to the length of the longest sequence in the dataset
  (``signal_len``), ensuring a uniform array shape.
- Replaces any ``inf`` or ``NaN`` values with ``0`` to prevent FFT errors.
- Initialises the ``fft_power``, ``fft_real``, ``fft_imag``, and ``fft_abs`` zero arrays
  ready to receive results from ``encode_sequences()``.

``pre_processing()`` is called automatically before ``encode_sequences()``.

----

Spectra
-------

The ``spectrum`` parameter controls which spectral representation is extracted from the
FFT output. The selected spectrum is stored in ``dsp.spectrum_encoding``.

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Spectrum
     - Attribute
     - Description
   * - ``power``
     - ``fft_power``
     - Magnitude of the FFT output: $|X[k]|$ (absolute value of the complex FFT array).
   * - ``absolute``
     - ``fft_abs``
     - Normalised absolute spectrum: $|X[k]| / N$ where $N$ is ``signal_len``.
   * - ``real``
     - ``fft_real``
     - Real part of the complex FFT output: $\operatorname{Re}(X[k])$.
   * - ``imaginary``
     - ``fft_imag``
     - Imaginary part of the complex FFT output: $\operatorname{Im}(X[k])$.

Set the spectrum in the config file or as a keyword argument:

.. code-block:: json

    {
        "pyDSP": {
            "spectrum": "power"
        }
    }

----

Window Functions
----------------

A window function tapers the encoded signal at its edges before the FFT is applied,
reducing spectral leakage. Set ``window_type`` to one of the values below. If
``window_type`` is ``null`` or omitted, no window is applied (equivalent to a
rectangular window of ones).

Window names are matched approximately, so ``"hamm"`` resolves to ``"hamming"``, etc.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Window
     - Description
   * - ``hamming``
     - Raised cosine with non-zero endpoints; good general-purpose choice.
   * - ``blackman``
     - Three-term cosine window; lower sidelobe levels than Hamming.
   * - ``blackmanharris``
     - Four-term cosine window with very low sidelobe levels.
   * - ``bartlett``
     - Triangular window with zero-valued endpoints.
   * - ``gaussian``
     - Bell-curve shaped window (default ``std=7``); smooth frequency response.
   * - ``kaiser``
     - Flexible window controlled by ``beta``; balances main-lobe width against sidelobe amplitude.
   * - ``hann``
     - Raised cosine with zero-valued endpoints; minimises spectral leakage.
   * - ``barthann``
     - Combination of Bartlett and Hann windows.
   * - ``bohman``
     - Convolution of two half-duration Bartlett windows; very low sidelobes.
   * - ``chebwin``
     - Dolph-Chebyshev window; equiripple sidelobes (default attenuation ``at=100`` dB).
   * - ``cosine``
     - Single half-period cosine window.
   * - ``exponential``
     - Exponentially decaying window.
   * - ``flattop``
     - Flat passband; accurate amplitude measurements in the frequency domain.
   * - ``boxcar``
     - Rectangular window (no tapering); useful as a reference.
   * - ``nuttall``
     - Continuous first-derivative window; very low sidelobes.
   * - ``parzen``
     - B-spline approximation window; smooth tapering.
   * - ``triang``
     - Triangular window with non-zero endpoints.
   * - ``tukey``
     - Cosine-tapered rectangular (top-hat) window, controlled by ``alpha``.

Set ``window_type`` in the config or as a kwarg, and pass optional window parameters
via ``window_parameters``:

.. code-block:: json

    {
        "pyDSP": {
            "window_type": "gaussian",
            "window_parameters": {
                "std": 10
            }
        }
    }

----

Filter Functions
----------------

An optional filter can be applied to each sequence's FFT output after the transform.
Set ``filter_type`` to one of the values below. If ``filter_type`` is ``null`` or
omitted, no filter is applied.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Filter
     - Description
   * - ``savgol``
     - Savitzky-Golay smoothing filter (``scipy.signal.savgol_filter``). Fits successive sub-sets of adjacent data points with a low-degree polynomial to smooth noisy spectra while preserving peak shape.
   * - ``medfilt``
     - Median filter (``scipy.signal.medfilt``). Replaces each sample with the median of its neighbours; effective at removing spike noise.
   * - ``lfilter``
     - Direct-form II transposed IIR filter (``scipy.signal.lfilter``). Requires ``b`` and ``a`` numerator/denominator coefficients to be supplied in ``filter_parameters``.
   * - ``hilbert``
     - Hilbert transform (``scipy.signal.hilbert``). Computes the analytic signal; the imaginary part is the Hilbert transform of the input.

Set ``filter_type`` and optional ``filter_parameters`` in the config:

.. code-block:: json

    {
        "pyDSP": {
            "filter_type": "savgol",
            "filter_parameters": {
                "window_length": 11,
                "polyorder": 2
            }
        }
    }

----

encode_sequences
----------------

``encode_sequences()`` is the main computation method. It iterates over each numerically
encoded protein sequence and:

1. Multiplies by the window function (or ``1`` if no window).
2. Applies ``scipy.fft.fft`` to obtain the complex FFT.
3. Applies the filter function to the FFT output (if configured).
4. Stores all four spectral arrays: ``fft_power``, ``fft_abs``, ``fft_real``, ``fft_imag``.
5. Sets ``spectrum_encoding`` to the array selected by the ``spectrum`` parameter.
6. Computes and stores FFT frequencies in ``fft_freqs``.

.. code-block:: python

    dsp = PyDSP(config_file="config/thermostability.json", protein_seqs=encoded_seqs)
    dsp.encode_sequences()

    # access the spectral feature matrix
    features = dsp.spectrum_encoding   # shape: (num_sequences, signal_len)

----

Utility Methods
---------------

``inverse_fft(a, n)``
    Returns the inverse Fourier Transform of array ``a`` truncated/padded to length ``n``
    (wraps ``numpy.fft.ifft``). The result is stored in ``dsp.inv_fft``.

    .. code-block:: python

        reconstructed = dsp.inverse_fft(dsp.fft[0], n=len(dsp.fft[0]))

``consensus_freq(freqs)``
    Computes the Consensus Frequency (CF) for a single encoded sequence:

    .. math::

        CF = \frac{\text{peak position}}{N}

    where *N* is the total number of sequences in the dataset. Accepts a 1-D array of
    FFT frequencies for one sequence.

``max_freq(freqs)``
    Returns ``(max_F, max_FI)`` — the maximum frequency value and its index — from a
    1-D array of FFT frequencies for a single sequence.

----

Config File
-----------

All DSP options are set under the ``[pyDSP]`` key in the config JSON:

.. code-block:: json

    {
        "pyDSP": {
            "spectrum": "power",
            "window_type": "hamming",
            "window_parameters": {},
            "filter_type": "",
            "filter_parameters": {}
        }
    }

Example config files for each dataset can be found in the ``config/`` directory of
the repository:

- `thermostability.json <https://github.com/amckenna41/pySAR/blob/master/config/thermostability.json>`_
- `absorption.json <https://github.com/amckenna41/pySAR/blob/master/config/absorption.json>`_
- `enantioselectivity.json <https://github.com/amckenna41/pySAR/blob/master/config/enantioselectivity.json>`_
- `localization.json <https://github.com/amckenna41/pySAR/blob/master/config/localization.json>`_
