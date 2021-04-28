
################################################################################
#################                  Protein DSP                 #################
################################################################################

import pandas as pd
import numpy as np
from difflib import get_close_matches
from scipy.signal import savgol_filter, blackman, hanning, hamming, bartlett, blackmanharris, kaiser, gaussian
try:
    # use scipy if available: it's faster
    from scipy.fftpack import fft, ifft, fftfreq, rfft, rfftfreq
except:
    from numpy.fft import fft, ifft, fftfreq, rfft, rfftfreq
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import utils as utils
from globals import OUTPUT_FOLDER, OUTPUT_DIR
from aaindex import AAIndex

#complete the func descriptoon
class ProDSP():

    """
    Transform protein sequences into their spectral form via a Fast Fourier Transform (FFT).
    Fourier analysis is fundamentally a method for expressing a function as a sum of periodic components,
    and for recovering the function from those components. When both the function and its Fourier
    transform are replaced with discretized counterparts,it is called the discrete Fourier transform (DFT).
    An implementation algorithm for the DFT is known as the FFT, which is used here. From the FFT
    transformations on the encoded protein sequences, various informational protein spectra
    can be generated, including the power, real, imaginary and absolute spectra.
    Prior to the FFT, a window function can be applied to the sequences which is a mathmatical
    function that applies a weighting to each discrete time series sample in a finite set.
    By default, the hamming function is applied; although the function can accept
    the blackman, blackmanharris, bartlett, gaussian and kaiser window funcs. A
    filter can also be applied, with the savgol filter only currently supported
    in this class with plans to future expansion.

    #insert FFT equation
    Attributes
    ----------
    encoded_sequences : np.ndarray
        protein sequences encoded via a specific AAI index feature value.
        encoded_sequences can be 1 or more sequences, if 1 sequence input, it
        will be reshaped to 2 dimensions. These encoded sequences are used to
        generate the various protein spectra.
    spectrum : str
        protein spectra to generate from the protein sequences.
    window : str
        window function to apply to the output of the FFT.
    filter: str
        filter function to apply to the output of the FFT.

    Methods
    -------

    """
    def __init__(self, encoded_sequences, spectrum='power', window="hamming", filter=None):

        self.encoded_sequences = encoded_sequences

        #if single sequence input then try reshape it to 2 dimensions
        if (encoded_sequences.ndim!=2):
            try:
                self.encoded_sequences.reshape((-1,1))
            except:
                raise ValueError('Error reshaping input sequences.')

        self.spectrum = spectrum
        self.window = window
        self.filter = filter

        #pre-processing of encoded protein sequences
        self.pre_processing()

        #transform sequences into the various informational protein spectra
        self.encode_seqs()

    def pre_processing(self):
        """
        Complete various pre-processing steps to encoded protein sequences before
        doing any of the DSP-related functions or transformations. Zero-pad
        the sequences, remove any +/- infinity or NAN values, get the approximate
        protein spectra and window function parameter names.

        """
        #zero pad encoded sequences so they are all the same length
        self.encoded_sequences = utils.zero_padding(self.encoded_sequences)
        self.num_seqs = self.encoded_sequences.shape[0]
        self.signal_len = self.encoded_sequences.shape[1]

        #replace any positive or negative infinity or NAN values with 0
        self.encoded_sequences[self.encoded_sequences == -np.inf] = 0
        self.encoded_sequences[self.encoded_sequences == np.inf] = 0
        self.encoded_sequences[self.encoded_sequences == np.nan] = 0

        #replace any NAN's with 0's
        # self.encoded_sequences.fillna(0,inplace=True)
        self.encoded_sequences = np.nan_to_num(self.encoded_sequences)

        #initialise zeros array to store all protein spectra
        self.fft_power = np.zeros((self.num_seqs, self.signal_len))
        self.fft_real = np.zeros((self.num_seqs, self.signal_len))
        self.fft_imag = np.zeros((self.num_seqs, self.signal_len))
        self.fft_abs = np.zeros((self.num_seqs, self.signal_len))

        #list of accepted spectra, window functions and filters
        all_spectra = ['power','absolute','real','imaginary']
        all_windows = ['hamming', 'blackman','blackmanharris','gaussian','bartlett',
                       'kaiser']
        all_filters = ['savgol']

        #get closest correct spectra from user input, if no close match then raise error
        spectra_matches = (get_close_matches(self.spectrum, all_spectra, cutoff=0.4))[0]

        if spectra_matches == [] or spectra_matches == None:
            raise ValueError('Invalid input Spectrum type ({}) not available in valid \
                spectra: {}'.format(self.spectrum, all_spectra))
        else:
            self.spectra = spectra_matches

        #get closest correct window function from user input
        window_matches = (get_close_matches(self.window, all_windows, cutoff=0.4))

        #check if sym=True or sym=False

        #get window function specified by window input parameter
        if window_matches != [] and window_matches != None:
            if window_matches[0] == 'hamming':
                self.window = hamming(self.signal_len, sym=True)
            elif window_matches[0] == "blackman":
                self.window = blackman(self.signal_len, sym=True)
            elif window_matches[0] == "blackmanharris":
                self.window = blackmanharris(self.signal_len, sym=True)
            elif window_matches[0] == "bartlett":
                self.window = bartlett(self.signal_len, sym=True)
            elif window_matches[0] == "gaussian":
                self.window = gaussian(self.signal_len, std=7,sym=True)
            elif window_matches[0] == "kaiser":
                self.window = kaiser(self.signal_len, beta=14,sym=True)
        else:
            self.window = 1     #window = 1 is the same as applying no window

        #get closest correct filter from user input
        filter_matches = (get_close_matches(self.window, all_filters, cutoff=0.4))

        #set filter attribute according to approximate user input
        if filter_matches ==[] or filter_matches == None:
            self.filter = ""    #no filter

    def encode_seqs(self):
        """
        Calculate the FFT and RFFT of the protein sequences already encoded using
        AAI indices, then use the output of the FFT to calculate the various
        informational protein spectra including the power, absolute, real and imaginary.

        """
        #create copy of protein sequences so the original instance var remains unchanged
        encoded_seq_copy = np.copy(self.encoded_sequences)

        '''From numpy.rfft documentation:
        If n is even, the length of the transformed axis is (n/2)+1.
        If n is odd, the length is (n+1)/2.

        Simply, set the 2nd dimension of the output array size used to store
        rfft output to (N/2)+1 if n is even and (N+1)/2 if odd, where N is the
        signal length (length of sequences).'''
        if (self.signal_len % 2 ==0):
            rfft_output_size = int((self.signal_len/2) + 1)
        else:
            rfft_output_size = int((self.signal_len+1)/2)

        #initialise zero arrays used to store output of both fft and rfft, set
        #   datatype to complex number as that is the output type of the transformation.
        encoded_freqs_rfft = np.zeros((self.encoded_sequences.shape))
        encoded_dataset_fft = np.zeros((self.encoded_sequences.shape),dtype=complex)

        #initialise zero arrays used to store output frequencies from fft & rfft transformations
        rttf_freq_size = int((rfft_output_size)/2 + 1)
        encoded_freqs_rfft = np.zeros((self.encoded_sequences.shape))
        encoded_freqs_fft = np.zeros(self.encoded_sequences.shape)

        #iterate through each sequence, applying the FFT to each
        for seq in range(0,self.num_seqs):

          encoded_rfft = np.zeros((self.encoded_sequences.shape[1]),dtype=complex)
          encoded_fft = np.zeros((self.encoded_sequences.shape[1]),dtype=complex)

          #apply filter
          # if (self.filter!=""):
          #    if (self.filter == 'savgol'):
          #       encoded_seq_copy[seq] = savgol_filter(encoded_seq_copy[seq], 17, polyorder=2, deriv=2)

          #apply window function to Fourier array
          encoded_rfft = rfft(encoded_seq_copy[seq] *self.window)
          encoded_fft = fft(encoded_seq_copy[seq] *self.window)

          #append transformation from current sequence seq to array of all transformed seqeunces
          encoded_dataset_rfft[seq] = encoded_rfft
          encoded_dataset_fft[seq] = encoded_fft

          #calcualte FFT/RFFT frequencies
          # freqs_rfft = rfftfreq(encoded_rfft.shape[0], self.signal_len)
          freqs_rfft = rfftfreq(encoded_rfft.size)
          freqs_fft = np.fft.fftfreq(encoded_fft.size)

          #append frequency from current sequence seq to array of all frequencies
          encoded_freqs_rfft[seq] = freqs_rfft
          encoded_freqs_fft[seq] = freqs_fft

        #set FFT and RFFT sequences and frequencies
        self.fft = encoded_dataset_fft
        self.rfft = encoded_dataset_rfft
        self.fft_freqs = encoded_freqs_fft
        self.rfft_freqs = encoded_freqs_rfft

        #get individual spectral values, calculated from the FFT and RFFT transformations
        self.fft_abs = abs(self.fft/self.signal_len)
        self.rfft_abs = abs(self.rfft/self.signal_len)
        self.fft_power = np.abs(self.fft[0:len(self.fft)])
        self.rfft_power = np.abs(self.rfft[0:len(self.rfft)])
        self.fft_real = self.fft.real
        self.rfft_real = self.rfft.real
        self.fft_imag = self.fft.imag
        self.rfft_imag = self.rfft.imag

        #set the spectrum_encoding attribute to the spectra specified by spectra
        #   class input parameter.
        if self.spectra == 'power':
            self.spectrum_encoding = self.fft_power
        elif self.spectra == 'real':
            self.spectrum_encoding = self.fft_real
        elif self.spectra == 'imaginary':
            self.spectrum_encoding = self.fft_imag
        elif self.spectra == 'abs':
            self.spectrum_encoding = self.fft_abs

    def inverse_FFT(self, a, n):
        """
        Get the inverse Fourier Transform of FFT.

        Parameters
        ----------
        a : np.ndarray
            input array of 1D Fourier Transform.
        n : int
            length of the output
        Returns
        -------
        inv_FFT: np.ndarray
            array of inverse Fourier Transform.
        """
        inv_FFT =np.fft.ifft(a,n)

        self.invFFT = invFFT

        return invFFT

    def consensus_freq(self, freqs):
        """
        Get the Consensus frequency from Fourier Transform of protein sequences.

        Parameters
        ----------
        freqs : np.ndarray
            frequencies of Fourier Transform.
        Returns
        -------
        CF : float
            consus frequency found in array of frequencies
        CFi : int
            index of consensus frequency
        """
        # CF = PP/N ( peak position/length of largest protein in dataset)
        CF, CFi = (self.max_freq(freqs))/self.num_seqs

        return CF, CFi

    def max_freq(self, freqs):
        """
        Get the maximum frequency from Fourier Transform of protein sequences.

        Parameters
        ----------
        freqs : np.ndarray
            frequencies of Fourier Transform.
        Returns
        -------
        max_F : float
            maximum frequency found in array of frequencies
        max_FI : int
            index of maximum frequency
        """
        max_F = max(freqs)
        max_FI = np.argmax(freqs)

        return max_F, max_FI

######################          Getters & Setters          ######################

    @property
    def fft_power(self):
        return self._fft_power

    @fft_power.setter
    def fft_power(self, val):
        self._fft_power = val

    @property
    def fft_real(self):
        return self._fft_real

    @fft_real.setter
    def fft_real(self, val):
        self._fft_real = val

    @property
    def fft_imag(self):
        return self._fft_imag

    @fft_imag.setter
    def fft_imag(self, val):
        self._fft_imag = val

    @property
    def fft_abs(self):
        return self._fft_abs

    @fft_abs.setter
    def fft_abs(self, val):
        self._fft_abs = val

    @property
    def fft_freqs(self):
        return self._fft_freqs

    @fft_freqs.setter
    def fft_freqs(self, val):
        self._fft_freqs = val

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, val):
        self._window = val

    @property
    def filter(self):
        return self._filter

    @filter.setter
    def filter(self, val):
        self._filter = val

################################################################################

    def __str__(self):
        return "Instance of ProDSP class, using parameters: {}".format(self.__dict__.keys())

    def __repr__(self):
        return ('<ProDSP: {}>'.format(self))
