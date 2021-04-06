
#########################################################################
###                         Protein DSP                               ###
#########################################################################

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

##FIX RFFT stuff!
class ProDSP():

    """
    Transform protein sequences into their spectral form via a Fast Fourier Transform (FFT).
    Fourier analysis is fundamentally a method for expressing a function as a sum of periodic components,
    and for recovering the function from those components. When both the function and its Fourier
    transform are replaced with discretized counterparts,it is called the discrete Fourier transform (DFT).
    An implementation algorithm for the DFT is known as the FFT, which is used here. From the FFT
    transformations on the encoded protein sequences, various informational protein spectra
    can be generated, including the power, real, imaginary and absolute spectra.
    Prior to the FFT, a window function can be applied to the sequences which are a mathmatical
    function that applies a weighting to each discrete time series sample in a finite set.
    By default, the hamming function is applied; all available window functions available can be found
    in the scipy.window documentation.


    Parameters
    ----------
    encoded_sequences : numpy array
        protein sequences encoded via a specific AAI index feature value.
        encoded_sequences has to be at least 2 dimensions, containing at least more
        than 1 protein seqence. These encoded sequences are used to generate the various
        protein spectra.
    spectrum : str
        protein spectra to generate from the protein sequences:
    window : str
        window function to apply to the output of the FFT.
    filter: str
        filter function to apply to the output of the FFT.

    Returns
    -------

    """
    def __init__(self, encoded_sequences, spectrum='power', window="hamming", filter=""):

        self.encoded_sequences = encoded_sequences
        assert self.encoded_sequences.ndim == 2, 'Input sequences not of correct dimension 2'
        self.spectrum = spectrum
        self.window = window
        self.filter = filter

        #do pre-processing of encoded protein sequences
        self.pre_processing()

        #transform sequences into the various informational protein spectra
        self.encode_seqs()

    def pre_processing(self):

        self.encoded_sequences = utils.zero_padding(self.encoded_sequences)
        self.num_seqs = self.encoded_sequences.shape[0]
        self.signal_len = self.encoded_sequences.shape[1]    #assumption seqs are all same len

        #replace any positive or negative infinity with NAN
        self.encoded_sequences[self.encoded_sequences == -np.inf] = 0
        self.encoded_sequences[self.encoded_sequences == np.inf] = 0
        self.encoded_sequences[self.encoded_sequences == np.nan] = 0
        # self.encoded_sequences.replace([np.inf,-np.inf], np.nan)

        #replace any NAN with 0's
        # self.encoded_sequences.fillna(0,inplace=True)

        #remove any NA's or missing values
        self.fft_power = np.zeros((self.num_seqs, self.signal_len))
        self.fft_real = np.zeros((self.num_seqs, self.signal_len))
        self.fft_imag = np.zeros((self.num_seqs, self.signal_len))
        self.fft_abs = np.zeros((self.num_seqs, self.signal_len))

        all_spectra = ['power','absolute','real','imaginary']
        all_windows = ['hamming', 'blackman','blackmanharris','gaussian','bartlett',
                       'kaiser']
        all_filters = []

        spectra_matches = (get_close_matches(self.spectrum, all_spectra, cutoff=0.4))[0]

        if spectra_matches == [] or spectra_matches == None:
            raise ValueError('Invalid input Spectrum type ({}) not available in valid \
                spectrums: {}'.format(self.spectrum, all_spectra))
        else:
            self.spectra = spectra_matches

        # self.spectra = (get_close_matches(self.spectrum, all_spectra, cutoff=0.4))[0]


        window_matches = (get_close_matches(self.window, all_windows, cutoff=0.4))[0]

        if window_matches == [] or window_matches == None:      #change this to just use no window
            raise ValueError('Invalid window function type ({}) not available in valid \
                windows: {}'.format(self.window, all_windows))
        else:
            # self.window = window_matches
            if window_matches == 'hamming':
                self.window = hamming(self.signal_len, sym=True)
            elif window_matches == "blackman":
                self.window = blackman(self.signal_len, sym=True)

        # window = (get_close_matches(self.window, all_windows, cutoff=0.4))[0]
        #
        # if window == 'hamming':
        #     self.window = hamming(self.signal_len, sym=True)
        # else:
        #     self.window = ""

    def encode_seqs(self):

        #create copy of protein sequences so the original instance var remains unchanged
        encoded_seq_copy = np.copy(self.encoded_sequences)
        # print('encoded_seq_copy',encoded_seq_copy.shape)

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
        #datatype to complex number as that is the output type of the transformation.
        # encoded_dataset_rfft = np.zeros((self.num_seqs,rfft_output_size),dtype=complex)
        encoded_dataset_rfft = np.zeros((rfft_output_size),dtype=complex)
        encoded_dataset_fft = np.zeros((self.encoded_sequences.shape),dtype=complex)
        # print('encoded_dataset_rfft',encoded_dataset_rfft.shape)

        #initialise zero arrays used to store output frequencies from fft and rfft
        #   transformations
        rttf_freq_size = int((rfft_output_size)/2 + 1)
        encoded_freqs_rfft = np.zeros((self.num_seqs,rttf_freq_size))
        encoded_freqs_fft = np.zeros(self.encoded_sequences.shape)

        for seq in range(0,self.num_seqs):

          # encoded_rfft = np.zeros((self.signal_len),dtype=complex)       #dont think I'm using this properly
          encoded_rfft = np.zeros((rfft_output_size),dtype=complex)       #dont think I'm using this properly
          encoded_fft = np.zeros((self.encoded_sequences.shape[1]),dtype=complex)
          # print('encoded_rfft',encoded_rfft.shape)

          # if (self.filter!=""):
          #    if (self.filter == 'savgol'):
          #       encoded_seq_copy[seq] = savgol_filter(encoded_seq_copy[seq], 17, polyorder=2, deriv=2)

          #if window passed into class then apply window function to sequences when
          # calculating the fft or rfft transformations for current sequence seq

          if self.window != "":
              w = self.window
            # if self.window == 'blackman':
            #     w = blackman(self.signal_len)
            # elif self.window == 'hamming':
            #     w = hamming(self.signal_len, sym=True)
            # elif self.window == 'blackmanharris':
            #     w = blackmanharris(self.signal_len)
            # elif self.window == 'gaussian':
            #     w = gaussian(self.signal_len, std=7)
            # elif self.window == 'bartlett':
            #     w = bartlett(self.signal_len)
            # elif self.window == 'kaiser':
            #     w = kaiser(self.signal_len, beta=14)

              encoded_rfft = rfft(encoded_seq_copy[seq] *w)
              encoded_fft = fft(encoded_seq_copy[seq] *w)

          else:
            #apply no window function and caluclate fft/rfft transformations for
            #   current sequence seq
            encoded_rfft = rfft(encoded_seq_copy[seq])
            encoded_fft = fft(encoded_seq_copy[seq])

          #append transformation from current sequence seq to array of all transformed
          # seqeunces
          # print('erfft',encoded_rfft.shape)
          # print('encoded_erfft',encoded_dataset_rfft.shape)
          # encoded_dataset_rfft[seq] = encoded_rfft
          encoded_dataset_fft[seq] = encoded_fft

          #calcualte FFT/RFFT frequencies
          # freqs = rfftfreq(encoded_rfft.shape[0], self.signal_len)
          freqs_rfft = rfftfreq(encoded_rfft.size, self.signal_len)
          freqs_fft = np.fft.fftfreq(encoded_fft.size)

          #append frequency from current sequence seq to array of all frequencies
          # encoded_freqs_rfft[seq] = freqs_rfft
          encoded_freqs_fft[seq] = freqs_fft

        #set FFT and RFFT sequences and frequencies instance variables
        self.fft = encoded_dataset_fft
        self.rfft = encoded_dataset_rfft
        self.fft_freqs = encoded_freqs_fft
        self.rfft_freqs = encoded_freqs_rfft

        #set spectral instance varibales, calculated from the FFT and RFFT transformations
        self.fft_abs = abs(self.fft/self.signal_len)
        self.rfft_abs = abs(self.rfft/self.signal_len)
        self.fft_power = np.abs(self.fft[0:len(self.fft)])
        self.rfft_power = np.abs(self.rfft[0:len(self.rfft)])
        self.fft_real = self.fft.real
        self.rfft_real = self.rfft.real
        self.fft_imag = self.fft.imag
        self.rfft_imag = self.rfft.imag

        # self.encode_seqs()
        if self.spectra == 'power':
            self.spectrum_encoding = self.fft_power
        elif self.spectra == 'real':
            self.spectrum_encoding = self.fft_real
        elif self.spectra == 'imaginary':
            self.spectrum_encoding = self.fft_imag
        elif self.spectra == 'abs':
            self.spectrum_encoding = self.fft_abs



    def plot_freq(self, fft_seq, fft_freqs):

        #plot_freq(self.fft[0], self.freqs[0])
        xf = fft_freqs
        yf = fft_seq    #sequence that has already been transformed via FFT

        #get positive frequencies and peak freq
        pos_mask = np.where(xf > 0)
        pos_freqs = xf[pos_mask]

        print('in plot')
        xf = np.concatenate((fft_freqs.real, fft_freqs.imag),axis = 1)
        # xf = np.delete(xf, xf[:,:-2],axis =1)
        xf = xf[:,:-2]
        print('xf',xf.shape)
        print(fft_freqs.real.shape)
        print(fft_freqs.imag.shape)

        plt.plot(xf, (yf))
        plt.xlabel('Frequecies')
        plt.ylabel('Amplitude')
        plt.title('abc')
        plt.grid()
        plt.show()

    def inverse_FFT(self,n):

        invFFT =np.fft.ifft(n)

        self.invFFT = invFFT

        return invFFT

    def consensus_freq(self):

        # CF = PP/N ( peak position/length of largest protein in dataset)
        CF, CFi = (self.max_freq(self.fft_freqs))/self.num_seqs

        return CF, CFi

    def max_freq(self, freqs):

        maxF = max(freqs)
        maxFI = np.argmax(freqs)

        return maxF, maxFI

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

    def __str__(self):

        return "Instance of ProDSP class"

    def __repr__(self):

        return 'Instance of {} class. Sequence Dimensions {}x{}. Spectrum {}. Window {}. Filter: {}'.format(self.__class__.__name__, self.num_seqs, self.signal_len, self.spectrum, self.window, self.filter)
