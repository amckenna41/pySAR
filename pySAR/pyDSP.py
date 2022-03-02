################################################################################
#################                  Protein DSP                 #################
################################################################################

from multiprocessing.sharedctypes import Value
import pandas as pd
import numpy as np
from difflib import get_close_matches
from scipy import signal
from scipy.signal import savgol_filter, medfilt, symiirorder1, lfilter, hilbert
from scipy.signal.windows import blackman, hanning, hamming, bartlett, blackmanharris, \
     kaiser, gaussian, barthann, bohman, chebwin, cosine, exponential, boxcar, \
        flattop, nuttall, parzen, tukey, triang
try:
    from scipy.fftpack import fft, ifft, fftfreq, rfft, rfftfreq
except:
    from numpy.fft import fft, ifft, fftfreq, rfft, rfftfreq
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import json
from json import JSONDecodeError

from .utils import *
from .globals_ import DATA_DIR, OUTPUT_DIR, OUTPUT_FOLDER
from .aaindex import AAIndex
class PyDSP():
    """
    Transform protein sequences into their spectral form via a Fast Fourier Transform (FFT).
    Fourier analysis is fundamentally a method for expressing a function as a sum of periodic components,
    and for recovering the function from those components. When both the function and its Fourier
    transform are replaced with discretized counterparts, it is called the Discrete Fourier transform (DFT).
    An implementation algorithm for the DFT is known as the FFT, which is used here. From the FFT
    transformations on the encoded protein sequences, various informational protein spectra
    can be generated, including the power, real, imaginary and absolute spectra.
    Prior to the FFT, a window function can be applied to the sequences which is a mathmatical
    function that applies a weighting to each discrete time series sample in a finite set.
    By default, the hamming function is applied; although the function can accept
    the blackman, blackmanharris, bartlett, gaussian and kaiser window functions. A
    filter function can also be applied. Additionally a convolution function can be 
    applied before the FFT.

    Additonal to the FFT, the RFFT (real-FFT) is calculated which removes some
    redundancy that is generated from the original FFT function which itself is
    Hermitian-symmetric meaning the negative frequency terms are just the complex
    conjugates of the corresponding positive frequency terms, thus the neg-freq
    terms are redundant.

    In the pipeline of pySAR this class and its functions are onyl used when the 'use_dsp'
    parameter is set to true in the config files meaning that the encoded protein sequences
    are passed through a Digital Signal Processing pipeline before being used as training
    data for the regression models. 

    Attributes
    ----------
    :dsp_config (str/json):
        path to configuration file containing DSP parameters OR JSON object of DSP parameters,
        depending on if the parameter is a valid filepath or not.
    :protein_seqs (np.ndarray):
        array of pre-encoded protein sequences. Class accepts only numerically encoded protein
        sequences, not in amino acid form.

    Methods
    -------
    pre_processing():
        complete pre-processing steps before completeing DSP functionality.
    encode_seqs():
        calculate FFT/RFFT of protein seqeuences.
    inverse_FFT():
        calculate inverse FFT of protein sequences.
    consensus_freq():
        calculate consensus frequency of FFT.
    max_freq():
        calculate max frequency of FFT
    """
    def __init__(self, dsp_config="", protein_seqs=None):

        self.protein_seqs = protein_seqs
        self.dsp_config = dsp_config
        self.parameters = {}

        config_filepath = ""
        #read protein seqs from dataset if not input as parameter,
        # parse DSP parameters from config file
        if not isinstance(dsp_config, str) or dsp_config is None:
            raise TypeError('JSON config file must be a filepath of type string, got type {}.'.format(type(dsp_config)))
        if os.path.isfile(self.dsp_config):
            config_filepath = self.dsp_config
        elif os.path.isfile(os.path.join('config', self.dsp_config)):
            config_filepath = os.path.join('config', self.dsp_config)
        else:
            raise OSError('JSON config file not found at path: {}.'.format(config_filepath))
        try:
            with open(config_filepath) as f:
                self.parameters = json.load(f)
        except:
            raise JSONDecodeError('Error parsing config JSON file: {}.'.format(config_filepath))

        #read protein seqs from dataset if not input as parameter
        if (self.protein_seqs is None):
            raise ValueError('Protein sequences input parameter cannot be empty or None.')

        #reshape protein sequences to 2 dimensions
        if (self.protein_seqs.ndim!=2):
            try:
                self.protein_seqs = self.protein_seqs.reshape((-1,1))
            except:
                raise ValueError('Error reshaping input sequences.')

        #set all DSP parameters
        self.dsp_parameters = self.parameters["pyDSP"]
        self.use_dsp = self.dsp_parameters[0]["use_dsp"]
        self.spectrum = self.dsp_parameters[0]["spectrum"]
        self.window_type = self.dsp_parameters[0]["window"]
        self.window = self.dsp_parameters[0]["window"]
        self.filter = self.dsp_parameters[0]["filter"]
        self.convolution = self.dsp_parameters[0]["convolution"]

        #pre-processing of encoded protein sequences
        self.pre_processing()

        #transform sequences into the various informational protein spectra
        self.encode_seqs()

    def pre_processing(self):
        """
        Complete various pre-processing steps for encoded protein sequences before
        doing any of the DSP-related functions or transformations. Zero-pad
        the sequences, remove any +/- infinity or NAN values, get the approximate
        protein spectra and window function parameter names.

        Parameters
        ----------
        :self (PyDSP object): 
            instance of PyDSP class.
            
        Returns
        -------
        None

        """
        #zero-pad encoded sequences so they are all the same length
        self.protein_seqs = zero_padding(self.protein_seqs)

        #get shape parameters of proteins seqs
        self.num_seqs = self.protein_seqs.shape[0]
        self.signal_len = self.protein_seqs.shape[1]

        #replace any positive or negative infinity or NAN values with 0
        self.protein_seqs[self.protein_seqs == -np.inf] = 0
        self.protein_seqs[self.protein_seqs == np.inf] = 0
        self.protein_seqs[self.protein_seqs == np.nan] = 0

        #replace any NAN's with 0's
        #self.protein_seqs.fillna(0, inplace=True)
        self.protein_seqs = np.nan_to_num(self.protein_seqs)

        #initialise zeros array to store all protein spectra
        self.fft_power = np.zeros((self.num_seqs, self.signal_len))
        self.fft_real = np.zeros((self.num_seqs, self.signal_len))
        self.fft_imag = np.zeros((self.num_seqs, self.signal_len))
        self.fft_abs = np.zeros((self.num_seqs, self.signal_len))

        #list of accepted spectra, window functions and filters
        all_spectra = ['power','absolute','real','imaginary']
        all_windows = ['hamming', 'blackman','blackmanharris','gaussian','bartlett',
                       'kaiser', 'barthann', 'bohman', 'chebwin', 'cosine', 'exponential'
                       'flattop', 'hann', 'boxcar', 'hanning', 'nuttall', 'parzen',
                        'triang', 'tukey']
        all_filters = ['savgol', 'medfilt', 'symiirorder1', 'lfilter', 'hilbert']

        #set required input parameters, raise error if spectrum is none
        if self.spectrum == None:
            raise ValueError('Invalid input Spectrum type ({}) not available in valid spectra: {}'.
                format(self.spectrum, all_spectra))
        else:
            #get closest correct spectra from user input, if no close match then raise error
            spectra_matches = (get_close_matches(self.spectrum, all_spectra, cutoff=0.4))

            if spectra_matches == []:
                raise ValueError('Invalid input Spectrum type ({}) not available in valid spectra: {}'.
                    format(self.spectrum, all_spectra))
            else:
                self.spectra = spectra_matches[0]   #closest match in array

        if self.window_type == None:
            self.window = 1       #window = 1 is the same as applying no window
        else:
            #get closest correct window function from user input
            window_matches = (get_close_matches(self.window, all_windows, cutoff=0.4))

            #check if sym=True or sym=False
            #get window function specified by window input parameter, if no match then window = 1
            if window_matches != []:
                if window_matches[0] == 'hamming':
                    self.window = hamming(self.signal_len, sym=True)
                    self.window_type = "hamming"
                elif window_matches[0] == "blackman":
                    self.window = blackman(self.signal_len, sym=True)
                    self.window = "blackman"
                elif window_matches[0] == "blackmanharris":
                    self.window = blackmanharris(self.signal_len, sym=True) #**
                    self.window_type = "blackmanharris"
                elif window_matches[0] == "bartlett":
                    self.window = bartlett(self.signal_len, sym=True)
                    self.window_type = "bartlett"
                elif window_matches[0] == "gaussian":
                    self.window = gaussian(self.signal_len, std=7, sym=True)
                    self.window_type = "gaussian"
                elif window_matches[0] == "kaiser":
                    self.window = kaiser(self.signal_len, beta=14, sym=True)
                    self.window_type = "kaiser"
                elif window_matches[0] == "hanning":
                    self.window = hanning(self.signal_len, sym=True)
                    self.window_type = "hanning"
                elif window_matches[0] == "barthann":
                    self.window = barthann(self.signal_len, sym=True)
                    self.window_type = "barthann"
                elif window_matches[0] == "bohman":
                    self.window = bohman(self.signal_len, sym=True)
                    self.window_type = "bohman"
                elif window_matches[0] == "chebwin":
                    self.window = chebwin(self.signal_len, sym=True)
                    self.window_type = "chebwin"
                elif window_matches[0] == "cosine":
                    self.window = cosine(self.signal_len, sym=True)
                    self.window_type = "cosine"
                elif window_matches[0] == "exponential":
                    self.window = exponential(self.signal_len, sym=True)
                    self.window_type = "exponential"
                elif window_matches[0] == "flattop":
                    self.window = flattop(self.signal_len, sym=True)
                    self.window_type = "flattop"
                elif window_matches[0] == "boxcar":
                    self.window = boxcar(self.signal_len, sym=True)
                    self.window_type = "boxcar"
                elif window_matches[0] == "nuttall":
                    self.window = nuttall(self.signal_len, sym=True)
                    self.window_type = "nuttall"
                elif window_matches[0] == "parzen":
                    self.window = parzen(self.signal_len, sym=True)
                    self.window_type = "parzen"
                elif window_matches[0] == "triang":
                    self.window = triang(self.signal_len, sym=True)
                    self.window_type = "triang"
                elif window_matches[0] == "tukey":
                    self.window = tukey(self.signal_len, sym=True)
                    self.window_type = "tukey"

            else:
                self.window = 1     #window = 1 is the same as applying no window

        #calculate convolution from protein sequences
        if self.convolution is not None:
            if self.window is not None:
                self.convoled_seqs = signal.convolve(self.protein_seqs, self.window, mode='same') / sum(self.window)

        if self.filter != None:
            #get closest correct filter from user input
            filter_matches = (get_close_matches(self.filter, all_filters, cutoff=0.4))

            #set filter attribute according to approximate user input
            if filter_matches != []:
                if filter_matches[0] == 'savgol':
                    self.filter = savgol_filter(self.signal_len, self.signal_len) 
                elif filter_matches[0] == 'medfilt':
                    self.filter = medfilt(self.signal_len)
                elif filter_matches[0] == 'symiirorder1':
                    self.filter = symiirorder1(self.signal_len, c0=1, z1=1)
                elif filter_matches[0] == 'lfilter':
                    self.filter = lfilter(self.signal_len)
                elif filter_matches[0] == 'hilbert':
                    self.filter = hilbert(self.signal_len)      
            else:
                self.filter = ""           #no filter

    def encode_seqs(self): 
        """
        Calculate the FFT and RFFT of the protein sequences already encoded using
        AAI indices, then use the output of the FFT to calculate the various
        informational protein spectra including the power, absolute, real and imaginary.
        The spectrum_encoding attribute will be set to the spectra inputted by
        user from the 'spectrum' config parameter.

        Parameters 
        ----------
        :self (PyDSP object): 
            instance of PyDSP class.

        Returns
        -------
        None
        """
        #create copy of protein sequences so the original instance var remains unchanged
        encoded_seq_copy = np.copy(self.protein_seqs)

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
        #  datatype to complex number as that is the output type of the transformation.
        encoded_dataset_rfft = np.zeros((self.protein_seqs.shape), dtype=complex)
        encoded_dataset_fft = np.zeros((self.protein_seqs.shape), dtype=complex)

        #initialise zero arrays used to store output frequencies from fft & rfft transformations
        rttf_freq_size = int((rfft_output_size)/2 + 1)
        encoded_freqs_rfft = np.zeros((self.protein_seqs.shape))
        encoded_freqs_fft = np.zeros(self.protein_seqs.shape)

        #iterate through each sequence, applying the FFT and RFFT to each
        for seq in range(0, self.num_seqs):

          encoded_rfft = np.zeros((self.protein_seqs.shape[1]), dtype=complex)
          encoded_fft = np.zeros((self.protein_seqs.shape[1]), dtype=complex)

          #apply filter *
        #   if (self.filter!=""):
        #      if (self.filter == 'savgol'):
        #         encoded_seq_copy[seq] = savgol_filter(encoded_seq_copy[seq], 17, polyorder=2, deriv=2)

          if not self.convolution:
            #apply window function to Fourier array
            encoded_rfft = rfft(encoded_seq_copy[seq] * self.window)
            encoded_fft = fft(encoded_seq_copy[seq] * self.window)
          else:
            #apply convolution and window function to encoded sequences
            encoded_rfft = rfft((signal.convolve(encoded_seq_copy[seq], self.window, mode='same') / sum(self.window)))
            encoded_fft = fft((signal.convolve(encoded_seq_copy[seq], self.window, mode='same') / sum(self.window)))

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

        #set the spectrum_encoding attribute to the spectra specified by 'spectra'
        #   class input parameter.
        if self.spectra == 'power':
            self.spectrum_encoding = self.fft_power
        elif self.spectra == 'real':
            self.spectrum_encoding = self.fft_real
        elif self.spectra == 'imaginary':
            self.spectrum_encoding = self.fft_imag
        elif self.spectra == 'absolute':
            self.spectrum_encoding = self.fft_abs

    def inverse_FFT(self, a, n):
        """
        Get the inverse Fourier Transform of FFT.

        Parameters
        ----------
        :a : np.ndarray
            input array of 1D Fourier Transform.
        :n : int
            length of the output.

        Returns
        -------
        :inv_FFT : np.ndarray
            array of inverse Fourier Transform.
        """
        self.inv_FFT = np.fft.ifft(a,n)
        return self.inv_FFT

    def consensus_freq(self, freqs):
        """
        Get the Consensus Frequency from Fourier Transform of encoded protein sequences.

        Parameters
        ----------
        :freqs : np.ndarray
            frequencies of Fourier Transform.

        Returns
        -------
        :CF : float
            consensus frequency found in array of frequencies.
        :CFi : int
            index of consensus frequency.
        """
        # CF = PP/N ( peak position/length of largest protein in dataset)
        CF, CFi = (self.max_freq(freqs))/self.num_seqs

        return CF, CFi

    def max_freq(self, freqs):
        """
        Get the maximum frequency from Fourier Transform of encoded protein sequences.

        Parameters
        ----------
        :freqs : np.ndarray
            frequencies of Fourier Transform.

        Returns
        -------
        :max_F : float
            maximum frequency found in array of frequencies.
        :max_FI : int
            index of maximum frequency.
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
    def window_type(self):
        return self._window_type

    @window_type.setter
    def window_type(self, val):
        self._window_type = val

    @property
    def filter_(self):
        return self._filter

    @filter_.setter
    def filter_(self, val):
        self._filter = val

    @property
    def convolution(self):
        return self._convolution

    @convolution.setter
    def convolution(self, val):
        self._convolution = val

################################################################################

    def __str__(self):
        return "Instance of PyDSP class, using parameters: {}".format(self.__dict__.keys())

    def __repr__(self):
        return ('<PyDSP: {}>'.format(self))
