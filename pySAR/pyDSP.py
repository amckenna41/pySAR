################################################################################
#################                  Protein DSP                 #################
################################################################################

from multiprocessing.sharedctypes import Value
import pandas as pd
import numpy as np
from difflib import get_close_matches
import inspect
from scipy import signal 
from scipy.signal import savgol_filter, medfilt, lfilter, hilbert
from scipy.signal.windows import blackman, hann, hamming, bartlett, blackmanharris, \
     kaiser, gaussian, barthann, bohman, chebwin, cosine, exponential, boxcar, \
        flattop, nuttall, parzen, tukey, triang
try:
    from scipy.fftpack import fft, ifft, fftfreq
except:
    from numpy.fft import fft, ifft, fftfreq
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import json
from json import JSONDecodeError
from .utils import *

class PyDSP():
    """
    Transform protein sequences into their spectral form via a Discrete Fourier Transform (DFT) using
    the Fast Fourier Transform (FFT) algorithm. Fourier analysis is fundamentally a method for 
    expressing a function as a sum of periodic components, and for recovering the function from those 
    components. When both the function and its Fourier transform are replaced with discretized 
    counterparts, it is called the Discrete Fourier transform (DFT). An implementation algorithm 
    for the DFT is known as the FFT, which is used here. From the FFT transformations on the 
    encoded protein sequences (encoded via amino acid property values of the AAI), various 
    informational protein spectra can be generated, including the power, real, imaginary and 
    absolute spectra. Prior to the FFT, a window function can be applied to the sequences 
    which is a mathmatical function that applies a weighting to each discrete time series sample 
    in a finite set. By default, the hamming window function is applied; although the function 
    can also accept the blackman, blackmanharris, bartlett, gaussia, bartlett, barthann, bohman, 
    chebwin, cosine, exponential, flattop, hann, boxcar, nuttall, parzen, triang and tukey windows.
    A filter function can also be applied, the class accepts the savgol, medfilt, lfilter and
    hilbert filters.

    In the pipeline of pySAR this class and its functions are onyl used when the 'use_dsp'
    parameter is set to true in the config files, meaning that the encoded protein sequences
    are passed through a Digital Signal Processing (DSP) pipeline before being used as 
    training data for the regression models. The protein sequences being numerically encoded
    is a pre-reqisite to use the functions in this class, meaning sequences cannot be directly
    input.

    Parameters
    ----------
    :config_file (str/json)
        path to configuration file containing DSP parameters OR JSON object of DSP parameters,
        depending on if the parameter is a valid filepath or not.
    :protein_seqs (np.ndarray)
        array of pre-encoded protein sequences. Class accepts only numerically encoded protein
        sequences, not in amino acid form.

    Methods
    -------
    pre_processing():
        complete pre-processing steps before completeing DSP functionality.
    encode_seqs():
        calculate FFT and various informational spectra of protein seqeuences.
    inverse_fft():
        calculate inverse FFT of protein sequences.
    consensus_freq():
        calculate consensus frequency of FFT.
    max_freq():
        calculate max frequency of FFT
    """
    def __init__(self, config_file="", protein_seqs=None):

        self.protein_seqs = protein_seqs
        self.config_file = config_file
        self.parameters = {}

        config_filepath = ""

        #read protein seqs from dataset if protein_seqs is None,
        if not (isinstance(config_file, str) or (isinstance(config_file, dict)) or (config_file is None)):
            raise TypeError('JSON config must be a filepath of type string or a dict of parameters, got type {}.'.
                format(type(config_file)))
        if (isinstance(config_file, str) and os.path.isfile(self.config_file)):
            config_filepath = self.config_file
        elif (isinstance(config_file, str) and os.path.isfile(os.path.join('config', self.config_file))):
            config_filepath = os.path.join('config', self.config_file)
        elif (isinstance(config_file, dict)):
            self.parameters = config_file
        else:
            raise OSError('JSON config file not found at path: {}.'.format(config_filepath))
        if (isinstance(config_file, str)):
            try:
                #open config file and parse parameters 
                with open(config_filepath) as f:
                    self.parameters = json.load(f)
            except:
                raise JSONDecodeError('Error parsing config JSON file: {}.'.format(config_filepath))

        #create instance of Map class so parameters in config can be accessed via dot notation
        self.parameters = Map(self.parameters)

        #raise error if protein sequences parameter is not set
        if (self.protein_seqs is None):
            raise ValueError('Protein sequences input parameter cannot be empty or None.')

        #direct protein sequences cannot be input to class, they must be encoded first, raise error if so
        for seq in protein_seqs:
            if (isinstance(seq, str)):
                raise ValueError("Protein sequences cannot be directly passed into the pyDSP class, you "
                                "must first encode the protein sequences using a specific aaindex code, "
                                "and then pass the resultant encoded sequence to the protein_seqs parameter.")
        
        #reshape protein sequences to 2 dimensions
        # if (self.protein_seqs.ndim != 2):
        #     try:
        #         self.protein_seqs = self.protein_seqs.reshape((-1, 1))
        #     except:
        #         raise ValueError('Error reshaping input sequences: {}'.format(protein_seqs))

        #set all DSP parameters
        self.dsp_parameters = self.parameters.pyDSP
        self.use_dsp = self.dsp_parameters["use_dsp"]
        self.spectrum = self.dsp_parameters["spectrum"]
        self.window_parameters = self.dsp_parameters["window"]
        self.window_type = self.window_parameters["type"]
        self.window = None
        self.filter_parameters = self.dsp_parameters["filter"]
        self.filter_type = self.filter_parameters["type"]
        self.filter = None
        
        #pre-processing of encoded protein sequences
        self.pre_processing()
        
        #transform sequences into the various informational protein spectra
        self.encode_seqs()

    def pre_processing(self):
        """
        Complete various pre-processing steps for encoded protein sequences before
        doing any of the DSP-related functions or transformations. Zero-pad the 
        sequences, remove any +/- infinity or NAN values, get the approximate
        protein spectra and window function parameter names.

        Parameters
        ----------
        None
            
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
        self.protein_seqs = np.nan_to_num(self.protein_seqs)

        #initialise zeros array to store all protein spectra
        self.fft_power = np.zeros((self.num_seqs, self.signal_len))
        self.fft_real = np.zeros((self.num_seqs, self.signal_len))
        self.fft_imag = np.zeros((self.num_seqs, self.signal_len))
        self.fft_abs = np.zeros((self.num_seqs, self.signal_len))

        #list of accepted spectra, window functions and filters
        all_spectra = ['power', 'absolute', 'real', 'imaginary']
        all_windows = ['hamming', 'blackman', 'blackmanharris', 'gaussian', 'bartlett',
                       'kaiser', 'barthann', 'bohman', 'chebwin', 'cosine', 'exponential',
                       'flattop', 'hann', 'boxcar', 'nuttall', 'parzen', 'triang', 'tukey']
        all_filters = ['savgol', 'medfilt', 'lfilter', 'hilbert']

        #set required input parameters, raise error if spectrum is none
        if (self.spectrum == None):
            raise ValueError('Invalid input Spectrum type ({}) not available: {}.'.
                format(self.spectrum, all_spectra))
        else:
            #get closest correct spectra from user input, if no close match then raise error
            spectra_matches = (get_close_matches(self.spectrum, all_spectra, cutoff=0.4))
            if (spectra_matches == []):
                raise ValueError('Invalid input Spectrum type ({}) not available: {}.'.
                    format(self.spectrum, all_spectra))
            else:
                self.spectrum = spectra_matches[0]   #closest match in array

        if (self.window_type == None):
            self.window = 1       #window = 1 is the same as applying no window
        else:
            #get closest correct window function from user input
            window_matches = (get_close_matches(self.window_type, all_windows, cutoff=0.4))

            #remove any null or None values from window parameters in config
            self.window_parameters = {k: v for k, v in self.window_parameters.items() if v}
            window_parameters = {}

            #get window function specified by window input parameter, if no match then window = 1,
            #pass in specific window parameters from those in the config, else use default parameters
            if (window_matches != []):
                if (window_matches[0] == 'hamming'):
                    for k, v in self.window_parameters.items():
                        if (k in inspect.getfullargspec(hamming).args): window_parameters[k] = self.window_parameters[k]
                    self.window = hamming(self.signal_len, **window_parameters)
                    self.window_type = "hamming"
                elif (window_matches[0] == "blackman"):
                    for k, v in self.window_parameters.items():
                        if (k in inspect.getfullargspec(blackman).args): window_parameters[k] = self.window_parameters[k]
                    self.window = blackman(self.signal_len, **window_parameters)
                    self.window_type = "blackman"
                elif (window_matches[0] == "blackmanharris"):
                    for k, v in self.window_parameters.items():
                        if (k in inspect.getfullargspec(blackmanharris).args): window_parameters[k] = self.window_parameters[k]
                    self.window = blackmanharris(self.signal_len, **window_parameters)
                    self.window_type = "blackmanharris"
                elif (window_matches[0] == "bartlett"):
                    for k, v in self.window_parameters.items():
                        if (k in inspect.getfullargspec(bartlett).args): window_parameters[k] = self.window_parameters[k]
                    self.window = bartlett(self.signal_len, **window_parameters)
                    self.window_type = "bartlett"
                elif (window_matches[0] == "gaussian"):
                    for k, v in self.window_parameters.items():
                        if (k in inspect.getfullargspec(gaussian).args): window_parameters[k] = self.window_parameters[k]
                    self.window = gaussian(self.signal_len, std=7, **window_parameters)
                    self.window_type = "gaussian"
                elif (window_matches[0] == "kaiser"):
                    for k, v in self.window_parameters.items():
                        if (k in inspect.getfullargspec(kaiser).args): window_parameters[k] = self.window_parameters[k]                    
                    window_parameters = {k: v for k, v in window_parameters.items() if k != "alpha"} #remove alpha parameter
                    self.window = kaiser(self.signal_len, **window_parameters)
                    self.window_type = "kaiser"
                elif (window_matches[0] == "hann"):
                    for k, v in self.window_parameters.items():
                        if (k in inspect.getfullargspec(hann).args): window_parameters[k] = self.window_parameters[k]
                    self.window = hann(self.signal_len, **window_parameters)
                    self.window_type = "hann"
                elif (window_matches[0] == "barthann"):
                    for k, v in self.window_parameters.items():
                        if (k in inspect.getfullargspec(barthann).args): window_parameters[k] = self.window_parameters[k]
                    self.window = barthann(self.signal_len, **window_parameters)
                    self.window_type = "barthann"
                elif (window_matches[0] == "bohman"):
                    for k, v in self.window_parameters.items():
                        if (k in inspect.getfullargspec(bohman).args): window_parameters[k] = self.window_parameters[k]
                    self.window = bohman(self.signal_len, **window_parameters)
                    self.window_type = "bohman"
                elif (window_matches[0] == "chebwin"):
                    for k, v in self.window_parameters.items():
                        if (k in inspect.getfullargspec(chebwin).args): window_parameters[k] = self.window_parameters[k]
                    self.window = chebwin(self.signal_len, at=100, **window_parameters)
                    self.window_type = "chebwin"
                elif (window_matches[0] == "cosine"):
                    for k, v in self.window_parameters.items():
                        if (k in inspect.getfullargspec(cosine).args): window_parameters[k] = self.window_parameters[k]
                    self.window = cosine(self.signal_len, **window_parameters)
                    self.window_type = "cosine"
                elif (window_matches[0] == "exponential"):
                    for k, v in self.window_parameters.items():
                        if (k in inspect.getfullargspec(exponential).args): window_parameters[k] = self.window_parameters[k]
                    self.window = exponential(self.signal_len, **window_parameters)
                    self.window_type = "exponential"
                elif (window_matches[0] == "flattop"):
                    for k, v in self.window_parameters.items():
                        if (k in inspect.getfullargspec(flattop).args): window_parameters[k] = self.window_parameters[k]
                    self.window = flattop(self.signal_len, **window_parameters)
                    self.window_type = "flattop"
                elif (window_matches[0] == "boxcar"):
                    for k, v in self.window_parameters.items():
                        if (k in inspect.getfullargspec(boxcar).args): window_parameters[k] = self.window_parameters[k]
                    self.window = boxcar(self.signal_len, **window_parameters)
                    self.window_type = "boxcar"
                elif (window_matches[0] == "nuttall"):
                    for k, v in self.window_parameters.items():
                        if (k in inspect.getfullargspec(nuttall).args): window_parameters[k] = self.window_parameters[k]
                    self.window = nuttall(self.signal_len, **window_parameters)
                    self.window_type = "nuttall"
                elif (window_matches[0] == "parzen"):
                    for k, v in self.window_parameters.items():
                        if (k in inspect.getfullargspec(parzen).args): window_parameters[k] = self.window_parameters[k]
                    self.window = parzen(self.signal_len, **window_parameters)
                    self.window_type = "parzen"
                elif (window_matches[0] == "triang"):
                    for k, v in self.window_parameters.items():
                        if (k in inspect.getfullargspec(triang).args): window_parameters[k] = self.window_parameters[k]
                    self.window = triang(self.signal_len, **window_parameters)
                    self.window_type = "triang"
                elif (window_matches[0] == "tukey"):
                    for k, v in self.window_parameters.items():
                        if (k in inspect.getfullargspec(tukey).args): window_parameters[k] = self.window_parameters[k]
                    self.window = tukey(self.signal_len, **window_parameters)
                    self.window_type = "tukey"
            else:
                self.window = 1     #window = 1 is the same as applying no window

        if ((self.filter_type != None) and (self.filter_type != "")):
            #get closest correct filter from user input
            filter_matches = get_close_matches(self.filter_type, all_filters, cutoff=0.4)
        
            #set filter attribute according to approximate user input
            if (filter_matches != []):
                if (filter_matches[0] == 'savgol'): #***
                    self.filter_type = "savgol"
                elif (filter_matches[0] == 'medfilt'):        
                    self.filter_type = "medfilt"
                elif (filter_matches[0] == 'lfilter'):
                    self.filter_type = "lfilter"       
                elif (filter_matches[0] == 'hilbert'):
                    self.filter_type = "hilbert"          

    def encode_seqs(self): 
        """
        Calculate the FFT of the protein sequences already encoded using
        the AAI indices, then use the output of the FFT to calculate the various
        informational protein spectra including the power, absolute, real and 
        imaginary. The spectrum_encoding attribute will be set to the spectrum 
        inputted by user from the 'spectrum' config parameter, if no valid 
        spectrum input as parameter then value error raised.

        Parameters 
        ----------
        None

        Returns
        -------
        None
        """
        #create copy of protein sequences so the original instance var remains unchanged
        encoded_seq_copy = np.copy(self.protein_seqs)

        #initialise zero arrays used to store output of both fft, set
        #datatype to complex number as that is the output type of the transformation
        encoded_dataset_fft = np.zeros((self.protein_seqs.shape), dtype=complex)

        #initialise zero arrays used to store output frequencies from fft transformations
        encoded_freqs_fft = np.zeros(self.protein_seqs.shape)

        #iterate through each sequence, applying the FFT to each
        for seq in range(0, self.num_seqs):
          
          #create temp zeros arrays to store current sequence's fft
          encoded_fft = np.zeros((self.signal_len), dtype=complex)

          #apply window function to Fourier array, multiplying by 1 if using no window function
          encoded_fft = fft(encoded_seq_copy[seq] * self.window)
          
          #apply filter to encoded sequences if filter_type not empty in config
          if ((self.filter_type != None) and (self.filter_type != "")):

            #remove any null or None values from filter parameters in config
            self.filter_parameters = {k: v for k, v in self.filter_parameters.items() if v}
            filter_parameters = {}

            #set filter attribute according to approximate user input
            if (self.filter_type == 'savgol'): 
                for k, v in self.filter_parameters.items():
                    if (k in inspect.getfullargspec(savgol_filter).args): filter_parameters[k] = self.filter_parameters[k]
                self.filter = savgol_filter(encoded_fft, **filter_parameters) 
            elif (self.filter_type == 'medfilt'):        
                for k, v in self.filter_parameters.items():
                    if (k in inspect.getfullargspec(medfilt).args): filter_parameters[k] = self.filter_parameters[k]
                self.filter = medfilt(encoded_fft, **filter_parameters) 
            elif (self.filter_type == 'lfilter'):
                for k, v in self.filter_parameters.items():
                    if (k in inspect.getfullargspec(lfilter).args): filter_parameters[k] = self.filter_parameters[k]
                self.filter = lfilter(encoded_fft, **filter_parameters) 
            elif (self.filter_type == 'hilbert'):
                for k, v in self.filter_parameters.items():
                    if (k in inspect.getfullargspec(hilbert).args): filter_parameters[k] = self.filter_parameters[k]
                self.filter = hilbert(encoded_fft, **filter_parameters) 
            else:
                self.filter = None #no filter

          #append transformation from current sequence seq to array of all transformed seqeunces
          encoded_dataset_fft[seq] = encoded_fft

          #calcualte FFT frequencies   
          freqs_fft = np.fft.fftfreq(encoded_fft.size)

          #append frequency from current sequence seq to array of all frequencies
          encoded_freqs_fft[seq] = freqs_fft

        #set FFT sequences and frequencies class attributes
        self.fft = encoded_dataset_fft
        self.fft_freqs = encoded_freqs_fft

        #get individual spectral values, calculated from the FFT transformations
        self.fft_abs = abs(self.fft/self.signal_len)
        self.fft_power = np.abs(self.fft[0:len(self.fft)])
        self.fft_real = self.fft.real
        self.fft_imag = self.fft.imag

        #set the spectrum_encoding attribute to the spectra specified by 'spectrum' class input parameter
        if (self.spectrum == 'power'):
            self.spectrum_encoding = self.fft_power
        elif (self.spectrum == 'real'):
            self.spectrum_encoding = self.fft_real
        elif (self.spectrum == 'imaginary'):
            self.spectrum_encoding = self.fft_imag
        elif (self.spectrum == 'absolute'):
            self.spectrum_encoding = self.fft_abs

    def inverse_fft(self, a, n):
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
        :inv_fft : np.ndarray
            array of inverse Fourier Transform.
        """
        self.inv_fft = np.fft.ifft(a,n)
        return self.inv_fft

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
        #raise error if more than one sequence passed into function
        if (freqs.ndim == 2 and freqs.shape[1] != 2):
            raise ValueError("Only one protein sequence should be passed into the function:"
                            " {}.".format(freqs))

        # CF = PP/N ( peak position/length of largest protein in dataset)
        CF, CFi = (self.max_freq(freqs))/self.num_seqs
        return CF, CFi

    def max_freq(self, freqs):
        """
        Get the maximum frequency from Fourier Transform of an encoded protein sequence.

        Parameters
        ----------
        :freqs : np.ndarray
            frequencies from Fourier Transform.

        Returns
        -------
        :max_F : float
            maximum frequency found in array of frequencies.
        :max_FI : int
            index of maximum frequency.
        """
        #raise error if more than one sequence passed into function
        if (freqs.ndim == 2 and freqs.shape[1] != 2):
            raise ValueError("Only one protein sequence should be passed into the function:"
                            "{}.".format(freqs))
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
    def filter_type(self):
        return self._filter_type

    @filter_type.setter
    def filter_type(self, val):
        self._filter_type = val

    def __str__(self):
        return "Instance of PyDSP class, using parameters: {}.".format(self.__dict__.keys())

    def __repr__(self):
        return ('<PyDSP: {}>'.format(self))