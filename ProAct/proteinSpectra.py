
import pandas as pd
import numpy as np
from utils import *
from globals import *
from aaindex import *
from scipy.signal import savgol_filter, blackman, hamming, bartlett, blackmanharris, kaiser, gaussian
from scipy.fftpack import fft,ifft

#https://scipy-lectures.org/intro/scipy/auto_examples/plot_fftpack.html
class ProteinDSP():

    def __init__(self, encoded_sequences, window_type="hamming", filter_type="",time_step=0.5):

        self.encoded_sequences = encoded_sequences
        self.window_type = window_type
        self.filter_type = filter_type
        self.time_step = time_step

    def encode_aaindex(self):

        encoded_seq_copy = np.copy(self.encoded_sequences)
        signal_len = self.encoded_sequences.shape[1]
        encoded_dataset_fft = np.zeros((self.encoded_sequences.shape),dtype=complex)
        encoded_freqs = np.zeros(self.encoded_sequences.shape)

        print(encoded_dataset_fft.shape)
        print('signallen - ', signal_len)

        for seq in range(0,encoded_seq_copy.shape[0]):
          encoded_fft = np.zeros((self.encoded_sequences.shape[1]),dtype=complex)
          if self.window_type != "":
            if self.window_type == 'blackman':
                w = blackman(signal_len)
            elif self.window_type == 'hamming':
                w = hamming(signal_len, sym=True)
            elif self.window_type == 'blackmanharris':
                w = blackmanharris(signal_len)
            elif self.window_type == 'gaussian':
                w = gaussian(signal_len, std=7)
            elif self.window_type == 'bartlett':
                w = bartlett(signal_len)
            elif self.window_type == 'kaiser':
                w = kaiser(signal_len)

            encoded_fft = fft(encoded_seq_copy[seq] * w)

          else:
            encoded_fft = fft(encoded_seq_copy[seq])
            # encoded_rfft = rfft(encoded_seq_copy[seq])
            # encoded_rfft_freqs = rfftfreq(encoded_fft.shape[0])

              # encoded_fft = fft(currentSeq,n) where n = len(sequence)


# Why use rfft instead of FFT:::
# When the DFT is computed for purely real input, the output is Hermitian-symmetric, i.e. the negative frequency terms are just the complex conjugates of the corresponding positive-frequency terms, and the negative-frequency terms are therefore redundant. This function does not compute the negative frequency terms, and the length of the transformed axis of the output is therefore n//2 + 1.
#
# When A = rfft(a) and fs is the sampling frequency, A[0] contains the zero-frequency term 0*fs, which is real due to Hermitian symmetry.
#
# If n is even, A[-1] contains the term representing both positive and negative Nyquist frequency (+fs/2 and -fs/2), and must also be purely real. If n is odd, there is no term at fs/2; A[-1] contains the largest positive frequency (fs/2*(n-1)/n), and is complex in the general case.
#
# If the input a contains an imaginary part, it is silently discarded.
#
#remove sample rate from fftfreq and just past in fft.size

          encoded_dataset_fft[seq] = encoded_fft
          # freqs = np.fft.fftfreq(encoded_fft.size, self.time_step)
          freqs = np.fft.fftfreq(encoded_fft.size)

          encoded_freqs[seq] = freqs

        self.fft = encoded_dataset_fft
        self.freqs = encoded_freqs
        self.power =  np.abs(self.fft[0:len(self.fft)])

        #get positive frequencies and peak freq
        pos_mask = np.where(self.freqs > 0)
        self.pos_freqs = self.freqs[pos_mask]
        self.peak_freq = self.pos_freqs[self.power[pos_mask].argmax()]

        # np.apply_along_axis(self.fft>0, 1, array)

        self.real = self.fft.real
        self.imag = self.fft.imag

        if (self.filter_type!=""):
          if (self.filter_type == 'savgol'):
            power = savgol_filter(power, 17, polyorder=2, deriv=2)
            R = savgol_filter(R, 17, polyorder=2, deriv=2)
            I = savgol_filter(I, 17, polyorder=2, deriv=2)

    def plot_freq(self):

        pass
        #https://scipy-lectures.org/intro/scipy/auto_examples/plot_fftpack.html
        # ps = self.get_power()
        # ps_len = self.encoded_seq
        #
        # time_step = 1 / 1
        # freqs = np.fft.fftfreq(data_scaled.size, time_step)
        # idx = np.argsort(freqs)
        #
        # plt.plot(freqs[idx], ps[idx])
        # plt.show()

    def get_freqs(self):

        pass

    def get_absolute(self):

        #return absolute spectrum

        pass
    def get_power(self):

        if hasattr(self, 'power'):
            return self.power
        else:
            raise ValueError('Power spectral values not set')

    def get_real(self):

        if hasattr(self, 'real'):
            return self.real
        else:
            raise ValueError('Real spectral values not set')

    def get_imag(self):

        if hasattr(self, 'imag'):
            return self.imag
        else:
            raise ValueError('Imaginary spectral values not set')

    def consensus_freq(self):

        # CF = PP/N ( peak position/length of largest protein in dataset)
        pass

    def max_freq(self, freqs):

        maxF = max(freqs)
        maxFI = np.argmax(freqs)

        return maxF, maxFI

class SavgolFilter():

    def __init__(self, X, window_length, polyorder, deriv, delta):
        self.X = X
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.delta = delta




#
# class Window():
#
#     def __init__(self, window_type, signal_len):
#
#         self.window_type = window_type
#         self.signal_len = signal
#
#     def hamming(self, M, sym):
#
#         return signal.hamming()
#
#         pass
#
#         #if window = hamming then do
#         pass



# sp = np.fft.fft(np.sin(t))
# freq = np.fft.fftfreq(t.shape[-1])
# plt.plot(freq, sp.real, freq, sp.imag)
# [<matplotlib.lines.Line2D object at 0x...>, <matplotlib.lines.Line2D object at 0x...>]
# plt.show()
