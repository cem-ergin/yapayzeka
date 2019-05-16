import librosa as librosa
import numpy




def zcr(frame):
    """Calculate the zero crossing rate of a frame.
    Implemented after Chen, C.H. (1988). Signal Processing Handbook.
    p. 531, New York: Dekker.
    """
    T = len(frame) - 1
    return 1 / T * numpy.sum(numpy.signbit(numpy.multiply(frame[1:T], frame[0:T - 1])))

def energy(frame):
    """Calculate the energy of a frame.
    Roghly implemented after Rabiner, L. R. and Sambur M. R.
    (1975). An Algorithm for Determining the Endpoints of Isolated
    Utterances. The Bell Systems Technical Journal, 54:297--315.
    """
    return numpy.sum(numpy.abs(frame))


y, sr = librosa.load('/Users/cemergin/Desktop/siralises/0001.wav')
print(energy(y))