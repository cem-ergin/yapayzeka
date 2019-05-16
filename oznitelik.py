import os

import librosa
import numpy as np


eps = 0.00000001

#zero crossing rate hesabı
def stZCR(frame):
    count = len(frame)
    countZ = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
    #print((numpy.float64(countZ) / numpy.float64(count-1.0)))
    return (np.float64(countZ) / np.float64(count-1.0))


def energy(frame):
    """Calculate the energy of a frame.
    Roghly implemented after Rabiner, L. R. and Sambur M. R.
    (1975). An Algorithm for Determining the Endpoints of Isolated
    Utterances. The Bell Systems Technical Journal, 54:297--315.
    """
    return np.sum(np.abs(frame))

#spectral_centroid hesabı
def spectral_centroid(x, samplerate):
    magnitudes = np.abs(np.fft.rfft(x))
    length = len(x)
    freqs = np.abs(np.fft.fftfreq(length, 1.0/samplerate)[:length//2+1])
    #print(np.sum(magnitudes*freqs) / np.sum(magnitudes))
    return np.sum(magnitudes*freqs) / np.sum(magnitudes)

#energy hesabı bu pek güzel sonuç vermiyor
def stEnergy(frame):
    #print(numpy.sum(frame ** 2) / numpy.float64(len(frame)))
    return np.sum(frame ** 2) / np.float64(len(frame))

#spectra_centroid ve spread
def stSpectralCentroidAndSpread(X, fs):
    """Computes spectral centroid of frame (given abs(FFT))"""
    ind = (np.arange(1, len(X) + 1)) * (fs/(2.0 * len(X)))

    Xt = X.copy()
    Xt = Xt / Xt.max()
    NUM = np.sum(ind * Xt)
    DEN = np.sum(Xt) + eps

    # Centroid:
    C = (NUM / DEN)

    # Spread:
    S = np.sqrt(abs(np.sum(((ind - C) ** 2) * Xt) / DEN))

    # Normalize:
    C = C / (fs / 2.0)
    S = S / (fs / 2.0)

    #print("bu centroid: ",C)
    #print("bu spread: ",S)
    return (C, S)




path = '/Users/cemergin/Documents/tess dataset/sad'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.wav' in file:
            files.append(os.path.join(r, file))


print(files)
file1 = open("tesscemuzgun.txt","a")
sayac=0
for f in files:
    #className = f[42:-5] #(emodb) path = '/Users/cemergin/Downloads/emo-DB/wav'
    #className = f[67:-16] #(ravdes) path = '/Users/cemergin/Downloads/Audio_Speech_Actors_01-24'


    #if(className != "02" and className != "08"): #ravdes
    #if (className != "L"):
        print(f)
        #print(className)
        data, sampling_rate = librosa.load(f)
        #print(f[34:])
        #print(f[67:-16])
        #mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        stft = np.abs(librosa.stft(data))
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)  # 40 values
        # zcr = np.mean(librosa.feature.zero_crossing_rate)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sampling_rate).T, axis=0)
        #mel = np.mean(librosa.feature.melspectrogram(data, sr=sampling_rate).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sampling_rate).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=sampling_rate).T,  # tonal centroid features
                          axis=0)
        print(sayac)
        for oz in mfccs:
            x = oz.astype(np.float)
            file1.write(str(oz))
            file1.write(",")
        for oz in chroma:
            file1.write(str(oz))
            file1.write(",")
        #for oz in mel:
        #    file1.write(str(oz))
        #    file1.write(",")
        for oz in contrast:
            file1.write(str(oz))
            file1.write(",")
        for oz in tonnetz:
            file1.write(str(oz))
            file1.write(",")
        #file1.write(str(energy(data)))
        #file1.write(",")
        #file1.write(str(stZCR(data)))
        #file1.write(",")
        #file1.write(str(spectral_centroid(data,sampling_rate)))
        #file1.write(",")
        #file1.write(str(stEnergy(data)))
        #file1.write(",")
        #c,s = stSpectralCentroidAndSpread(data,sampling_rate)
        #file1.write(str(c))
        #file1.write(",")
        #file1.write(str(s))
        #file1.write(",")

        #file1.write(f[34:])
         #ravdes
        '''
            if (className == "03"):
                file1.write("Mutlu")
                sayac += 1
            if (className == "04"):
                file1.write("Uzuntulu")
                sayac += 1
            
            if (className == "06"):
                file1.write("Korku")
                sayac += 1
            if (className == "07"):
                file1.write("İgrenmis")
                sayac += 1
            if (className == "05"):
                file1.write("Sinirli")
                sayac += 1
            if (className == "01"):
                file1.write("Normal")
                sayac += 1
        '''
        '''
        if (className == "F"):
           file1.write("Mutlu")
           sayac += 1
        if (className == "T"):
           file1.write("Uzuntulu")
           sayac += 1
        if (className == "A"):
           file1.write("Korku")
           sayac += 1
        if (className == "E"):
           file1.write("İgrenmis")
           sayac += 1
        if (className == "W"):
           file1.write("Sinirli")
           sayac += 1
        if (className == "N"):
           file1.write("Normal")
           sayac += 1
        '''

        '''
        if (sayac < 97):
            file1.write("dogal")
        elif (sayac >= 97 and sayac < 288):
            file1.write("sakin")
        elif (sayac >= 288 and sayac < 480):
            file1.write("mutlu")
        elif (sayac >= 480 and sayac < 672):
            file1.write("uzgun")
        elif (sayac >= 672 and sayac < 864):
            file1.write("sinirli")
        elif (sayac >= 864 and sayac < 1056):
            file1.write("korkulu")
        elif (sayac >= 1056 and sayac < 1248):
            file1.write("igrenmis")
        elif (sayac >= 1248 and sayac < 1440):
            file1.write("saskin")
        '''
        file1.write("Uzuntulu")
        file1.write("\n")
        #print(sayac)

file1.close()