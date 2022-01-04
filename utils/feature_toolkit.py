import numpy as np
from scipy import signal
from scipy.stats import moment, entropy, skew, kurtosis
from scipy.fftpack import rfft, rfftfreq

def CPM(x):
    """
    count per minute features stats and value 
    """
    days, _, _ = x.shape
    x = np.clip(x - np.ones_like(x), a_min=0, a_max=None)
    cpm = (np.sum(x, axis=-1)).reshape(-1)  # (ndays x nmins)

    cpm_mu = np.mean(cpm)
    cpm_std = np.std(cpm)
    cpm_25 = np.quantile(cpm, .25)
    cpm_50 = np.median(cpm)
    cpm_75 = np.quantile(cpm, .75)
    cpm_range = np.max(cpm) - np.min(cpm)
    cpm_skew = skew(cpm)
    cpm_kurt = kurtosis(cpm)

    cpm_beta_a = ((1 - cpm_mu) / cpm_std**2 - 1 / cpm_mu) * cpm_mu**2
    cpm_beta_b = cpm_beta_a * (1 / cpm_mu - 1)

    cpm_entropy = Entropy(cpm)

    cpm = cpm.reshape(days, -1)
    cpm = np.mean(cpm, axis=0)

    return (cpm, (cpm_mu, cpm_std, cpm_25, cpm_50, cpm_75, cpm_range, cpm_skew, cpm_kurt, cpm_beta_a, cpm_beta_b, cpm_entropy))


def VMC(x):
    """
    vector magnitude count stats and value 
    """
    days, _, _ = x.shape
    x = np.clip(x - np.ones_like(x), a_min=0, a_max=None)
    vmc = np.mean(np.abs(x - np.mean(x, axis=-1, keepdims=True)),
                  axis=-1)  # (days, nepochs)

    vmc = vmc.reshape(-1)

    vmc_mu = np.mean(vmc)
    vmc_std = np.std(vmc)
    vmc_25 = np.quantile(vmc, .25)
    vmc_50 = np.median(vmc)
    vmc_75 = np.quantile(vmc, .75)
    vmc_range = np.max(vmc) - np.min(vmc)
    vmc_skew = skew(vmc)
    vmc_kurt = kurtosis(vmc)

    vmc_beta_a = ((1 - vmc_mu) / vmc_std**2 - 1 / vmc_mu) * vmc_mu**2
    vmc_beta_b = vmc_beta_a * (1 / vmc_mu - 1)

    vmc_entropy = Entropy(vmc)

    vmc = vmc.reshape(days, -1)
    vmc = np.mean(vmc, axis=0)

    return (vmc, (vmc_mu, vmc_std, vmc_25, vmc_50, vmc_75, vmc_range, vmc_skew, vmc_kurt, vmc_beta_a, vmc_beta_b, vmc_entropy))


def Entropy(x):
    den, _ = np.histogram(x, bins=200, density=True)
    den /= den.sum()
    return entropy(den)

def PAEE(x, sample_frequency):
    hist, bin_edges = np.histogram(x, bins=1000, density=False)
    paee = np.array([(bin_edges[i] + bin_edges[i + 1]) /
                     2 for i in range(len(bin_edges) - 1)])
    paee *= hist / sample_frequency
    mvpa_cutoff = np.quantile(np.unique(paee), .75, axis=0)
    mvpa_paee = np.array([paee[i] if paee[i] >
                          mvpa_cutoff else 0 for i in range(len(paee))])

    return paee.sum(), mvpa_paee.sum() / paee.sum()

def FFTstats(x, sample_frequency):
    """
    FFT stats and value 
    """
    xfft = np.abs(rfft(x))
    xfreq = rfftfreq(x.shape[0], 1 / sample_frequency)
    fentropy = entropy(xfft)
    f, psd = signal.periodogram(x, sample_frequency)
    psd_mu = np.mean(psd)
    psd_std = np.sqrt(np.var(psd, axis=0))
    rms_amplitude = np.sqrt(psd.max())
    mean_freq = (f * psd).sum() / psd.sum()
    median_freq = psd.sum() / 2
    kurt = moment(x, moment=4, axis=0) / psd_std**4
    skew = moment(x, moment=3, axis=0) / psd_std**3

    ind = xfft.argsort()[-15:]
    top_15_freq = xfreq[ind]
    top_15_fft = xfft[ind]

    return (top_15_fft, top_15_freq, fentropy, psd_mu, psd_std, rms_amplitude, kurt, skew, mean_freq, median_freq)
