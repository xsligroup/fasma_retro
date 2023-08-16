from fasma.core import functional
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import psutil
import math


@dataclass
class Spectrum:
    freq: np.ndarray
    spect: np.ndarray
    x: np.ndarray
    y: np.ndarray


@dataclass
class RTimeSpectrum(Spectrum):
    def gen_spect(self, damp: float = 0.0001, wlim: tuple = (0, 4/27), res: float = 400000, every_step: int = 100, meth: str = "pade"):
        meth = meth.lower()
        chosen_times = self.x[::every_step]
        chosen_dipoles = self.x[::every_step]
        step_size = chosen_times[1] - chosen_times[0]
        spects = []
        for x in range(3):
            s = chosen_dipoles[:, x]
            s -= s[0]
            s *= np.exp[-damp * chosen_times]
            if meth == "pade":
                transformer = functional.pade_tx
            elif meth == "gaussian":
                transformer = functional.fourier_tx
            else:
                raise ValueError('Unsupported distribution "{0}" specified'.format(meth))
            self.freq, f = transformer(s, step_size, wlim, res)
            spects.append(f)
        self.spect = sum([f.imag for f in spects]) / 3
        self.spect *= -self.freq


@dataclass
class ImportedSpectrum(Spectrum):
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None


@dataclass
class SimulatedSpectrum(Spectrum):
    freq: np.ndarray = field(init=False)
    spect: np.ndarray = field(init=False)

    def gen_spect(self, broad: float = 0.5, wlim=None, res: float = 100, xshift: float = 0, meth: str = 'lorentz'):
        meth = meth.lower()
        nani = False
        if meth == 'lorentz':
            if nani:
                meth = functional.lorentzian_2
            else:
                meth = functional.lorentzian
        elif meth == 'gaussian':
            if nani:
                meth = functional.gaussian_2
            else:
                meth = functional.gaussian
        else:
            raise ValueError('Unsupported distribution "{0}" specified'.format(meth))

        if wlim is None:
            print("Spectral range not specified... " +
                  "Automatically generating spectral range")
            percent = 0.930

            # Find max and min energies
            min = self.x.min()
            max = self.x.max()

            # Use quartile function of lorentz distribution regardless of distribution type
            lower_bound = broad * np.tan(((1 - percent) - 0.5) * np.pi) + min
            upper_bound = broad * np.tan((percent - 0.5) * np.pi) + max
            wlim = (lower_bound, upper_bound)
        n_points = int((wlim[1] - wlim[0]) * res)

        self.freq = np.linspace(wlim[0], wlim[1], n_points) + xshift
        #self.freq = self.freq[:, None]
        self.spect = np.zeros(n_points)

        if nani:
            stats = psutil.virtual_memory()  # returns a named tuple
            available = getattr(stats, 'available') / 2
            max_roots_per_cycle = math.floor(available / (8 * n_points))
            if max_roots_per_cycle > len(self.x):
                max_roots_per_cycle = len(self.x)
            print(len(self.x))
            chunks = math.ceil(len(self.x) / max_roots_per_cycle)
            print(chunks)
            split_x = np.array_split(self.x, chunks)
            split_y = np.array_split(self.y, chunks)

            for current_x, current_y in zip(split_x, split_y):
                self.spect += meth(broad, current_x, current_y, self.freq)
        else:
            #with concurrent.futures.ThreadPoolExecutor() as executor:
                #futures = [executor.submit(meth, broad, current_x, current_y, self.freq) for current_x, current_y in zip(self.x, self.y) if current_y != 0]
                #for future in concurrent.futures.as_completed(futures):
                    #self.spect += future.result()
            for current_x, current_y in zip(self.x, self.y):
                self.spect += meth(broad, current_x, current_y, self.freq)
