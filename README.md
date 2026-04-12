# waveguides

A Python package for computing waveguide mode properties, including cutoff frequencies, wave impedances, phase shifts, and transverse electric field distributions for rectangular and circular waveguides.

## Features

- **Rectangular waveguides** (`RecWG`): TE and TM modes, arbitrary dimensions and filling material
- **Circular waveguides** (`CirWG`): TE and TM modes with full polarisation tracking (sine/cosine pairs)
- Per-mode cutoff frequency, wave impedance, and phase shift (with optional lossless approximation)
- Transverse electric field distribution on arbitrary grid points, returned as a plain `EField2D` dataclass
- Built-in serialization (`dump` / `load`) for saving and restoring waveguide configurations
- Vectorised multi-frequency computation with optional multiprocessing (`heavy_computation` module)

## Installation

Clone the repository and install from source:

```bash
git clone https://github.com/PigDuo/waveguides.git
cd waveguides
pip install .
```

To also install development tools:

```bash
pip install ".[dev]"
```

Requires Python ≥ 3.10. Core dependencies are `numpy` and `scipy` only.

## Quick Start

```python
import numpy as np
from waveguides import RecWG, CirWG

# Rectangular waveguide: WR-28 (a = 7.112 mm, b = 3.556 mm), length 10 mm, copper walls
rwg = RecWG(a=7.112e-3, b=3.556e-3, l=10e-3, N=20)

# List mode names
print(rwg.mode_name_list)
# ['TE1,0', 'TE0,1', 'TE2,0', ...]

# Cutoff frequency of the dominant mode (Hz)
print(rwg.mode_info_list[0].fc)

# Wave impedance at 30 GHz for all modes
Z = rwg.impedance_at(30e9)

# Phase shift through the waveguide at 30 GHz
ps = rwg.phaseshift_at(30e9)

# Lossless phase shift over a frequency sweep
fs = np.linspace(26e9, 40e9, 101)
ps_array = rwg.phaseshift_at_list(fs, lossless=True)   # shape (101, 20)
```

```python
# Circular waveguide: radius 4.2 mm, length 1.3 mm
cwg = CirWG(r=4.2e-3, l=1.3e-3, N=50)

# Transverse E-field distribution of mode 0 on a Cartesian grid
X, Y = np.meshgrid(np.linspace(-5e-3, 5e-3, 100),
                   np.linspace(-5e-3, 5e-3, 100))
field = cwg.get_mode_efield_distribution_at_gridpoints(0, X, Y)
# field.Ex, field.Ey, field.Ez are numpy arrays of shape (100, 100)
```

## Multi-frequency Computation with Multiprocessing

For large frequency sweeps or high mode counts, the `heavy_computation` module provides a parallelised interface that accepts a standard `multiprocessing.Pool`:

```python
from multiprocessing import Pool
import waveguides.heavy_computation as hc

fs = np.linspace(26e9, 40e9, 201)

# Single-process (default)
ps = hc.phaseshift_array(cwg, fs)

# Multi-process
with Pool(processes=6) as pool:
    ps = hc.phaseshift_array(cwg, fs, pool=pool, chunksize=4096)
    Z  = hc.impedance_array(cwg, fs, pool=pool, chunksize=4096)
# Both return arrays of shape (len(fs), N)
```

## Serialization

```python
import json

# Save
data = rwg.dump()
with open("rwg.json", "w") as f:
    json.dump(data, f)

# Load
from waveguides import WG
with open("rwg.json") as f:
    rwg2 = WG.load(json.load(f))
```

## API Reference

### `RecWG(a, b, l, N, er, sigma)`

| Parameter | Description | Default |
|-----------|-------------|---------|
| `a` | Broad-wall width (m) | `1` |
| `b` | Narrow-wall height (m) | `0.5` |
| `l` | Length (m) | `1` |
| `N` | Number of modes to compute | `1` |
| `er` | Relative permittivity of filling | `1` |
| `sigma` | Wall conductivity (S/m) | `5.8e7` (copper) |

### `CirWG(r, l, N, er, sigma)`

| Parameter | Description | Default |
|-----------|-------------|---------|
| `r` | Inner radius (m) | `1` |
| `l` | Length (m) | `1` |
| `N` | Number of modes to compute | `1` |
| `er` | Relative permittivity of filling | `1` |
| `sigma` | Wall conductivity (S/m) | `5.8e7` (copper) |

### Common methods (both classes)

| Method / Property | Description |
|-------------------|-------------|
| `mode_info_list` | List of `ModeInfo` objects, one per mode |
| `mode_name_list` | List of mode name strings, e.g. `'TE1,0'` |
| `mode_name_list_latex` | LaTeX-formatted mode name strings for matplotlib |
| `impedance_at(f)` | Wave impedance array at frequency `f` (Hz) |
| `impedance_at_list(fs)` | Wave impedance matrix over a frequency list |
| `phaseshift_at(f, lossless)` | Phase-shift array at frequency `f` (Hz) |
| `phaseshift_at_list(fs, lossless)` | Phase-shift matrix over a frequency list |
| `wavelength_at(f)` | Guide wavelength array at frequency `f` (Hz) |
| `get_mode_efield_distribution_at_gridpoints(mode_idx, X, Y)` | Transverse E-field as `EField2D` |
| `dump()` / `WG.load(data)` | Serialize / deserialize |

### `EField2D`

A dataclass returned by `get_mode_efield_distribution_at_gridpoints`:

```python
@dataclass
class EField2D:
    X:  np.ndarray   # grid x-coordinates (m)
    Y:  np.ndarray   # grid y-coordinates (m)
    Ex: np.ndarray   # x-component of E-field (complex)
    Ey: np.ndarray   # y-component of E-field (complex)
    Ez: np.ndarray   # z-component of E-field (complex, always zero for TE/TM)
```

## License

MIT License. See [LICENSE](LICENSE) for details.
