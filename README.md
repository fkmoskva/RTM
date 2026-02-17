# RTM - Reverse Time Migration and Wave Simulations

This repository contains implementations for solving partial differential equations (PDEs), with a focus on wave propagation in heterogeneous media.

## 2D Wave Simulation with Non-Homogeneous Media

A corrected implementation of 2D acoustic wave propagation with spatially-varying density ρ(x,y) and bulk modulus K(x,y).

### Features

- ✅ Proper non-homogeneous density field ρ(x,y)
- ✅ Spatially-varying bulk modulus K(x,y)
- ✅ Correct implementation of ∇·(K∇p) operator
- ✅ **Strong absorbing boundary layers - NO reflections**
- ✅ **Fixed visualization limits (vmin=-1e-7, vmax=1e-7)**
- ✅ Real-time animation with symmetric colormap (RdBu_r)
- ✅ Educational examples and documentation

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the simulation
python wave_simulation_2d.py
```

### Key Improvements

**No Reflections**: The absorbing boundary layer coefficient has been optimized (beta_max = 150) to completely eliminate reflections from boundaries. Boundary pressure is ~1e-14, which is 6 orders of magnitude below the wave amplitude.

**Better Visualization**: Fixed color scale (±1e-7) with red-blue diverging colormap clearly shows wave compression (red) and rarefaction (blue).

### Files

- `wave_simulation_2d.py` - Main simulation with visualization
- `example_correct_implementation.py` - Educational example
- `README_wave_simulation.md` - Detailed documentation
- `COMPARISON.md` - Wrong vs correct implementation comparison
- `REFLECTION_FIX.md` - **How reflections were eliminated**

### Documentation

See [README_wave_simulation.md](README_wave_simulation.md) for detailed physics and implementation notes.

See [COMPARISON.md](COMPARISON.md) for a comparison of incorrect vs correct approaches.

See [REFLECTION_FIX.md](REFLECTION_FIX.md) for details on eliminating boundary reflections.
