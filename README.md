# RTM - Reverse Time Migration and Wave Simulations

This repository contains implementations for solving partial differential equations (PDEs), with a focus on wave propagation in heterogeneous media.

## 2D Wave Simulation with Non-Homogeneous Media

A corrected implementation of 2D acoustic wave propagation with spatially-varying density ρ(x,y) and bulk modulus K(x,y).

### Features

- ✅ Proper non-homogeneous density field ρ(x,y)
- ✅ Spatially-varying bulk modulus K(x,y)
- ✅ Correct implementation of ∇·(K∇p) operator
- ✅ Absorbing boundary layers
- ✅ Real-time animation
- ✅ Educational examples and documentation

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the simulation
python wave_simulation_2d.py
```

### Files

- `wave_simulation_2d.py` - Main simulation with visualization
- `example_correct_implementation.py` - Educational example
- `README_wave_simulation.md` - Detailed documentation
- `COMPARISON.md` - Wrong vs correct implementation comparison

### Documentation

See [README_wave_simulation.md](README_wave_simulation.md) for detailed physics and implementation notes.

See [COMPARISON.md](COMPARISON.md) for a comparison of incorrect vs correct approaches.
