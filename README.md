# An Open-source Supported Guide to High-speed Camera Based Structural Dynamics Identification

**Code repository for the tutorial paper published in *Mechanical Systems and Signal Processing*.**

> D. Gorjup, K. Zaletelj, J. Slavič,
> *An Open-source Supported Guide to High-speed Camera Based Structural Dynamics Identification*,
> Mechanical Systems and Signal Processing, TODO (TODO).
> DOI: [TODO](https://doi.org/TODO)

## Getting started

The main entry point is **`image_based_dynamics_tutorial.ipynb`** — a Jupyter notebook that walks through the
complete measurement pipeline in the same order as the paper.  Start there.

Supporting implementations are in `methods/` (`calibration`, `triangulation`, `modal`,
`visualization`, `utils`); the notebook imports from these as needed.

Dependencies: Python 3.11+, [pyIDI](https://github.com/ladisk/pyidi),
[SDyPy](https://github.com/sdypy/sdypy). See `pyproject.toml` for the full list.

## Citation

If you use this code in your work, please cite the accompanying paper:

```bibtex
@article{gorjup2026tutorial,
  author  = {Gorjup, Domen and Zaletelj, Klemen and Slavi\v{c}, Janko},
  title   = {An Open-source Supported Guide to High-speed Camera Based Structural Dynamics Identification},
  journal = {Mechanical Systems and Signal Processing},
  year    = {TODO},
  volume  = {TODO},
  pages   = {TODO},
  doi     = {TODO},
}
```

## Authors

- Domen Gorjup — University of Ljubljana, Faculty of Mechanical Engineering
- Klemen Zaletelj — University of Ljubljana, Faculty of Mechanical Engineering
- Janko Slavič — University of Ljubljana, Faculty of Mechanical Engineering
