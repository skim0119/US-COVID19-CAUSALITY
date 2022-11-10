# US COVID-19 Spread Analysis/Visualization using Information Theory

> STAT 430 final project

## Contributed by

- Yashraj Bhosale
- Seung Hyun Kim

## How to run

We provide `jupyter notebook` file `covid_raw_data_testing.ipynb` to download and visualize correlation data between state-wise case reports.

To further analyze directional causality, use the script:

```sh
python analysis.py         # Single-core (debug)
mpirun python analysis.py  # Multi-core
```

## Related theories

### References 

- Schreiber, Thomas., Measuring Information Transfer, 2000, https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.85.461
