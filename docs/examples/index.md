# Examples

The notebooks live in
[`examples/`](https://github.com/northflowlabs/climval/tree/main/examples)
in the repository and are shown here with their stored outputs. Both read
public data anonymously from Google Cloud Storage; no account or key is
needed.

| Notebook | Data | Read from the cloud |
|---|---|---|
| [ERA5 against CMIP6 over Europe](era5_cmip6_europe.ipynb) | ARCO-ERA5 2 m temperature; five CMIP6 historical models (CanESM5, IPSL-CM6A-LR, MPI-ESM1-2-LR, NorESM2-LM, UKESM1-0-LL); Europe, 1995 to 2014 | about 560 MB |
| [A CI gate on real data](ci_gate_real_data.ipynb) | ARCO-ERA5; NorESM2-LM and MPI-ESM1-2-LR; AR6 Northern Europe land, 2005 to 2014 | about 170 MB |

To run them:

```bash
pip install "climval[cloud,catalog,regions]" matplotlib jupyter
jupyter lab examples/
```

`examples/basic_comparison.py` runs `climval.compare` on the synthetic
demonstration fields, without any download.
