## Benchmarks

### Tracking

We use the trackML dataset from the Codalab challenge.

| Parameter | Pixel | Full detector |
| --------- | ----- | ------------- |
| N objects | 10k   | 11k           |
| N hits    | 66k   | 126k          |

The benchmarks below are for a GNN model with 1.9M parameters trained on the
pixel dataset. The exact numbers might depend on the model architecture and the
edges that were built.

| Data  | Loss fct             | Max memory | Persistent memory | Time | Notes       |
| ----- | -------------------- | ---------- | ----------------- | ---- | ----------- |
| Pixel | default              | 25         | 14                | 0.1  |
| Pixel | tiger no compilation | 12.3       | 4                 | 0.58 | `n_rep=11k` |
| Pixel | tiger compiled       | 9          | 6                 | 0.14 | `n_rep=11k` |
| Full  | tiger compiled       | 26         | 8                 | 0.18 | `n_rep=15k` |

Memory is in GB, time in s. Persistent memory is an estimate of the memory
consumption after loss has been evaluated but before backwards propagation. Max
memory is the maximum memory consumption encountered during the evaluation of
the loss function minus the memory just before starting evaluation of the loss
function.
