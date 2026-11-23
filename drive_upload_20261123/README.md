# Dataset and results downloads

Drive folder: [Download folder](https://drive.google.com/drive/folders/1SceShb6mBJXSaWDNzq6VrOgVajY-4ru0?usp=sharing)

| File | Bytes | SHA-256 | Supports | Role |
|---|---:|---|---|---|
| `proxy-data-final-pool-small.tar.gz` | 152684158 | `3471f14f4438bc4aee92557bab7e006edfbfe6573294ed930b143b09152d3e09` | Permuted MNIST, Permuted Kuzushiji-49, TAPE Stability, IMDB, enwik8, Lipophilicity raw CSV | Existing |
| `proxy-data-final-pool-mad.tar.gz` | 327195 | `bbb6ade421d87e8373b81c4d885e6cbde53a332e6f76dbfc780b4dddab853503` | MAD Compression | Existing |
| `proxy-data-final-pool-parity.tar.gz` | 279977 | `c333fecc00d8938b5ab2275eb5ce781eeec7b910a3e1bff89375eb5ad4fe3fb8` | Parity | Existing |
| `proxy-data-final-pool-speech_commands.tar.gz` | 2427893343 | `c77a5d5f7a3ffb8442abda5da0b461410f4900d6c50f3aa5a1a1ee36d2f01a61` | Speech Commands | Existing |
| `proxy-data-speech-cache.tar.gz` | 693340132 | `00af438336bae43b4e6145711cd5408876c47c63946aaf8f41d6208aa414664c` | Speech Commands exact evaluated arrays | Supplement |
| `proxy-data-tomita5.tar.gz` | 287766 | `af40131e417d004bbae40fab0c88788ef60b459ed6366422d30d63398a290a41` | Tomita 5 | supplement |
| `proxy-data-emnist.tar.gz` | 32111870 | `61e0969a195e48f970aa8e951eeee62714e5c9e6693d078876fa72201542e4cf` | Permuted EMNIST | supplement |
| `proxy-data-lipophilicity.tar.gz` | 6952 | `c6eac263c8a50b3090c87d64c0c4a774d3b9c3d5cc3e1f657fd3b678a2862e7b` | Lipophilicity frozen split; combine with small | supplement |
| `proxy-data-promoters.tar.gz` | 2567887 | `72245e300b5fc124e10b31c365766cc9ad0a7d16ef9690dc764fe0dd473ba42e` | Genomics (Human non-TATA promoters) | New |
| `proxy-data-bach.tar.gz` | 160200 | `984d96459d66b55a85116bdbb4276371dfae773d7e0f30bd0543ecd8407530c2` | Bach chorales | New |

Verify each archive against the SHA-256 above. Extract small, MAD and parity, then the six supplements into the same directory. The original Speech Commands WAV archive is optional when using the frozen Speech cache. Keep the paths inside each archive. The older combined additions archive is redundant when using these supplements.

The four existing archives cover eight file-backed tasks. Tomita 5, Permuted EMNIST, and the frozen Lipophilicity split require supplements. Genomics and Bach require the new archives. The Speech cache supplement keeps the exact evaluated float arrays; recomputing from the original raw Speech archive can differ slightly. The other five tasks generate data at runtime.

All data archives are below 4 GB. Detailed results are kept in private storage. Genomics arrays are derived from Genomic Benchmarks (CC BY 4.0); Bach arrays are derived from DCMLab Bach Chorales (CC0 1.0).

Historical run reports and the detailed score tables are kept in a separate
restricted supplement (`B_restricted_snapshot.tar.gz`). No install or data
preparation step requires it. Third-party components are documented in
[the dependency notices](../third_party/DEPENDENCIES.md).
