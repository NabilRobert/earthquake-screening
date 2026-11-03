# Sources and Their Terms

Domain knowledge comes from papers and public guidance. We use their
methods and cite them; no text is copied. All read 2026-11-03.
"Unconfirmed" means no primary source was found.

## Guidance and models

| Source | What we use | Terms |
|---|---|---|
| [FEMA P-58](https://femap58.atcouncil.org/) (2nd ed., 2018) | The method from drift to damage states and loss; free PACT tool | No licence on ATC's pages; fema.gov says most of its material is free of copyright ([reuse](https://www.fema.gov/about/website-information)). Reuse of the fragility data: unconfirmed |
| [Hazus earthquake model](https://www.fema.gov/flood-maps/products-tools/hazus) (7.2; technical manual 6.1) | Capacity and fragility curves by building class, as the baseline | As above. The software needs ArcGIS Pro (paid), so we take the parameters from pelicun's library instead |
| [pelicun](https://github.com/NHERI-SimCenter/pelicun) 3.10 and [DamageAndLossModelLibrary](https://github.com/NHERI-SimCenter/DamageAndLossModelLibrary) | Hazus and P-58 damage models as code and data | BSD-3. The FEMA data inside isn't licensed separately: unconfirmed |

## Ground motions

| Source | Terms | Used |
|---|---|---|
| [ESM v2.0](https://data.ingv.it/dataset/418) | CC BY 4.0; cite Luzi et al. (2020), doi:10.13127/esm.2 | Slice 1 |
| [PEER NGA-West2](https://ngawest2.berkeley.edu/) | © UC Regents, all rights reserved; no grant to redistribute; account needs a university or company email; capped downloads | No (ADR-005) |
| [K-NET/KiK-net](https://www.mowlas.bosai.go.jp/policy/data/?LANG=en) | Registration; no redistribution; uses outside research must be reported to NIED | No |
| [CESMD](https://www.strongmotioncenter.org/NCESMD/reports/CESMD_DataPolicy.htm) | Citation encouraged; commercial use not addressed | Not yet |
| AFAD (Türkiye), BMKG (Indonesia) | Unconfirmed | No |

## Datasets

| Dataset | Content | Terms |
|---|---|---|
| [DesignSafe PRJ-2048](https://www.designsafe-ci.org/api/publications/v2/PRJ-2048) (Guan, Burton and Shokrabadi, 2020) | 621 steel moment frames; peak drift, floor acceleration and residual drift under 240 records | Open Data Commons Attribution. Built from records whose own rights may differ: we use responses only and don't redistribute |

## Papers

| Paper | Role |
|---|---|
| Sainct, Feau, Martinez, Garnier (2020), Structural Safety 86:101972; [arXiv:1810.01240](https://arxiv.org/abs/1810.01240) | Slice 1 reproduction (ADR-006) |
| Rezaeian and Der Kiureghian (2010), stochastic ground-motion model | The record generator Sainct et al. use |
| Guan, Burton, Shokrabadi (2021), Earthquake Spectra 37(2); Guan's [dissertation](https://escholarship.org/uc/item/5bd6r600) | Slice 2 data and comparison |
| Li and Ceferino (2026), [engrXiv 10.31224/7702](https://engrxiv.org/preprint/view/7702), CC BY | Where neural surrogates fail; a reference, not a target |
| Gentile and Galasso (2020), Structural Safety 87:101980 | Gaussian processes for portfolio fragility; background |

## Licences of the engines

See the [licence gate](licence-gate.md).
