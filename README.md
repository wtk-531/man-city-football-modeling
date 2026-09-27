# Manchester City Football Modeling

**Bayesian team-strength updating, probabilistic score forecasting, and exploratory dynamic modeling**  
*Version 1 · September 2026*

This project models Manchester City's Premier League matches. It combines league-wide Bayesian strength updates with a Poisson score model, while documenting experiments with short-term dynamics, coaching changes, midfield composition, and transfer-based priors.

> **Project status:** Version 1 is frozen for prospective evaluation during 2026/27. Model parameters remain fixed during the season; team-strength estimates update as completed results are entered.

## About This Project

I'm an undergraduate studying applied mathematics and a Manchester City fan. I started this independent project to connect my interest in football with what I had been learning about Bayesian inference, statistical modeling, linear algebra, and dynamical systems.

It became an opportunity to practice the whole modeling process: defining assumptions, building representations, comparing models, and learning from results that challenged the original idea. I welcome feedback and discussion from others interested in football analytics, Bayesian methods, or applied mathematics.

## Where to Start

- **Read the project narrative:** [Full Integrated Reflection](docs/Manchester_City_Project_Full_Integrated_Reflection.ipynb) covers the reasoning, experiments, results, and limitations.
- **Use the predictor:** [Standalone Forecaster](forecaster/CITY_FORECASTER_STANDALONE.ipynb) provides the interface for recording results and predicting the next City match. See [Quick Start](#quick-start).
- **Check the data:** [Data Sources and Provenance](docs/DATA_SOURCES_AND_PROVENANCE.ipynb) documents providers, transformations, and dataset lineage.

## Research Questions and Development

The project asks whether evolving team strength can support useful score probabilities, and whether recent performance, coaching changes, or midfield composition add information beyond strength and venue.

The notebooks follow the development sequence below. These are historical research stages; they do not all rerun when a new match result is entered.

| Notebook | Main purpose |
|---|---|
| [team_strength.ipynb](notebooks/team_strength.ipynb) | Construct opponent-adjusted multivariate team-performance states and instability measures. |
| [strength_index_validation.ipynb](notebooks/strength_index_validation.ipynb) | Build a scalar strength index and evaluate its weights. |
| [strength_index_pca_elo_validation.ipynb](notebooks/strength_index_pca_elo_validation.ipynb) | Combine home/away information with PCA and compare the resulting index with ClubElo. |
| [E0_E1_bridge.ipynb](notebooks/E0_E1_bridge.ipynb) | Initialize promoted teams using Championship information. |
| [bayesian_prior.ipynb](notebooks/bayesian_prior.ipynb) | Calibrate season-opening priors and sequential league-wide strength updates. |
| [dynamic_system_model.ipynb](notebooks/dynamic_system_model.ipynb) | Evaluate City residual dynamics, structural scenarios, and score forecasts. |

## Operational Version 1

During the season, the workflow is to record completed league results, update strengths, and save a probability forecast for the next City fixture.

### Bayesian strength updates

Each club has a latent strength estimate. Season-opening uncertainty uses a common fitted variance scale, $Q_0$, multiplied by the club's relative instability in the previous season:

$$
P_{i,0}^{(s)}=Q_0M_{i,s-1}.
$$

Recorded results update the strength vector and its covariance matrix. The pre-match difference between City and its opponent, $\Delta S_t$, then enters the score model. ClubElo serves as an external comparison benchmark for the internally constructed index; its ratings are not direct inputs to issued forecasts.

### Score probabilities

Separate Poisson regressions model City goals for ($GF$) and against ($GA$):

$$
\log\lambda_{o,t}
=\alpha_o+\beta_o\Delta S_t+\gamma_o Home_t,
\qquad o\in\{GF,GA\}.
$$

Here, $Home_t$ indicates a City home match. Combining two conditionally independent Poisson distributions gives expected goals, win/draw/loss probabilities, and exact-score probabilities. The interface displays the five most likely scores, a 0–5 score grid, and the probability mass outside that grid.

## Exploratory Extensions

- **Short-term dynamics:** AR, VAR, and latent-state models tested persistence in opponent-adjusted possession, shots, shot-on-target rate, and finishing efficiency. They did not reliably improve the reported historical comparisons and are excluded from operational Version 1.
- **Coaching scenarios:** Chelsea's Maresca-era performance was compared with an earlier baseline to construct conditional scenarios for City. The current analysis does not isolate the coaching effect from simultaneous squad and tactical changes.
- **Midfield composition:** SVD reduces player information to six components, which a regularized model maps to four team-performance dimensions with strength controls. Same-season player statistics and minutes limit its interpretation as a preseason forecasting model.
- **Transfer adjustments:** Transfer expenditure and income were tested as prior adjustments but were not retained in the final specification.

Scenario adaptation equations were also examined using eigenvalue, Lyapunov, and trajectory checks. Their fixed-reference stability follows from the assumed dynamics; these checks do not establish real adaptation speeds or forecasting accuracy. The structural scenarios remain separate from issued score probabilities.

## Historical Evaluation

The City analysis uses 114 Premier League matches across 2023/24–2025/26. The first season supplies initial training data, the second supports model selection, and the third provides a historical evaluation. Models are refitted on the first 76 matches before that evaluation. The operational score model is subsequently refitted on all 114 matches for 2026/27.

The strength-and-venue specification was selected on 2024/25 validation data: mean exact-score negative log-likelihood (NLL) was **3.0992**, versus **3.2128** for venue only. The 38-match 2025/26 comparison, including a later exploratory state extension, was:

| Model | Mean score NLL | GF RMSE | GA RMSE |
|---|---:|---:|---:|
| Venue only | 2.9032 | 1.2360 | 0.9892 |
| Strength + venue | 2.9274 | 1.2285 | 1.0230 |
| Strength + venue + latent state | 2.9910 | 1.2368 | 1.1046 |

Lower values are better. NLL evaluates the probability assigned to the realized score; RMSE evaluates predicted goal counts. The selected model slightly improved goals-for RMSE but performed worse on score NLL and goals-against RMSE than venue only. Its Top 5 included the actual score in **17 of 38 matches**, compared with **22 of 38** for venue only.

These results do not establish a consistent advantage over the simpler benchmark. Version 1 retains the validation-selected specification for prospective testing. Later development revisited 2025/26, and some structural representations use pooled historical information, so this evidence comprises chronological historical comparisons and exploratory analyses, not a fully untouched end-to-end test.

To avoid choosing the model retrospectively based on the final historical season, Version 1 keeps the specification selected using the earlier validation period.

## Quick Start

The standalone forecaster runs without rerunning the historical research notebooks or loading their raw datasets. From a terminal at the repository root:

```bash
cd forecaster
python -m pip install -r requirements.txt
```

1. Open `forecaster/` as your working folder in VS Code, then open `CITY_FORECASTER_STANDALONE.ipynb`.
2. Select the Python environment where the requirements were installed and run the setup cell.
3. Enter completed league results dated before the next City fixture, including other teams' matches when available.
4. Enter the fixture date, opponent, and City venue, then save the forecast before kickoff.
5. After the match, record the result and review the season report.

**Date rule:** Version 1 uses calendar dates rather than kickoff times. Save City's forecast before entering any results bearing that fixture's date, including other teams' results from earlier that day.

Keep and back up `season_state.json` between sessions. It stores the frozen parameters, opening priors, recorded results, and saved forecasts needed to continue the season. Only forecasts recorded before their matches count as prospective evidence; retrospective replays should be identified separately.

## Data Sources

The table identifies the historical sources used during development. Some support the operational model; others support validation or exploratory work.

| Source | Data and role in this project |
|---|---|
| [Football-Data.co.uk](https://www.football-data.co.uk/data.php) | Premier League and Championship results and match statistics for team-strength construction, promotion analysis, and Bayesian updating; also supplies shot, shot-on-target, and finishing outcomes for the midfield bridge. |
| [ClubElo](https://clubelo.com/) | Historical club ratings used as an external comparison for the internally constructed strength index. |
| [FBref / Sports Reference](https://fbref.com/en/) | City and Chelsea match-log exports used in performance and coaching analyses; additional historical player/team tables collected during development. |
| [FotMob](https://www.fotmob.com/en-GB/leagues/47/stats/the-premier-league) | Player-statistic extracts used for midfield profiles, plus a separate team-season possession extract used in the midfield bridge. |
| [Understat](https://understat.com/league/EPL) | Historical expected-goal information and match extracts retained for exploratory analysis. |
| [Transfermarkt](https://www.transfermarkt.com/premier-league/transfers/wettbewerb/GB1/) | Summer-window expenditure and income used in exploratory tests of transfer-related prior adjustments. |

The midfield bridge combines **FotMob possession** with **Football-Data shot and finishing measures** and internally estimated strength controls. FotMob's team possession data were collected separately from its player statistics; folder names containing “FBref” do not change their attribution.

See [Data Sources and Provenance](docs/DATA_SOURCES_AND_PROVENANCE.ipynb) for the source-to-file record. Provider pages and available fields can change; saved extracts and processing records document the historical inputs used here.

Raw third-party datasets are not redistributed. Selected processed inputs and model-derived outputs are included where useful for reproducibility and permitted by the underlying data sources.

## Repository Guide

| Directory | Contents |
|---|---|
| `notebooks/` | The six research notebooks linked above. |
| `docs/` | The full integrated reflection and data-provenance notebooks. |
| `data/` | Data documentation in [README.md](data/README.md), plus `processed/` and `derived/` tables where included. |
| `forecaster/` | `CITY_FORECASTER_STANDALONE.ipynb`, `city_online_forecast.py`, `requirements.txt`, and `season_state.json`. |

The research notebooks document development and depend on their associated data and paths. The standalone forecaster is the entry point for continued use during the season.

## Limitations and Next Steps

The City-specific sample is small. The score model assumes conditionally independent Poisson goals and uses posterior mean strengths without integrating all parameter and state uncertainty. Coaching and midfield scenarios describe associations under explicit assumptions, rather than validated causal effects.

The immediate priority is a prospective evaluation of the fixed Version 1 model: preserve forecast timing, enter completed league results, and monitor score NLL, goal RMSE, outcome Brier scores, and calibration. A possible Version 2 would share information across clubs through a hierarchical state model, with all transformations fitted inside historical training windows and new inputs evaluated against the same baselines.

## AI-Assisted Development

Generative AI tools supported code implementation, debugging, documentation, and language refinement. I directed the modeling questions, assumptions, experiments, and interpretation, and used the project to learn how to check generated implementations against the intended mathematics. I remain responsible for the analyses and claims presented here.

## License and Data Use

No open-source license is currently granted for the original project code and documentation. Third-party football data remain subject to their providers' terms; source attribution does not itself grant permission to redistribute those datasets.


---

*This project is intended for educational and research purposes. It is not betting advice.*
