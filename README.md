# Manchester City Football Modeling

**Bayesian team-strength updating, probabilistic score forecasting, and exploratory dynamic modeling**  
*Version 1 · September 2026*

This project develops a sequential forecasting framework for **Manchester City Premier League matches**. The operational Version 1 model updates league-wide team strengths as new results arrive, converts Manchester City's pre-match strength difference and venue into expected goals, and produces full score and outcome probabilities.

The repository also documents several ideas that were tested but **not** retained in the final forecasting pipeline, including short-term residual dynamics, coaching-regime scenarios, midfield composition models, and transfer-based prior adjustments. A major goal of the project is not only to produce forecasts, but also to record which modeling ideas were supported by the available evidence and which were not.

> **Project status:** Version 1 is frozen for prospective evaluation during the 2026/27 season. Score-model coefficients remain fixed during the season; team strengths continue to update as completed results are entered.

## About This Project

I'm an undergraduate studying applied mathematics and a Manchester City fan. I started this as an independent project to connect my interest in football with methods I had been learning in Bayesian inference, statistical modeling, linear algebra, and dynamical systems.

The project gradually became less about building the most complicated possible predictor and more about learning how to make modeling decisions: how to define a useful latent quantity, how to separate prediction from interpretation, how to evaluate extensions against simpler baselines, and how to keep a model honest when an appealing idea does not improve out-of-sample performance.

I am sharing the project as part of that learning process and would be happy to exchange ideas with others interested in football analytics, Bayesian methods, applied mathematics, or statistical modeling.

## Project Questions

The project is organized around four main questions:

1. Can recent match results be used to maintain a meaningful, evolving estimate of team strength?
2. Can pre-match strength difference and venue be translated into a useful probability distribution over Manchester City scores?
3. Do short-term performance dynamics add predictive information beyond strength and venue?
4. Can coaching changes and midfield composition be represented mathematically without confusing structural interpretation with validated forecast improvement?

## Version 1 Architecture

The deployed forecasting chain is intentionally compact:

```text
completed EPL results
        ↓
Bayesian team-strength update
        ↓
pre-match strength difference (ΔS) + venue
        ↓
Poisson goal models for GF and GA
        ↓
exact-score probabilities + W/D/L probabilities
```

### 1. Bayesian team-strength layer

The model maintains a league-wide latent strength estimate for each club. Season-opening uncertainty is scaled using a common fitted variance parameter and a team-specific instability measure from the previous season:

$$
P_{i,0}=Q_0M_{i,s-1}.
$$

As completed league results are recorded, the strength vector and its covariance matrix are updated sequentially. The resulting pre-match difference between City and its opponent, $\Delta S_t$, is the main strength input to the score model.

ClubElo is used as an **external comparison benchmark** for the internally constructed strength representation; ClubElo values are not directly inserted into the operational forecasting model.

### 2. Score model

Manchester City's goals for and goals against are modeled separately with log-linked Poisson regressions:

$$
\log\lambda_{o,t}
=\alpha_o+\beta_o\Delta S_t+\gamma_o Home_t,
\qquad o\in\{GF,GA\}.
$$

The two estimated goal rates are combined using conditionally independent Poisson distributions. The model outputs:

- expected Manchester City and opponent goals;
- win, draw, and loss probabilities;
- the five most likely exact scores;
- a displayed 0–5 score-probability grid;
- probability mass outside the displayed 0–5 grid.

During the 2026/27 season, the Poisson regression coefficients remain frozen. Only the league-strength state changes as new results are entered.

## What Was Tested Beyond the Final Model

### Short-term dynamics

I tested whether recent deviations in four performance dimensions — possession, shots, shot-on-target rate, and finishing efficiency — contained useful predictive information after controlling for strength difference and venue.

AR, VAR, and latent-state formulations were compared with simpler no-dynamics baselines. In the saved historical comparisons, these extensions did **not** provide a reliable improvement, so they were excluded from the Version 1 operational rates.

### Coaching-regime scenarios

A separate analysis examined Chelsea's performance during the Maresca era relative to an earlier baseline. The purpose was to explore how a coaching-associated performance signature might be represented and transferred into a conditional scenario for City.

This is treated as **structural analysis rather than a causal estimate**. Simultaneous squad and tactical changes make it impossible to identify a pure coaching effect from the available data.

### Midfield composition

Player information was reduced with SVD and mapped to team-level performance using a regularized linear model with strength controls. This module is useful for exploring conditional squad scenarios, but same-season player statistics and minutes limit its use as a clean preseason forecast input.

For that reason, the midfield model remains separate from the operational Poisson score model.

## Historical Evaluation

The strength-and-venue score model was originally selected using 2024/25 validation data. Its mean exact-score negative log-likelihood was **3.0992**, compared with **3.2128** for a venue-only specification.

After refitting the selected specifications, the 38-match 2025/26 historical comparison produced the following results:

| Model | Mean score NLL | GF RMSE | GA RMSE |
|---|---:|---:|---:|
| Venue only | 2.9032 | 1.2360 | 0.9892 |
| Strength + venue | 2.9274 | 1.2285 | 1.0230 |
| Strength + venue + latent state | 2.9910 | 1.2368 | 1.1046 |

Lower values are better. The selected strength-and-venue model slightly improved goals-for RMSE relative to venue only, but it was worse on exact-score NLL and goals-against RMSE. Its Top-5 score set contained the realized score in **17 of 38 matches**, versus **22 of 38** for venue only.

The tested dynamic extensions also failed to improve their matched historical benchmarks. I therefore do **not** interpret the project as evidence that the more complex model is universally superior. Instead, Version 1 preserves the validation-selected specification as a fixed model for prospective testing while keeping the mixed historical evidence visible.

## Why Keep the Simpler Version?

One of the main lessons from the project is that mathematical structure, interpretability, and predictive accuracy are different things.

A dynamic system can be stable without improving forecasts. A plausible football narrative can be measurable without being causal. A richer feature set can describe a match more completely while still failing to predict the next one better.

For that reason, Version 1 deliberately keeps the deployed model narrower than the full research process. The more exploratory modules remain documented, but they are not silently added to the issued forecast probabilities.

## Operational Forecaster

The standalone package is the user-facing version of the model. It can run without rerunning the historical research notebooks.

Its saved state contains:

- the frozen 2026/27 Bayesian parameters;
- the corrected 2026/27 league roster and opening priors;
- the frozen `Strength_Home` Poisson score model;
- completed results already entered into the season state;
- an append-only archive of later results and issued forecasts.

The operating rule is chronological: completed results dated **strictly before** a fixture are entered first, the City forecast is saved before the fixture, and later results update the strength state for future predictions.

Because Version 1 stores calendar dates rather than kickoff times, results from the same date should not be entered before that date's City forecast is saved.

## Quick Start

The standalone forecaster requires Python plus the packages listed in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

Then:

1. Open `CITY_FORECASTER_STANDALONE.ipynb`.
2. Select the Python environment in which the requirements were installed.
3. Run the setup cell.
4. Enter completed results dated before the next City fixture.
5. Enter the opponent, venue, and fixture date and save the forecast before kickoff.
6. After the match, record the result and review the season report.
7. Keep `season_state.json` between sessions; it stores the evolving model state and forecast history.

## Data Sources

The project combines data from several public football-data providers. Not every collected variable is used by the deployed model; some sources support validation or exploratory modules only.

| Source | Main role in the project |
|---|---|
| [Football-Data.co.uk](https://www.football-data.co.uk/data.php) | Premier League and Championship results and match statistics; team-strength construction, promotion analysis, and Bayesian updating |
| [ClubElo](https://clubelo.com/) | External comparison for the internally constructed strength index |
| [FBref / Sports Reference](https://fbref.com/en/) | Manchester City and Chelsea match logs and historical player/team tables |
| [FotMob](https://www.fotmob.com/en-GB/leagues/47/stats/the-premier-league) | Player profiles and team-season possession statistics used in structural analyses |
| [Understat](https://understat.com/league/EPL) | Expected-goal information and exploratory historical match extracts |
| [Transfermarkt](https://www.transfermarkt.com/premier-league/transfers/wettbewerb/GB1/) | Summer transfer expenditure/income used in exploratory prior-adjustment tests |

A separate provenance document records the lineage of the major local datasets and processing notebooks. Third-party data remain subject to the terms and attribution requirements of their original providers.

## Repository Guide

The public project is organized around four layers:

```text
man-city-football-modeling/
│
├── README.md
│
├── notebooks/
│   ├── team_strength.ipynb
│   │   └── opponent-adjusted multivariate team states
│   │
│   ├── strength_index_validation.ipynb
│   │   └── scalar strength construction and weight validation
│   │
│   ├── strength_index_pca_elo_validation.ipynb
│   │   └── home-away PCA and ClubElo validation
│   │
│   ├── E0_E1_bridge.ipynb
│   │   └── promoted-team strength initialization
│   │
│   ├── bayesian_prior.ipynb
│   │   └── recursive Bayesian priors and league-wide strength updates
│   │
│   └── dynamic_system_model.ipynb
│       └── City residual dynamics, structural scenarios, and score forecasting
│
├── docs/
│   ├── Manchester_City_Project_Full_Integrated_Reflection.ipynb
│   │   └── full project narrative, experiments, rejected ideas, and reflection
│   │
│   └── DATA_SOURCES_AND_PROVENANCE.ipynb
│       └── data sources, attribution, and dataset lineage
│
└── forecaster/
    ├── CITY_FORECASTER_STANDALONE.ipynb
    ├── city_online_forecast.py
    ├── requirements.txt
    └── season_state.json
```

The historical research notebooks explain how the model was developed. The `forecaster/` folder is the smaller operational package intended for continued use during the season.

Raw or intermediate provider datasets are not required to run the standalone forecaster. Where data are shared, their original sources should be credited according to the provenance document.

## Main Limitations

The current version has several important limitations:

- The City-specific score-regression sample is small, and later stages of the project revisit some historical periods repeatedly.
- The historical evidence is therefore better described as chronological comparison plus exploratory analysis than as one untouched end-to-end test.
- The score model assumes conditional independence between City and opponent goals.
- Forecast probabilities use posterior mean strengths rather than integrating all parameter and state uncertainty.
- Coaching and midfield modules describe associations and conditional scenarios; they do not establish causal effects.
- A meaningful assessment of live performance requires a genuinely prospective archive of forecasts created before matches.

These limitations are part of the project rather than hidden implementation details. They directly motivate the next stage of evaluation.

## Next Steps

The immediate priority is to evaluate Version 1 prospectively during the 2026/27 season while keeping the model specification fixed. That includes preserving forecast timing, entering completed league results, and tracking exact-score NLL, goal RMSE, outcome Brier scores, and calibration.

A later Version 2 may explore a league-wide hierarchical state model so that short-term information can be shared across clubs and opponent states can be represented directly. Any such extension should be compared against the same strength baseline under a controlled chronological design before it is added to the operational forecaster.

## Project Perspective

The original idea was to combine Bayesian inference, dynamic systems, coaching changes, player composition, and probabilistic score forecasting in one model. The development process ultimately suggested a more restrained conclusion: not every mathematically interesting component deserves to become a forecasting input.

For me, that is one of the most useful outcomes of the project. The current model is simpler than the initial concept, but the repository preserves the reasoning, failed extensions, validation choices, and limitations that led to it.

## Feedback

This is an independent undergraduate project and an ongoing learning exercise. Suggestions, criticism, and discussion are welcome — especially around Bayesian sports modeling, sequential evaluation, probabilistic forecasting, or alternative ways to represent football dynamics.

## License and Data Use

No open-source license is currently granted for the original project code and documentation. Third-party football data remain subject to the terms of their respective providers. Please consult the provenance documentation before reusing or redistributing external datasets.

---

*This project is intended for educational and research purposes. It is not betting advice.*
