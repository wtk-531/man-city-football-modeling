# Reusable Manchester City Forecaster — User Guide

This guide supplements the [main README](../README.md). **You do not need a new ZIP or a new model for each opponent.**

## Initial setup (once)

1. Download or clone the repository.
2. Open the `forecaster` directory in VS Code.
3. In a terminal, run: `python -m pip install -r requirements.txt`.
4. Open `FORECAST_EVERY_MATCH.ipynb`. Select the Python environment you used above.
5. Keep `season_state.json`, `city_online_forecast.py`, `FORECAST_EVERY_MATCH.ipynb`, and `epl_results.csv` in the **same folder**.

## Before **every** City Premier League game

1. Update `epl_results.csv` with finished league games since your previous prediction. One row per game; do not add future games or cup fixtures.
2. Change the three values at the top of `FORECAST_EVERY_MATCH.ipynb`:

   ```python
   NEXT_DATE = "2026-10-11"     # example only: change for each fixture
   NEXT_OPPONENT = "Liverpool"  # match a name from the strength table
   NEXT_VENUE = "Away"          # from Manchester City's perspective
   ```
3. Run all cells in order **before kickoff**. The notebook imports eligible prior results, updates Bayesian league strengths, and saves the match forecast.
4. Preserve `season_state.json`; it holds prior results and issued forecasts. Do not replace it with the original repository snapshot, or you will lose local records.

## After each game

Append the final score to `epl_results.csv` (in **home-team-first** order). It will be imported the next time you run the notebook. Review prior predictions with the season-report cell.

CSV columns (names must match exactly):

```csv
Date,HomeTeam,AwayTeam,FTHG,FTAG
2026-08-23,Man City,Bournemouth,2,1
```

This is an example already present in the initial state. Scores are **home team first**; they are not City first when City plays away.

## Important reproducibility rules

- Only results dated **strictly before** the predicted match are imported. The underlying Version 1 model uses calendar dates, not kickoff timestamps. On the same day, save the City forecast first; do not import any matches from that date beforehand.
- Results are deduplicated by fixture, so old rows may safely remain in the CSV. Conflicting scores are rejected.
- A successful `predict()` saves a forecast immediately. Only forecasts genuinely made **before kickoff** should count as prospective evaluations. The program records the save time, but does not verify kickoff time.
- Model coefficients remain fixed; entering new results updates latent team-strength estimates, not the training fit.
- The file `epl_results.csv` is a user-maintained log, **not an automatic live feed**. The provided initial CSV is a previously compiled snapshot through 2026-10-10; independently verify it against official league match records if evaluating results publicly.
- Back up both `season_state.json` and `epl_results.csv` regularly. Use only one working state file to avoid diverging forecast histories.

## Common errors

- **Unknown team:** Inspect the team names printed by the setup cell. The CSV and `NEXT_OPPONENT` must match the model's roster exactly.
- **Forecast date must be later than all entered result dates:** Some results from the target date or a later date are already stored in `season_state.json`. A saved-state backup from before those imports is required for a truly contemporaneous forecast.
- **Conflicting result for fixture:** Correct the score in the existing CSV / state carefully instead of adding a second inconsistent record. The operational append-only state intentionally rejects conflicts.
- **FileNotFoundError:** Open VS Code in `forecaster/` so relative paths point to the model and data files.

This notebook is an optional convenience interface; the original `CITY_FORECASTER_STANDALONE.ipynb` remains valid and supports manual entry of both City and other league results.
