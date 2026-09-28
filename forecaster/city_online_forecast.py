"""Match-by-match City forecasts, continuing notebook Sections 8 and 9.2.

Requires numpy, pandas and scipy (already used by the notebook).
Place this file beside the notebook and import CitySeasonForecaster.

Create once from the *original* preseason prior CSV, Bayesian parameter CSV,
and the fitted forecast_score_models objects. Later use .load(state_file).
Each saved JSON contains the original model snapshot, completed EPL results,
and issued predictions. Keep this JSON when moving computers or folders.

Daily use:
    prediction = online.predict(opponent=..., venue="Home", date="YYYY-MM-DD")
    display(prediction["Match"])
    display(prediction["Outcome_Probabilities_pct"].round(2))
    display(prediction["Top5"].round(2))
    # Only after the match finishes:
    online.record_city_result(opponent=..., venue="Home", date=..., gf=..., ga=...)

For other EPL results, use record_results with columns:
Date, HomeTeam, AwayTeam, FTHG, FTAG. Goals in record_city_result are always
City-first; goals in record_results are home-first.

Model:
    home_goal_difference = h + beta * (S_home - S_away) + error,
    error variance = R.
Updates preserve the full team covariance matrix and match the earlier
daily Gaussian replay. Q0 is already in the opening variances; it is NOT
added each match. Poisson regression coefficients stay fixed during this
season. Score probabilities use posterior mean strengths (plug-in rates).
Coach/midfield scenarios and the four-dimensional performance filter are
not included in these rates. Missing league results are not inferred.

Dates are calendar dates: all predictions for one date precede that day's
updates. Same-date results are assimilated jointly on chronological replay.
Previously issued predictions are immutable, including when late results
are supplied. Use one notebook/kernel to write a given state file.
"""

from __future__ import annotations

from datetime import date as Date, datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import poisson, skellam


def _date(value):
    """Accept calendar dates or explicit ISO / football-data date formats."""
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%Y/%m/%d"):
            try:
                return datetime.strptime(value.strip(), fmt).date().isoformat()
            except ValueError:
                pass
    elif isinstance(value, (Date, datetime, pd.Timestamp, np.datetime64)):
        ts = pd.Timestamp(value)
        if not pd.isna(ts) and ts.tzinfo is None and ts == ts.normalize():
            return ts.date().isoformat()
    raise ValueError(f"Use a calendar date (YYYY-MM-DD), without time: {value!r}")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return sha256(_canonical(value).encode("utf-8")).hexdigest()


def _write_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _distribution(lambda_city, lambda_opp):
    """Same independent-Poisson probabilities and tail rule as Section 8.3."""
    rates = np.array([lambda_city, lambda_opp], dtype=float)
    if not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError("Goal rates must be finite and positive.")
    limits_float = np.maximum(5, poisson.ppf(1 - 1e-10 / 2, rates))
    if not np.isfinite(limits_float).all() or np.prod(limits_float + 1) > 1_000_000:
        raise ValueError("Implausibly large score grid; check strengths and coefficients.")
    limits = limits_float.astype(int)
    joint = np.outer(poisson.pmf(np.arange(limits[0] + 1), rates[0]),
                     poisson.pmf(np.arange(limits[1] + 1), rates[1]))
    flat_indices = np.argsort(-joint.ravel(), kind="stable")[:5]
    city_goals, opponent_goals = np.unravel_index(flat_indices, joint.shape)
    probabilities = joint[city_goals, opponent_goals]
    tails = poisson.sf(limits, rates)
    omitted = float(tails.sum() - tails.prod())
    if omitted > 1.01e-10 or probabilities[-1] <= omitted:
        raise ValueError("Score-grid accuracy check failed.")
    top5 = [{"Rank": rank, "Score_City_Opp": f"{city}-{opponent}",
             "Probability": float(probability)}
            for rank, (city, opponent, probability) in enumerate(
                zip(city_goals, opponent_goals, probabilities), start=1)]
    tails5 = poisson.sf(5, rates)
    summary = {
        "P_Win": float(skellam.sf(0, *rates)),
        "P_Draw": float(skellam.pmf(0, *rates)),
        "P_Loss": float(skellam.cdf(-1, *rates)),
        "Top5_Probability_Mass": float(probabilities.sum()),
        "P_Outside_0_5": float(tails5.sum() - tails5.prod()),
    }
    if not np.isfinite(list(summary.values())).all() or not np.isclose(
        sum(summary[k] for k in ("P_Win", "P_Draw", "P_Loss")), 1, atol=1e-10
    ):
        raise ValueError("Outcome probability check failed.")
    return summary, top5, joint[:6, :6]


class CitySeasonForecaster:
    """A frozen preseason model snapshot plus an append-only result history."""

    RESULT_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]

    @classmethod
    def create(cls, *, state_file, prior, bayes_params, score_models, feature_sets,
               selected_model, training_cutoff, season="2026/27", city="Man City"):
        """Create a new season. Existing state files are never reset here."""
        path = Path(state_file).expanduser().resolve()
        if path.exists():
            raise FileExistsError(f"State already exists. Use .load(...): {path}")
        required = {"Team", "Prior_Mean", "Prior_Variance"}
        if not required.issubset(prior.columns):
            raise ValueError(f"Prior requires columns: {sorted(required)}")
        season = season.replace("-", "/")
        if not re.fullmatch(r"\d{4}/\d{2}", season):
            raise ValueError("Season must look like 2026/27.")
        year = int(season[:4])
        if int(season[-2:]) != (year + 1) % 100:
            raise ValueError("Season years must be consecutive.")
        if "Season" in prior and not prior["Season"].astype(str).str.replace(
            "-", "/", regex=False
        ).eq(season).all():
            raise ValueError("Prior rows must belong to the target season.")
        opening = prior.loc[:, ["Team", "Prior_Mean", "Prior_Variance"]].copy()
        if opening["Team"].isna().any():
            raise ValueError("Missing team names in prior.")
        opening["Team"] = opening["Team"].astype(str).str.strip()
        if opening["Team"].eq("").any() or opening["Team"].duplicated().any():
            raise ValueError("Prior team names must be nonempty and unique.")
        opening = opening.sort_values("Team").reset_index(drop=True)
        opening[["Prior_Mean", "Prior_Variance"]] = opening[
            ["Prior_Mean", "Prior_Variance"]
        ].astype(float)
        values = opening[["Prior_Mean", "Prior_Variance"]].to_numpy()
        if not np.isfinite(values).all() or (values[:, 1] < 0).any():
            raise ValueError("Invalid prior means or variances.")
        if len(opening) < 2 or city not in set(opening["Team"]):
            raise ValueError("The prior must include City and at least one opponent.")
        parameters = {k: float(bayes_params[k]) for k in ("h", "beta", "R")}
        if "Q0" in bayes_params:
            parameters["Q0"] = float(bayes_params["Q0"])
            if parameters["Q0"] < 0:
                raise ValueError("Q0 must be nonnegative.")
            if "M" in prior and not np.allclose(
                prior["Prior_Variance"].astype(float),
                parameters["Q0"] * prior["M"].astype(float), rtol=1e-5, atol=1e-9
            ):
                raise ValueError("Prior_Variance does not agree with Q0 * M.")
        if (not np.isfinite(list(parameters.values())).all()
                or parameters["beta"] <= 0 or parameters["R"] <= 0):
            raise ValueError("Require finite h and positive beta and R.")
        models = {}
        for name, fitted_targets in score_models.items():
            features = list(feature_sets[name])
            if (not features or len(features) != len(set(features))
                    or not set(features).issubset({"Home", "Delta_S"})):
                raise ValueError("Supported score features are Home and Delta_S.")
            models[name] = {"features": features, "targets": {}}
            for target in ("GF", "GA"):
                fitted = fitted_targets[target]
                coefficients = np.asarray(fitted.coef_, dtype=float)
                intercept = float(fitted.intercept_)
                if coefficients.shape != (len(features),) or not np.isfinite(
                    np.r_[intercept, coefficients]
                ).all():
                    raise ValueError(f"Invalid score coefficients: {name}/{target}")
                if hasattr(fitted, "feature_names_in_") and list(
                    fitted.feature_names_in_
                ) != features:
                    raise ValueError("Score feature order differs from fitted model.")
                models[name]["targets"][target] = {
                    "intercept": intercept, "coefficients": coefficients.tolist()
                }
        if selected_model not in models:
            raise ValueError("Selected score model is missing.")
        cutoff = _date(training_cutoff)
        if cutoff >= f"{year}-07-01":
            raise ValueError("Training cutoff must precede the target season.")
        config = {"season": season, "city": city, "training_cutoff": cutoff,
                  "prior": opening.to_dict("records"), "bayes": parameters,
                  "score_models": models, "selected_model": selected_model,
                  "season_start": f"{year}-07-01", "season_end": f"{year+1}-06-30"}
        _write_atomic(path, {"schema_version": 1, "config": config,
                             "results": [], "forecasts": []})
        return cls.load(path)

    @classmethod
    def load(cls, state_file):
        """Resume using the saved full-precision parameters and completed results."""
        obj = cls()
        obj.path = Path(state_file).expanduser().resolve()
        obj._read()
        return obj

    def _read(self):
        with self.path.open(encoding="utf-8") as stream:
            self._saved = json.load(stream)
        if self._saved.get("schema_version") != 1:
            raise ValueError("Unsupported state-file version.")
        self._config = self._saved["config"]
        self._opening = pd.DataFrame(self._config["prior"])
        self._teams = self._opening["Team"].tolist()
        self._index = {team: i for i, team in enumerate(self._teams)}
        self._results = self._normalize_results(pd.DataFrame(
            self._saved["results"], columns=self.RESULT_COLUMNS
        ))
        self._mu, self._covariance = self._replay(self._results)

    def _match_date(self, value):
        day = _date(value)
        if not (self._config["season_start"] <= day <= self._config["season_end"]):
            raise ValueError(f"Date is outside target season {self._config['season']}.")
        return day

    def _team(self, value):
        if not isinstance(value, str) or value.strip() not in self._index:
            raise ValueError(f"Unknown team {value!r}. Use names from strength_table().")
        return value.strip()

    def _normalize_results(self, frame):
        if not set(self.RESULT_COLUMNS).issubset(frame.columns):
            raise ValueError(f"Results require columns {self.RESULT_COLUMNS}")
        rows, seen, team_days = [], {}, set()
        for record in frame[self.RESULT_COLUMNS].to_dict("records"):
            row = {"Date": self._match_date(record["Date"]),
                   "HomeTeam": self._team(record["HomeTeam"]),
                   "AwayTeam": self._team(record["AwayTeam"])}
            if row["HomeTeam"] == row["AwayTeam"]:
                raise ValueError("A team cannot play itself.")
            for name in ("FTHG", "FTAG"):
                number = float(record[name])
                if not np.isfinite(number) or number < 0 or not number.is_integer():
                    raise ValueError("Completed scores must be nonnegative integers.")
                row[name] = int(number)
            key = (row["HomeTeam"], row["AwayTeam"])
            if key in seen:
                if row != seen[key]:
                    raise ValueError(f"Conflicting result for fixture {key}; nothing saved.")
                continue
            for team in key:
                team_day = (row["Date"], team)
                if team_day in team_days:
                    raise ValueError(f"Two fixtures for {team} on {row['Date']}.")
                team_days.add(team_day)
            seen[key] = row
            rows.append(row)
        return sorted(rows, key=lambda r: (r["Date"], r["HomeTeam"], r["AwayTeam"]))

    def _replay(self, results):
        mean = self._opening["Prior_Mean"].to_numpy(dtype=float, copy=True)
        covariance = np.diag(self._opening["Prior_Variance"].to_numpy(dtype=float))
        if not results:
            return mean, covariance
        params = self._config["bayes"]
        identity = np.eye(len(mean))
        for _, day in pd.DataFrame(results).groupby("Date", sort=True):
            design = np.zeros((len(day), len(mean)))
            row_indices = np.arange(len(day))
            design[row_indices, day["HomeTeam"].map(self._index)] = params["beta"]
            design[row_indices, day["AwayTeam"].map(self._index)] = -params["beta"]
            predicted = params["h"] + design @ mean
            actual = (day["FTHG"] - day["FTAG"]).to_numpy(dtype=float)
            cross_covariance = covariance @ design.T
            innovation_covariance = design @ cross_covariance + params["R"] * np.eye(len(day))
            gain = np.linalg.solve(innovation_covariance, cross_covariance.T).T
            mean = mean + gain @ (actual - predicted)
            # Joseph form preserves the full posterior covariance numerically.
            residual_map = identity - gain @ design
            covariance = (residual_map @ covariance @ residual_map.T
                          + params["R"] * gain @ gain.T)
            covariance = (covariance + covariance.T) / 2
        if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
            raise ValueError("Non-finite posterior; no result changes were saved.")
        return mean, covariance

    def summary(self):
        self._read()
        city = self._config["city"]
        return pd.DataFrame([{
            "Target_Season": self._config["season"],
            "Selected_Model": self._config["selected_model"],
            "N_Teams": len(self._teams), "N_Results": len(self._results),
            "N_City_Results": sum(city in (r["HomeTeam"], r["AwayTeam"])
                                  for r in self._results),
            "Last_Result_Date": self._results[-1]["Date"] if self._results else None,
            "N_Saved_Forecasts": len(self._saved["forecasts"]),
        }])

    def strength_table(self):
        self._read()
        return pd.DataFrame({"Team": self._teams, "Strength_Mean": self._mu,
                             "Strength_SD": np.sqrt(np.maximum(0, np.diag(self._covariance)))
                             }).sort_values("Strength_Mean", ascending=False).reset_index(drop=True)

    def record_city_result(self, *, opponent, venue, date, gf, ga):
        """Enter an actual completed EPL score in City-opponent order."""
        self._read()
        if venue not in ("Home", "Away"):
            raise ValueError("Venue must be Home or Away.")
        city = self._config["city"]
        home, away, hg, ag = ((city, opponent, gf, ga) if venue == "Home"
                              else (opponent, city, ga, gf))
        return self.record_results(pd.DataFrame([{
            "Date": date, "HomeTeam": home, "AwayTeam": away, "FTHG": hg, "FTAG": ag
        }]))

    def record_results(self, completed_results):
        """Add completed EPL games; exact duplicates are skipped, conflicts fail.

        Late results are allowed. The current posterior is rebuilt in date order;
        already issued forecasts retain exactly the information originally used.
        """
        self._read()
        incoming = self._normalize_results(completed_results)
        combined = self._normalize_results(pd.DataFrame(
            self._results + incoming, columns=self.RESULT_COLUMNS
        ))
        added = len(combined) - len(self._results)
        self._replay(combined)  # Validate the complete update before writing.
        if added:
            self._saved["results"] = combined
            _write_atomic(self.path, self._saved)
        return self.summary().assign(Added_Results=added)

    def predict(self, *, opponent, venue, date, model=None):
        """Issue and save a forecast using supplied earlier-date results only."""
        self._read()
        opponent = self._team(opponent)
        city = self._config["city"]
        if opponent == city or venue not in ("Home", "Away"):
            raise ValueError("Choose an opponent and venue Home or Away.")
        day = self._match_date(date)
        home, away = (city, opponent) if venue == "Home" else (opponent, city)
        if any((r["HomeTeam"], r["AwayTeam"]) == (home, away) for r in self._results):
            raise ValueError("This fixture already has a result. Read forecast_history().")
        if self._results and day <= self._results[-1]["Date"]:
            raise ValueError("Forecast date must be later than all entered result dates.")
        model = model or self._config["selected_model"]
        if model not in self._config["score_models"]:
            raise ValueError(f"Unknown score model: {model}")
        city_index, opp_index = self._index[city], self._index[opponent]
        difference = float(self._mu[city_index] - self._mu[opp_index])
        features = {"Delta_S": difference, "Home": int(venue == "Home")}
        spec = self._config["score_models"][model]
        vector = np.array([features[name] for name in spec["features"]])
        with np.errstate(over="raise", invalid="raise"):
            rates = [float(np.exp(spec["targets"][target]["intercept"]
                      + vector @ np.array(spec["targets"][target]["coefficients"])))
                     for target in ("GF", "GA")]
        probabilities, top5, grid = _distribution(*rates)
        result_hash = _digest(self._results)
        forecast_id = _digest([_digest(self._config), result_hash, day, opponent, venue, model])
        match = {"Forecast_ID": forecast_id[:16], "Date": day, "Venue": venue,
                 "Opponent_Key": opponent, "Model": model,
                 "S_City": float(self._mu[city_index]), "S_Opp": float(self._mu[opp_index]),
                 "Delta_S": difference, "Lambda_City": rates[0], "Lambda_Opp": rates[1],
                 "N_Results_Used": len(self._results),
                 "Last_Result_Date": self._results[-1]["Date"] if self._results else None}
        if not any(row["id"] == forecast_id for row in self._saved["forecasts"]):
            self._saved["forecasts"].append({
                "id": forecast_id, "issued_utc": datetime.now(timezone.utc).isoformat(),
                "results_sha256": result_hash, "results_used": self._results,
                "match": match, "probabilities": probabilities, "top5": top5,
            })
            _write_atomic(self.path, self._saved)
        return {
            "Match": pd.DataFrame([match]),
            "Outcome_Probabilities_pct": pd.DataFrame([probabilities]) * 100,
            "Top5": pd.DataFrame(top5).set_index("Rank").assign(
                Probability_pct=lambda table: table["Probability"] * 100
            )[["Score_City_Opp", "Probability_pct"]],
            "Grid_0_5": pd.DataFrame(grid, index=pd.Index(range(6), name="City goals"),
                                     columns=pd.Index(range(6), name="Opponent goals")),
        }

    def forecast_history(self):
        """All issued predictions; later outcomes never replace these rates."""
        self._read()
        rows = []
        for forecast in self._saved["forecasts"]:
            row = dict(forecast["match"], Issued_UTC=forecast["issued_utc"])
            row.update({f"Top{r['Rank']}": f"{r['Score_City_Opp']} ({r['Probability']:.2%})"
                        for r in forecast["top5"]})
            row.update(forecast["probabilities"])
            rows.append(row)
        return pd.DataFrame(rows)

    def correct_2026_27_roster(self):
        """Repair the verified 2026/27 roster without changing retained teams.

        Add Leeds and Sunderland; remove West Ham and Wolves. New means were
        recovered from the archived 2025/26 Bayesian match history, including
        indirect updates after each team's last fixture. Reconstruction agreed
        with every archived pre/post strength to 3.2e-15. M uses the existing
        team_strength_all_results.xlsx / Instability_U sheet (2025/26 EPL).
        Parameter source: bayesian_model_parameters.csv, final recursive fit.
        No 2026/27 score is used to construct these opening priors.

        The correction is specific to the previously checked final calibration.
        Existing results and issued forecasts are preserved. The prior config
        and a complete pre-correction backup are retained for audit purposes.
        """
        self._read()
        additions = {
            "Leeds": (-0.7362129430577615, 0.9532189955343428),
            "Sunderland": (-0.681340851032558, 0.991599164258899),
        }
        removed = {"West Ham", "Wolves"}
        expected = {"Arsenal", "Aston Villa", "Bournemouth", "Brentford",
                    "Brighton", "Chelsea", "Coventry", "Crystal Palace",
                    "Everton", "Fulham", "Hull", "Ipswich", "Leeds",
                    "Liverpool", "Man City", "Man United", "Newcastle",
                    "Nott'm Forest", "Sunderland", "Tottenham"}
        if set(self._teams) == expected:
            return self.summary().assign(Roster_Corrected=False)
        if self._config["season"] != "2026/27" or set(self._teams) != (
            expected - set(additions) | removed
        ):
            raise ValueError("This roster correction applies only to the archived 2026/27 setup.")
        reference = {"Q0": 0.9816907997148884, "h": 0.2828943874366233,
                     "beta": 0.3685819997716233, "R": 2.7192778604934182}
        if any(k not in self._config["bayes"] or not np.isclose(
            self._config["bayes"][k], v, rtol=0, atol=6e-7
        ) for k, v in reference.items()):
            raise ValueError("Saved Bayesian parameters differ from the archived final calibration.")
        city_prior = self._opening.set_index("Team").loc["Man City", "Prior_Mean"]
        if not np.isclose(city_prior, 3.026023571113532, rtol=0, atol=6e-7):
            raise ValueError("Opening means differ from the checked recursive prior.")
        if any(removed.intersection((r["HomeTeam"], r["AwayTeam"])) for r in self._results):
            raise ValueError("Recorded target-season games include removed teams; inspect those results first.")
        old_config = json.loads(_canonical(self._config))
        new_config = json.loads(_canonical(self._config))
        rows = [r for r in new_config["prior"] if r["Team"] not in removed]
        for team, (mean, multiplier) in additions.items():
            rows.append({"Team": team, "Prior_Mean": mean,
                         "Prior_Variance": new_config["bayes"]["Q0"] * multiplier})
        new_config["prior"] = sorted(rows, key=lambda r: r["Team"])
        backup = self.path.with_name(self.path.stem + ".before_roster_correction.json")
        if not backup.exists():
            _write_atomic(backup, self._saved)
        self._saved.setdefault("config_revisions", []).append({
            "changed_utc": datetime.now(timezone.utc).isoformat(),
            "reason": "Align 2026/27 EPL membership; recover archived 2025/26 terminal priors.",
            "previous_config": old_config,
        })
        self._saved["config"] = new_config
        _write_atomic(self.path, self._saved)
        return self.summary().assign(Roster_Corrected=True)

    def season_report(self, *, replay_missing=False, model=None):
        """Compare saved forecasts with results, one row per City EPL fixture.

        By default absent forecasts are explicitly marked No_saved_forecast.
        replay_missing=True reconstructs them using the frozen model and only
        entered results from strictly earlier dates. Those rows are labelled
        Replayed_now; they are neither written to the archive nor described as
        predictions actually issued before the match. Existing saved forecasts
        always take priority. The table also includes pending saved fixtures.
        Only the selected model is shown unless model is supplied explicitly.
        """
        self._read()
        model = model or self._config["selected_model"]
        if model not in self._config["score_models"]:
            raise ValueError(f"Unknown score model: {model}")
        city = self._config["city"]
        fixtures = {}
        for r in self._results:
            if city not in (r["HomeTeam"], r["AwayTeam"]):
                continue
            home = r["HomeTeam"] == city
            key = ("Home" if home else "Away", r["AwayTeam"] if home else r["HomeTeam"])
            fixtures[key] = {"Date": r["Date"], "GF": r["FTHG"] if home else r["FTAG"],
                             "GA": r["FTAG"] if home else r["FTHG"]}
        archives = {}
        for saved in self._saved["forecasts"]:
            match = saved["match"]
            if match["Model"] != model:
                continue
            key = (match["Venue"], match["Opponent_Key"])
            archives.setdefault(key, []).append(saved)
            fixtures.setdefault(key, {"Date": match["Date"], "GF": None, "GA": None})
        output = []
        for (venue, opponent), result in fixtures.items():
            day, gf, ga = result["Date"], result["GF"], result["GA"]
            top5, forecast = [], None
            source = "No_saved_forecast"
            candidates = sorted(archives.get((venue, opponent), []), key=lambda r: r["issued_utc"])
            if candidates:
                def issued_day(record):
                    return datetime.fromisoformat(record["issued_utc"]).astimezone(
                        ZoneInfo("Europe/London")
                    ).date().isoformat()
                earlier = [r for r in candidates if issued_day(r) < day]
                same_day = [r for r in candidates if issued_day(r) == day]
                chosen = (earlier[-1] if earlier else same_day[-1] if same_day else candidates[0])
                forecast, top5 = chosen["match"], chosen["top5"]
                source = ("Saved_before_match_date" if issued_day(chosen) < day
                          else "Saved_same_day_time_unverified" if issued_day(chosen) == day
                          else "Saved_replay")
            elif replay_missing and gf is not None:
                prior_results = [r for r in self._results if r["Date"] < day]
                mean, _ = self._replay(prior_results)
                delta = float(mean[self._index[city]] - mean[self._index[opponent]])
                spec = self._config["score_models"][model]
                features = {"Delta_S": delta, "Home": int(venue == "Home")}
                vector = np.array([features[k] for k in spec["features"]])
                rates = [float(np.exp(spec["targets"][target]["intercept"]
                         + vector @ np.asarray(spec["targets"][target]["coefficients"])))
                         for target in ("GF", "GA")]
                _, top5, _ = _distribution(*rates)
                forecast = {"Forecast_ID": None, "Date": day, "Lambda_City": rates[0],
                            "Lambda_Opp": rates[1], "N_Results_Used": len(prior_results)}
                source = "Replayed_now"
            actual = f"{gf}-{ga}" if gf is not None else "Pending"
            matching = [r["Rank"] for r in top5 if r["Score_City_Opp"] == actual]
            row = {"Date": day, "Venue": venue, "Opponent": opponent,
                   "Actual_Score": actual, "Prediction_Source": source,
                   "Forecast_ID": forecast["Forecast_ID"] if forecast else None,
                   "Forecast_For_Date": forecast["Date"] if forecast else None,
                   "N_Results_Used": forecast["N_Results_Used"] if forecast else None,
                   "Lambda_City": forecast["Lambda_City"] if forecast else None,
                   "Lambda_Opp": forecast["Lambda_Opp"] if forecast else None}
            for rank in range(1, 6):
                p = next((r for r in top5 if r["Rank"] == rank), None)
                row[f"Top{rank}"] = f"{p['Score_City_Opp']} ({p['Probability']:.2%})" if p else None
            row["Actual_Rank"] = (str(matching[0]) if matching else ">5") if top5 and gf is not None else None
            row["Top5_Hit"] = bool(matching) if top5 and gf is not None else None
            output.append(row)
        if not output:
            return pd.DataFrame(columns=["Date", "Venue", "Opponent", "Actual_Score", "Prediction_Source"])
        return pd.DataFrame(output).sort_values(["Date", "Opponent"]).reset_index(drop=True)
