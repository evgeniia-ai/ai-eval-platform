"""Admin dashboard (Streamlit).

Run with:  streamlit run src/app.py

Pages:
  - Overview        : all evaluated calls, scores, filters, run-pipeline control
  - Call detail     : drill into one evaluation (dimensions, evidence, transcript, vs ground truth)
  - Rep summary     : per-rep averages, score trend, recurring weaknesses, coaching generation
  - Data generation : create synthetic transcripts to augment the seed set
"""

import json
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# Make `src` importable when launched via `streamlit run src/app.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.express as px
import streamlit as st
import traceback  # to print the full error stack to the UI instead of swallowing it

from src import coaching, config, data_gen, ingest, storage
from src.qa_engine import QAError, evaluate
from src.models import Transcript
from src.rubric import RUBRIC, DIMENSION_KEYS
from src.test_sets import SMOKE_SET, REGRESSION_SET

_NAME = {d.key: d.name for d in RUBRIC}
_MODEL_SHORT = {
    "claude-opus-4-8": "Opus",
    "claude-sonnet-4-6": "Sonnet",
    "claude-haiku-4-5-20251001": "Haiku",
}
_MOUNTAIN = ZoneInfo("America/Denver")


def _utc_to_mountain(utc_str: str) -> str:
    """Convert a stored UTC timestamp string to Mountain Time (DST-aware) for display."""
    if not utc_str:
        return ""
    try:
        dt = datetime.fromisoformat(utc_str.replace(" ", "T")).replace(tzinfo=timezone.utc)
        return dt.astimezone(_MOUNTAIN).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return utc_str[:19].replace("T", " ")


_JUDGE_MODELS = [
    ("Opus 4.8 — most capable (default)", "claude-opus-4-8"),
    ("Sonnet 4.6 — balanced", "claude-sonnet-4-6"),
    ("Haiku 4.5 — fastest / cheapest", "claude-haiku-4-5-20251001"),
]

st.set_page_config(page_title="HealthBridge Call QA", page_icon="📞", layout="wide")
storage.init()


def _format_run_label(run: dict) -> str:
    if not run.get("model"):
        return "legacy · model unknown"
    raw_ts = run.get("created_at")
    if raw_ts:
        ts = _utc_to_mountain(raw_ts)
    else:
        # run_label is YYYYMMDD-HHMMSS (UTC) for programmatic runs
        label = run.get("run_label", "")
        try:
            iso = f"{label[:4]}-{label[4:6]}-{label[6:8]}T{label[9:11]}:{label[11:13]}:{label[13:15]}"
            ts = _utc_to_mountain(iso)
        except Exception:
            ts = "timestamp unknown"
    model_short = _MODEL_SHORT.get(run["model"], run["model"])
    version = f"v{run['rubric_version']}" if run.get("rubric_version") else "v?"
    return f"{ts} · {model_short} · {version} · #{run['run_id']}"


def _score_color(v: float) -> str:
    if v >= 4.0:
        return "🟢"
    if v >= 3.0:
        return "🟡"
    return "🔴"


# ---------------------------------------------------------------------------
# Sidebar nav
# ---------------------------------------------------------------------------
st.sidebar.title("📞 HealthBridge")
st.sidebar.caption("Call QA & Self-Improvement")
st.sidebar.metric("Evaluations stored", storage.count())
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Call detail", "Rep summary", "Data generation", "Test runs", "Review Queue"],
)
if page not in ("Test runs", "Review Queue"):
    _chosen_model_label = st.sidebar.selectbox(
        "Judge model",
        [label for label, _ in _JUDGE_MODELS],
        index=next((i for i, (_, mid) in enumerate(_JUDGE_MODELS) if mid == config.model()), 0),
    )
    os.environ["ANTHROPIC_MODEL"] = dict(_JUDGE_MODELS)[_chosen_model_label]
    st.sidebar.caption(f"`{dict(_JUDGE_MODELS)[_chosen_model_label]}`")


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
def page_overview():
    st.header("All evaluated calls")
    rows = storage.all_evaluations()

    if not rows:
        st.info("No evaluations yet. Load and evaluate transcripts to get started.")
        if st.button("▶ Evaluate seed + generated transcripts", type="primary"):
            bar = st.progress(0.0, text="Starting…")
            def prog(done, total, msg):
                bar.progress(done / total, text=f"[{done}/{total}] {msg}")
            try:
                ok, fail = ingest.run(progress=prog)
            except Exception as e:
                # Hard crash inside the pipeline: show it and STOP (no rerun) so it stays visible.
                st.error(f"{type(e).__name__}: {e}")
                st.code(traceback.format_exc())
                st.stop()
            if ok == 0:
                # Every evaluation failed. Do NOT rerun — that would wipe this message
                # and bounce you straight back to the empty state.
                st.error(f"All {fail} evaluations failed — nothing was stored. "
                         f"Check the terminal for the per-call error.")
                st.stop()
            st.success(f"Evaluated {ok} calls ({fail} failed).")
            st.rerun()
        return

    df = pd.DataFrame(rows)
    display_cols = ["call_id", "rep_id", "call_type", "overall_score"] + DIMENSION_KEYS + ["ground_truth_overall"]
    df_view = df[display_cols].rename(columns={**_NAME, "overall_score": "overall", "ground_truth_overall": "ground_truth"})

    c1, c2, c3 = st.columns(3)
    c1.metric("Calls evaluated", len(df))
    c2.metric("Mean overall score", round(df["overall_score"].mean(), 2))
    # MAE vs human ground truth — seed calls only (exclude HB-SYNTH, whose GT
    # is model-generated and circular: the same model family labels its own output).
    seed_gt = df[
        ~df["call_id"].str.startswith("HB-SYNTH") & df["ground_truth_overall"].notna()
    ]
    if len(seed_gt):
        mae = round((seed_gt["overall_score"] - seed_gt["ground_truth_overall"]).abs().mean(), 2)
        c3.metric("MAE vs human ground truth", mae, help="Lower is better — how close the engine is to human reviewers.")
        st.caption(f"MAE computed against human (seed) labels only — {len(seed_gt)} call(s). Synthetic calls excluded.")
    else:
        c3.metric("MAE vs human ground truth", "—")
        st.caption("No human-labeled seed calls in the database yet.")

    types = st.multiselect("Filter by call type", sorted(df["call_type"].unique()))
    if types:
        df_view = df_view[df["call_type"].isin(types)]

    st.dataframe(df_view, use_container_width=True, hide_index=True)

    st.subheader("Average score by dimension")
    avg = df[DIMENSION_KEYS].mean().rename(index=_NAME).round(2)
    fig = px.bar(avg, labels={"value": "avg score", "index": "dimension"}, range_y=[0, 5])
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("MAE by dimension (vs human ground truth)")
    dim_errors: dict[str, list[float]] = {k: [] for k in DIMENSION_KEYS}
    for row in rows:
        if row["call_id"].startswith("HB-SYNTH"):
            continue  # exclude circular synthetic GT
        if not row.get("transcript_json"):
            continue
        try:
            t = Transcript.model_validate_json(row["transcript_json"])
        except Exception:
            continue
        if not t.ground_truth_qa:
            continue
        gt_dim = t.ground_truth_qa.dimension_scores
        for key in DIMENSION_KEYS:
            if key in gt_dim and row.get(key) is not None:
                dim_errors[key].append(abs(row[key] - gt_dim[key]))
    if any(dim_errors[k] for k in DIMENSION_KEYS):
        mae_series = pd.Series(
            {_NAME[k]: round(sum(v) / len(v), 2) for k, v in dim_errors.items() if v}
        )
        fig2 = px.bar(mae_series, labels={"value": "mean absolute error", "index": ""}, range_y=[0, 4])
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Seed (human-labeled) calls only — synthetic calls excluded.")
    else:
        st.caption("No human-labeled calls with ground-truth dimension scores in the database yet.")

with st.expander("Re-run evaluation pipeline"):
        if st.button("▶ Re-evaluate all transcripts"):
            bar = st.progress(0.0, text="Starting…")
            def prog(done, total, msg):
                bar.progress(done / total, text=f"[{done}/{total}] {msg}")
            try:
                ok, fail = ingest.run(progress=prog)
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
                st.code(traceback.format_exc())
                st.stop()
            if fail:
                # Some calls failed but don't crash — surface the count instead of hiding it.
                st.warning(f"{fail} call(s) failed during evaluation — check the terminal.")
            if ok == 0:
                st.error("Nothing was evaluated successfully. See terminal for details.")
                st.stop()
            st.success(f"Evaluated {ok} calls ({fail} failed).")
            st.rerun()

        st.divider()
        st.markdown("**Evaluate a single call**")
        _all_transcripts = ingest.load_transcripts(include_generated=True)
        _sel_id = st.selectbox(
            "Select call",
            [t.call_id for t in _all_transcripts],
            key="single_call_id",
        )
        if st.button("▶ Evaluate selected call only"):
            _t = next(t for t in _all_transcripts if t.call_id == _sel_id)
            try:
                _ev = evaluate(_t)
                storage.save(_ev, transcript=_t)
                st.success(f"{_t.call_id} → overall {_ev.overall_score}")
                st.rerun()
            except QAError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
                st.code(traceback.format_exc())


# ---------------------------------------------------------------------------
# Call detail
# ---------------------------------------------------------------------------
def page_call_detail():
    st.header("Call detail")
    rows = storage.all_evaluations()
    if not rows:
        st.info("No evaluations yet — see the Overview page.")
        return

    call_id = st.selectbox("Select a call", [r["call_id"] for r in rows])

    runs = storage.runs_for_call(call_id)

    def _ts(run: dict) -> str:
        raw = run.get("created_at")
        if raw:
            return _utc_to_mountain(raw)
        label = run.get("run_label", "")
        try:
            iso = f"{label[:4]}-{label[4:6]}-{label[6:8]}T{label[9:11]}:{label[11:13]}:{label[13:15]}"
            return _utc_to_mountain(iso)
        except Exception:
            return "unknown"

    runs_df = pd.DataFrame([
        {
            "timestamp": _ts(r),
            "model": _MODEL_SHORT.get(r.get("model") or "", r.get("model") or "unknown"),
            "rubric": r.get("rubric_version") or "—",
            "overall": r.get("overall_score"),
            "greeting": r.get("greeting_identity_verification"),
            "empathy": r.get("empathy_tone"),
            "accuracy": r.get("accuracy_completeness"),
            "protocol": r.get("protocol_adherence"),
            "closing": r.get("closing_next_steps"),
        }
        for r in runs
    ])
    st.subheader("Run history")
    st.dataframe(runs_df, use_container_width=True, hide_index=True)

    selected_run = st.selectbox(
        "Select run",
        runs,
        format_func=_format_run_label,
        index=0,
    )
    rec = storage.get_evaluation(call_id, run_id=selected_run["run_id"])
    detail = rec["detail"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Overall score", f"{rec['overall_score']} {_score_color(rec['overall_score'])}")
    c2.metric("Rep", rec["rep_id"])
    c3.metric("Call type", rec["call_type"])
    if rec.get("ground_truth_overall") is not None:
        st.caption(f"Human ground-truth overall: **{rec['ground_truth_overall']}**")

    st.subheader("Summary")
    st.write(detail.summary)
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Top strengths**")
        for s in detail.top_strengths:
            st.markdown(f"- {s}")
    with cols[1]:
        st.markdown("**Top improvements**")
        for s in detail.top_improvements:
            st.markdown(f"- {s}")

    st.subheader("Dimension scores")
    for d in RUBRIC:
        de = getattr(detail, d.key)
        with st.expander(f"{_score_color(de.score)} {d.name} — {de.score}/5"):
            st.markdown(f"**Reasoning:** {de.reasoning}")
            st.markdown(f"**Evidence:** _{de.evidence}_")
            st.markdown(f"**Suggestion:** {de.suggestion}")

    if rec.get("transcript"):
        with st.expander("Full transcript"):
            for u in rec["transcript"].transcript:
                who = "🧑‍⚕️ Rep" if u.speaker == "rep" else "🧑 Patient"
                st.markdown(f"`{u.timestamp}` **{who}:** {u.text}")


# ---------------------------------------------------------------------------
# Rep summary
# ---------------------------------------------------------------------------
def page_rep_summary():
    st.header("Representative summary")
    reps = storage.rep_ids()
    if not reps:
        st.info("No evaluations yet — see the Overview page.")
        return

    rep_id = st.selectbox("Select a representative", reps)
    rows = storage.evaluations_for_rep(rep_id)
    avgs = storage.rep_dimension_averages(rep_id)

    c1, c2 = st.columns(2)
    c1.metric("Calls evaluated", len(rows))
    overall_avg = round(sum(r["overall_score"] for r in rows) / len(rows), 2)
    c2.metric("Average overall", f"{overall_avg} {_score_color(overall_avg)}")

    st.subheader("Average by dimension")
    avg_df = pd.Series({_NAME[k]: v for k, v in avgs.items()})
    fig = px.bar(avg_df, range_y=[0, 5], labels={"value": "avg", "index": ""})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Score trend across calls")
    trend = pd.DataFrame([
        {"call_id": r["call_id"], "overall": r["overall_score"], **{_NAME[k]: r[k] for k in DIMENSION_KEYS}}
        for r in rows
    ])
    fig2 = px.line(trend, x="call_id", y="overall", markers=True, range_y=[0, 5])
    st.plotly_chart(fig2, use_container_width=True)

    weak = coaching.recurring_weaknesses(rep_id)
    if weak:
        st.warning("Recurring weaknesses: " + ", ".join(f"{_NAME[k]} ({v})" for k, v in weak))
    else:
        st.success("No dimension averages below 3.5 — solid all around.")

    st.subheader("Coaching")
    if st.button("✨ Generate coaching summary", type="primary"):
        with st.spinner("Generating personalized coaching…"):
            try:
                summary = coaching.generate_coaching(rep_id)
                st.session_state[f"coaching_{rep_id}"] = summary
            except QAError as e:
                st.error(str(e))

    summary = st.session_state.get(f"coaching_{rep_id}")
    if summary:
        st.markdown(f"### {summary.headline}")
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Strengths**")
            for s in summary.strengths:
                st.markdown(f"- {s}")
        with cols[1]:
            st.markdown("**Focus areas**")
            for s in summary.focus_areas:
                st.markdown(f"- {s}")
        st.markdown("**Coaching plan**")
        st.write(summary.coaching_plan)
        st.markdown("**Suggested actions**")
        for a in summary.suggested_actions:
            st.markdown(f"- {a}")
        with st.expander("🔁 Feedback loop: coaching as a live-assist directive"):
            st.caption(
                "This directive can be injected into the system prompt of a real-time "
                "call-assist agent for this rep, closing the improvement loop."
            )
            st.code(coaching.coaching_directive_for_prompt(summary), language="text")


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def page_data_gen():
    st.header("Synthetic data generation")
    st.write(
        "Augment the 7 seed transcripts with synthetic calls across a scenario matrix "
        "(call types × quality levels × edge cases). Generated calls include ground-truth "
        "QA scores, so they double as an eval set for the engine."
    )
    n = st.slider("How many to generate", 3, len(data_gen.SPECS), 6)
    if st.button("▶ Generate transcripts", type="primary"):
        bar = st.progress(0.0, text="Starting…")
        def prog(done, total, msg):
            bar.progress(done / total, text=f"[{done}/{total}] {msg}")
        calls = data_gen.generate(n=n, progress=prog)
        path = data_gen.save(calls)
        st.success(f"Generated {len(calls)} transcripts → `{path}`")
        st.caption("Now go to Overview and run the evaluation pipeline to score them.")
        st.dataframe(
            pd.DataFrame([
                {"call_id": c["call_id"], "type": c["call_type"], "scenario": c["_scenario"],
                 "gt_overall": c["ground_truth_qa"]["overall_score"]}
                for c in calls
            ]),
            use_container_width=True, hide_index=True,
        )


# ---------------------------------------------------------------------------
# Test runs
# ---------------------------------------------------------------------------
def page_test_runs():
    st.header("Test runs")

    all_transcripts = ingest.load_transcripts(include_generated=True)
    by_type: dict[str, list] = {}
    for t in all_transcripts:
        by_type.setdefault(t.call_type, []).append(t)

    col_set, col_model = st.columns([1, 1])
    with col_set:
        # Default-set radio — switching it resets the checkbox state.
        set_choice = st.radio(
            "Default set",
            ["Smoke", "Regression"],
            horizontal=True,
            key="tr_set_choice",
        )
    with col_model:
        tr_model_label = st.selectbox(
            "Judge model",
            [label for label, _ in _JUDGE_MODELS],
            index=next(
                (i for i, (_, mid) in enumerate(_JUDGE_MODELS) if mid == config.model()), 0
            ),
            key="tr_model_label",
        )
    tr_model_id = dict(_JUDGE_MODELS)[tr_model_label]
    os.environ["ANTHROPIC_MODEL"] = tr_model_id

    default_ids = SMOKE_SET if set_choice == "Smoke" else REGRESSION_SET

    if st.session_state.get("tr_prev_set_choice") != set_choice:
        for t in all_transcripts:
            st.session_state[f"tr_cb_{t.call_id}"] = t.call_id in default_ids
        st.session_state["tr_prev_set_choice"] = set_choice

    # Checkboxes grouped by call_type — state lives in st.session_state.
    selected: list[str] = []
    for call_type in sorted(by_type):
        st.markdown(f"**{call_type}**")
        for t in sorted(by_type[call_type], key=lambda x: x.call_id):
            cb_key = f"tr_cb_{t.call_id}"
            if cb_key not in st.session_state:
                st.session_state[cb_key] = t.call_id in default_ids
            if st.checkbox(t.call_id, key=cb_key):
                selected.append(t.call_id)

    n = len(selected)
    model_short = _MODEL_SHORT.get(tr_model_id, tr_model_id)
    st.caption(f"{n} call{'s' if n != 1 else ''} selected · model: {model_short} · ~{n} API call{'s' if n != 1 else ''}")

    if st.button("▶ Run selected", type="primary", disabled=(n == 0)):
        transcript_map = {t.call_id: t for t in all_transcripts}
        ok_scores: list[float] = []
        failed_msgs: list[str] = []       # "call_id: error" for UI display
        failed_ids: list[str] = []        # bare call_ids for storage
        skipped_ids: list[str] = []       # bare call_ids for storage
        gt_pairs: list[tuple[float, float]] = []

        bar = st.progress(0.0, text="Starting…")
        try:
            for i, cid in enumerate(selected):
                bar.progress(i / n, text=f"[{i}/{n}] {cid}…")
                t = transcript_map.get(cid)
                if t is None:
                    skipped_ids.append(cid)
                    continue
                try:
                    ev = evaluate(t)
                    storage.save(ev, transcript=t)
                    ok_scores.append(ev.overall_score)
                    if t.ground_truth_qa:
                        gt_pairs.append((ev.overall_score, t.ground_truth_qa.overall_score))
                except QAError as e:
                    failed_ids.append(cid)
                    failed_msgs.append(f"{cid}: {e}")
                except Exception as e:
                    failed_ids.append(cid)
                    failed_msgs.append(f"{cid}: {type(e).__name__}: {e}")
            bar.progress(1.0, text=f"Done — {len(ok_scores)} evaluated.")
        finally:
            storage.save_suite_run(
                suite_name=set_choice,
                selected_call_ids=selected,
                failed_call_ids=failed_ids,
                skipped_call_ids=skipped_ids,
                ok_scores=ok_scores,
                gt_pairs=gt_pairs,
                model=tr_model_id,
            )

        st.subheader("Run summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Evaluated", len(ok_scores))
        c2.metric("Failed", len(failed_ids))
        c3.metric("Skipped", len(skipped_ids))
        if ok_scores:
            mean_score = round(sum(ok_scores) / len(ok_scores), 2)
            st.metric("Mean overall score", mean_score)
        if gt_pairs:
            mae = round(sum(abs(pred - gt) for pred, gt in gt_pairs) / len(gt_pairs), 2)
            st.metric("MAE vs ground truth", mae, help="Lower is better")
        if failed_msgs:
            st.error("Failures:\n" + "\n".join(f"- {f}" for f in failed_msgs))
        if skipped_ids:
            st.warning("Skipped (no transcript on disk):\n" + "\n".join(f"- {s}" for s in skipped_ids))

    # -----------------------------------------------------------------------
    # Suite run history — always rendered, not only post-run
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Suite run history")

    suite_rows = storage.all_suite_runs()
    if not suite_rows:
        st.caption("No suite runs yet — run a set above to start tracking history.")
    else:
        def _suite_table(rows: list[dict]) -> pd.DataFrame:
            return pd.DataFrame([{
                "timestamp": _utc_to_mountain(r.get("created_at", "")),
                "model": _MODEL_SHORT.get(r["model"], r["model"]),
                "n_calls": r["n_calls"],
                "ok / fail / skip": f"{r['n_ok']} / {r['n_failed']} / {r['n_skipped']}",
                "mean overall": r["mean_overall"],
                "MAE vs GT": r["mae_vs_gt"],
            } for r in rows])

        smoke_rows = [r for r in suite_rows if r["suite_name"] == "Smoke"]
        reg_rows   = [r for r in suite_rows if r["suite_name"] == "Regression"]

        tab_smoke, tab_reg = st.tabs(["Smoke", "Regression"])
        for tab, rows, label in [
            (tab_smoke, smoke_rows, "Smoke"),
            (tab_reg,   reg_rows,   "Regression"),
        ]:
            with tab:
                if not rows:
                    st.caption(f"No {label} runs yet.")
                else:
                    st.dataframe(_suite_table(rows), use_container_width=True, hide_index=True)
                    if any(r["n_failed"] > 0 or r["n_skipped"] > 0 for r in rows):
                        with st.expander("Failed & skipped call details"):
                            for r in rows:
                                if r["n_failed"] == 0 and r["n_skipped"] == 0:
                                    continue
                                ts = _utc_to_mountain(r.get("created_at", ""))
                                st.markdown(f"**Run #{r['suite_run_id']} — {ts}**")
                                failed_ids  = json.loads(r["failed_call_ids"])
                                skipped_ids = json.loads(r["skipped_call_ids"])
                                if failed_ids:
                                    st.markdown("Failed: " + ", ".join(f"`{c}`" for c in failed_ids))
                                if skipped_ids:
                                    st.markdown("Skipped: " + ", ".join(f"`{c}`" for c in skipped_ids))

        # Chart: MAE by model — one bar per model, averaged across runs of that suite
        chart_rows = [r for r in suite_rows if r["mae_vs_gt"] is not None]
        if chart_rows:
            st.subheader("MAE vs ground truth by judge model")
            col_s, col_r = st.columns(2)
            for col, suite_label in [(col_s, "Smoke"), (col_r, "Regression")]:
                sub = [r for r in chart_rows if r["suite_name"] == suite_label]
                with col:
                    st.markdown(f"**{suite_label}**")
                    if not sub:
                        st.caption("No runs with MAE yet.")
                    else:
                        # Average MAE per model; label includes run count
                        from collections import defaultdict
                        mae_by_model: dict[str, list[float]] = defaultdict(list)
                        for r in sub:
                            short = _MODEL_SHORT.get(r["model"], r["model"])
                            mae_by_model[short].append(r["mae_vs_gt"])
                        bar_data = pd.DataFrame([
                            {
                                "model": f"{m} (n={len(vs)})" if len(vs) > 1 else m,
                                "MAE vs GT": round(sum(vs) / len(vs), 2),
                                "_sort": list(_MODEL_SHORT.values()).index(m)
                                         if m in _MODEL_SHORT.values() else 99,
                            }
                            for m, vs in mae_by_model.items()
                        ]).sort_values("_sort").drop(columns=["_sort"])
                        fig = px.bar(
                            bar_data, x="model", y="MAE vs GT",
                            range_y=[0, 4],
                            text="MAE vs GT",
                            color="model",
                            color_discrete_map={
                                k: v for k, v in zip(
                                    bar_data["model"],
                                    px.colors.qualitative.Plotly,
                                )
                            },
                        )
                        fig.update_traces(textposition="outside", texttemplate="%{text:.2f}")
                        fig.update_layout(showlegend=False, xaxis_title=None)
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("Lower MAE = closer to human labels = better.")


# ---------------------------------------------------------------------------
# Review Queue
# ---------------------------------------------------------------------------
def page_review_queue():
    from collections import Counter
    st.header("Human Review Queue")

    all_rows = storage.all_reviews()
    if not all_rows:
        st.info(
            "No reviews yet — evaluations that trip a quality or safety trigger "
            "will appear here automatically after the next pipeline run."
        )
        return

    counts = Counter(r["status"] for r in all_rows)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pending", counts.get("Pending", 0))
    c2.metric("Confirmed Issue", counts.get("Confirmed Issue", 0))
    c3.metric("False Alarm", counts.get("False Alarm", 0))
    c4.metric("Needs Rubric Update", counts.get("Needs Rubric Update", 0))

    # ------------------------------------------------------------------
    # Pending
    # ------------------------------------------------------------------
    pending = [r for r in all_rows if r["status"] == "Pending"]
    st.subheader(f"Pending ({len(pending)})")

    if not pending:
        st.success("Nothing pending — all reviews resolved.")
    else:
        for row in pending:
            ev = storage.get_evaluation(row["call_id"], run_id=row["run_id"])
            score = ev["overall_score"] if ev else None
            call_type = ev["call_type"] if ev else "unknown"
            score_label = f"overall {score} {_score_color(score)}" if score is not None else "score unknown"

            with st.expander(f"{row['call_id']}  ·  {score_label}  ·  {row['reason']}"):
                mc1, mc2, mc3 = st.columns(3)
                mc1.markdown(f"**Call type:** {call_type}")
                mc2.markdown(f"**Run ID:** #{row['run_id']}")
                mc3.markdown(f"**Queued:** {_utc_to_mountain(row['created_at'])}")

                if ev:
                    dim_df = pd.DataFrame([
                        {"dimension": _NAME.get(k, k), "score": ev[k]}
                        for k in DIMENSION_KEYS if ev.get(k) is not None
                    ])
                    st.dataframe(
                        dim_df,
                        hide_index=True,
                        use_container_width=False,
                        column_config={
                            "dimension": st.column_config.TextColumn(width="medium"),
                            "score": st.column_config.NumberColumn(width="small"),
                        },
                    )

                # Judge context
                detail = ev.get("detail") if ev else None
                if detail:
                    st.markdown(f"**Judge summary:** {detail.summary}")
                    ctx1, ctx2 = st.columns(2)
                    with ctx1:
                        st.markdown("**Strengths**")
                        for s in detail.top_strengths:
                            st.markdown(f"- {s}")
                    with ctx2:
                        st.markdown("**Top improvements**")
                        for s in detail.top_improvements:
                            st.markdown(f"- {s}")
                elif ev:
                    st.caption("Judge reasoning not available for this run.")

                # Full transcript (collapsed by default)
                transcript = ev.get("transcript") if ev else None
                if transcript:
                    with st.expander("Show full transcript"):
                        for u in transcript.transcript:
                            who = "🧑‍⚕️ Rep" if u.speaker == "rep" else "🧑 Patient"
                            st.markdown(f"`{u.timestamp}` **{who}:** {u.text}")
                elif ev:
                    st.caption("Transcript not stored for this run.")

                new_status = st.selectbox(
                    "Set status",
                    ["Confirmed Issue", "False Alarm", "Needs Rubric Update"],
                    key=f"rq_status_{row['review_id']}",
                )
                note = st.text_area(
                    "Reviewer note (optional)",
                    value=row.get("reviewer_note") or "",
                    key=f"rq_note_{row['review_id']}",
                    placeholder="Explain the decision or add context",
                )
                if st.button("Save", key=f"rq_save_{row['review_id']}"):
                    storage.update_review_status(row["review_id"], new_status, note or None)
                    st.success(f"Marked as **{new_status}**.")
                    st.rerun()

    # ------------------------------------------------------------------
    # Resolved
    # ------------------------------------------------------------------
    _ALL_STATUSES = ["Pending", "Confirmed Issue", "False Alarm", "Needs Rubric Update"]

    resolved = [r for r in all_rows if r["status"] != "Pending"]
    if resolved:
        st.subheader(f"Resolved ({len(resolved)})")

        # 1. Compact overview table — read-only, scan at a glance.
        res_df = pd.DataFrame([{
            "call_id": r["call_id"],
            "status": r["status"],
            "reasons": r["reason"],
            "reviewer_note": r.get("reviewer_note") or "",
            "updated": _utc_to_mountain(r.get("updated_at") or ""),
        } for r in resolved])
        st.dataframe(res_df, use_container_width=True, hide_index=True)

        # 2. Edit-on-demand — pick one to open its editor.
        edit_options = ["— none —"] + [
            f"{r['call_id']}  [{r['status']}]" for r in resolved
        ]
        edit_choice = st.selectbox(
            "Edit a resolved review",
            edit_options,
            index=0,
            key="rq_res_edit_select",
        )

        if edit_choice != "— none —":
            chosen_idx = edit_options.index(edit_choice) - 1  # offset for "— none —"
            row = resolved[chosen_idx]
            ev = storage.get_evaluation(row["call_id"], run_id=row["run_id"])
            call_type = ev["call_type"] if ev else "unknown"

            st.divider()
            ec1, ec2, ec3 = st.columns(3)
            ec1.markdown(f"**Call type:** {call_type}")
            ec2.markdown(f"**Run ID:** #{row['run_id']}")
            ec3.markdown(f"**Updated:** {_utc_to_mountain(row.get('updated_at') or '')}")

            if ev:
                dim_df = pd.DataFrame([
                    {"dimension": _NAME.get(k, k), "score": ev[k]}
                    for k in DIMENSION_KEYS if ev.get(k) is not None
                ])
                st.dataframe(
                    dim_df,
                    hide_index=True,
                    use_container_width=False,
                    column_config={
                        "dimension": st.column_config.TextColumn(width="medium"),
                        "score": st.column_config.NumberColumn(width="small"),
                    },
                )

            detail = ev.get("detail") if ev else None
            if detail:
                st.markdown(f"**Judge summary:** {detail.summary}")
                ctx1, ctx2 = st.columns(2)
                with ctx1:
                    st.markdown("**Strengths**")
                    for s in detail.top_strengths:
                        st.markdown(f"- {s}")
                with ctx2:
                    st.markdown("**Top improvements**")
                    for s in detail.top_improvements:
                        st.markdown(f"- {s}")
            elif ev:
                st.caption("Judge reasoning not available for this run.")

            transcript = ev.get("transcript") if ev else None
            if transcript:
                with st.expander("Show full transcript"):
                    for u in transcript.transcript:
                        who = "🧑‍⚕️ Rep" if u.speaker == "rep" else "🧑 Patient"
                        st.markdown(f"`{u.timestamp}` **{who}:** {u.text}")
            elif ev:
                st.caption("Transcript not stored for this run.")

            cur_idx = _ALL_STATUSES.index(row["status"]) if row["status"] in _ALL_STATUSES else 0
            new_status = st.selectbox(
                "Status",
                _ALL_STATUSES,
                index=cur_idx,
                key="rq_res_edit_status",
            )
            note = st.text_area(
                "Reviewer note (optional)",
                value=row.get("reviewer_note") or "",
                key="rq_res_edit_note",
                placeholder="Explain the decision or add context",
            )
            if st.button("Save", key="rq_res_edit_save"):
                storage.update_review_status(row["review_id"], new_status, note or None)
                st.success(f"Saved as **{new_status}**.")
                st.rerun()


PAGES = {
    "Overview": page_overview,
    "Call detail": page_call_detail,
    "Rep summary": page_rep_summary,
    "Data generation": page_data_gen,
    "Test runs": page_test_runs,
    "Review Queue": page_review_queue,
}
PAGES[page]()
