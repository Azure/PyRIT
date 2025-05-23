# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Dict, Union
from pathlib import Path


def read_txt(file) -> List[Dict[str, str]]:
    return [{"prompt": line.strip()} for line in file.readlines()]


def write_txt(file, examples: List[Dict[str, str]]):
    file.write("\n".join([ex["prompt"] for ex in examples]))


def sanitize(text: str) -> str:
    return str(text).replace("<", "&lt;").replace(">", "&gt;")


def format_execution_time(seconds: float) -> str:
    seconds = int(round(seconds))
    mins, secs = divmod(seconds, 60)
    return f"{mins}m {secs}s" if mins else f"{secs}s"


def _render_report_html(
        title: str,
        description: str,
        results: list,
        threshold: float,
        execution_time: float,
        is_chat_eval: bool,
        strict_step_failures: bool
) -> str:
    passed_cases = 0
    total_cases = len(results)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset='utf-8'>
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; background: #f4f4f4; padding: 20px; }}
    .container {{ background: #fff; padding: 30px; border-radius: 10px; max-width: 1100px; margin: auto; }}
    h1 {{ text-align: center; color: #2c3e50; }}
    .summary {{ font-size: 1rem; text-align: center; color: #444; margin-bottom: 30px; }}
    details {{ border: 1px solid #ccc; border-radius: 6px; background: #eaf4fe; margin-bottom: 15px; }}
    summary {{ padding: 12px; font-weight: bold; cursor: pointer; color: #01579b; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
    th, td {{ padding: 12px; border-bottom: 1px solid #eee; vertical-align: top; }}
    th {{ background: #0277bd; color: #fff; }}
    .score-pass {{ color: green; font-weight: bold; }}
    .score-fail {{ color: red; font-weight: bold; }}
    .badge {{ display: inline-block; padding: 4px 10px; border-radius: 5px; font-weight: bold; }}
    .badge.pass {{ background: #c8e6c9; color: #1b5e20; }}
    .badge.fail {{ background: #ffcdd2; color: #b71c1c; }}
    .explanation {{ font-size: 0.9rem; margin-top: 6px; color: #555; }}
  </style>
</head>
<body>
<div class='container'>
  <h1>{title}</h1>
  <p class='overview'>{description}</p>
  <div class='summary'>
    Total Test Cases: {total_cases} |
    Passed: {{passed}} |
    Failed: {{failed}} |
    Execution Time: {format_execution_time(execution_time)}
  </div>
"""

    for idx, result in enumerate(results, start=1):
        # ✅ Objective fallback for both types
        if "objective" in result:
            objective = result["objective"]
        elif "prompt" in result:
            objective = result["prompt"]
        elif "conversation" in result:
            first_user = next((msg for msg in result["conversation"] if msg.get("role") == "user"), {})
            objective = first_user.get("message", "N/A")
        else:
            objective = "N/A"

        objective = sanitize(objective)

        if is_chat_eval:
            transcript = result.get("transcript", [])
            if not transcript and "conversation" in result:
                conv = result["conversation"]
                for i in range(0, len(conv), 2):
                    turn = {"turn_index": (i // 2) + 1, "pieces": []}
                    if i < len(conv):
                        turn["pieces"].append({
                            "role": conv[i].get("role", ""),
                            "converted_value": conv[i].get("message", ""),
                            "scores": conv[i].get("scores", [])
                        })
                    if i + 1 < len(conv):
                        turn["pieces"].append({
                            "role": conv[i + 1].get("role", ""),
                            "converted_value": conv[i + 1].get("message", ""),
                            "scores": conv[i + 1].get("scores", [])
                        })
                    transcript.append(turn)

            turns = result.get("aggregated_metrics", {}).get("total_turns", len(transcript))
            raw_score = result.get("aggregated_metrics", {}).get("final_score", 0.0)

            # Handle both boolean-like strings and numeric strings
            if isinstance(raw_score, str):
                if raw_score.lower() == "true":
                    final_score = True
                elif raw_score.lower() == "false":
                    final_score = False
                else:
                    try:
                        final_score = float(raw_score)
                    except ValueError:
                        final_score = 0.0
            else:
                final_score = raw_score

            score_values = []
            for turn in transcript:
                for piece in turn["pieces"]:
                    for score in piece.get("scores", []):
                        val = score.get("score", 0.0)
                        try:
                            score_values.append(float(val))
                        except:
                            score_values.append(1.0 if str(val).lower() == "true" else 0.0)

            if strict_step_failures:
                passed = all(s >= threshold for s in score_values)
            else:
                try:
                    passed = float(final_score) >= threshold
                except:
                    passed = str(final_score).lower() == "true"

        else:
            # Dataset report: single or multi-step
            if "conversation" in result:
                conv = result["conversation"]
                transcript = []
                for i in range(0, len(conv), 2):
                    turn = {"turn_index": (i // 2) + 1, "pieces": []}
                    if i < len(conv):
                        turn["pieces"].append({
                            "role": conv[i].get("role", ""),
                            "converted_value": conv[i].get("message", ""),
                            "scores": conv[i].get("scores", [])
                        })
                    if i + 1 < len(conv):
                        turn["pieces"].append({
                            "role": conv[i + 1].get("role", ""),
                            "converted_value": conv[i + 1].get("message", ""),
                            "scores": conv[i + 1].get("scores", [])
                        })
                    transcript.append(turn)
                turns = len(transcript)
            else:
                transcript = [{
                    "turn_index": 1,
                    "pieces": [
                        {"role": "user", "converted_value": result.get("prompt", "")},
                        {
                            "role": "assistant",
                            "converted_value": result.get("assistant_response", ""),
                            "scores": result.get("scores", [])
                        }
                    ]
                }]
                turns = 1

            score_values = []
            for turn in transcript:
                for piece in turn["pieces"]:
                    for score in piece.get("scores", []):
                        val = score.get("score_value", 0.0)
                        try:
                            score_values.append(float(val))
                        except:
                            score_values.append(1.0 if str(val).lower() == "true" else 0.0)

            if strict_step_failures:
                passed = all(s >= threshold for s in score_values)
            else:
                passed = min(score_values, default=0.0) >= threshold

            if turns == 1:
                final_score = max(score_values, default=0.0)
            else:
                final_score = min(score_values, default=0.0)

        if passed:
            passed_cases += 1

        badge = "pass" if passed else "fail"
        label = "Pass" if passed else "Fail"

        summary_parts = [
            f"Test Case {idx}: <strong>Objective:</strong> {objective}",
            f"<strong>Achieved:</strong> <span class='badge {badge}'>{label}</span>",
            f"<strong>Turns:</strong> {turns}"
        ]

        if not isinstance(final_score, bool):
            final_score_display = f"{final_score:.2f}" if isinstance(final_score, (int, float)) else "N/A"
            summary_parts.append(f"<strong>Final Score:</strong> {final_score_display}")

        html += f"""
        <details>
          <summary>{' | '.join(summary_parts)}</summary>
          <table>
        """
        # Update table header and row ordering for dataset report: input, assistant response, expected output, score.
        if not is_chat_eval:
            html += "<thead><tr><th>User</th><th>Assistant</th><th>Expected Output</th><th>Score</th></tr></thead>"
        else:
            html += "<thead><tr><th>User</th><th>Assistant</th><th>Score</th></tr></thead>"
        html += "<tbody>"

        for turn in transcript:
            user_piece = next((p for p in turn["pieces"] if p["role"] == "user"), {"converted_value": ""})
            assistant_piece = next((p for p in turn["pieces"] if p["role"] == "assistant"), {"converted_value": "", "scores": []})

            user_text = sanitize(user_piece["converted_value"])
            assistant_text = sanitize(assistant_piece["converted_value"])

            scores_html = ""
            for score in assistant_piece.get("scores", []):
                val = score.get("score", score.get("score_value", None))
                rationale = sanitize(score.get("rationale", score.get("score_rationale", "")))
                try:
                    val = float(val)
                except:
                    val = True if str(val).lower() == "true" else False
                cls = "score-pass" if val >= threshold else "score-fail"
                if isinstance(val, bool):
                    val_display = "✔️ True" if val else "❌ False"
                else:
                    val_display = f"{val:.2f}"
                scores_html += f"<div><strong class='{cls}'>{val_display}</strong><div class='explanation'>{rationale}</div></div>"

            if not is_chat_eval:
                # Retrieve expected output from the first score object.
                if assistant_piece.get("scores") and len(assistant_piece["scores"]) > 0:
                    assistant_expected = sanitize(assistant_piece["scores"][0].get("expected_output", "N/A"))
                else:
                    assistant_expected = "N/A"
                # New order: User, Assistant, Expected Output, Score.
                html += f"<tr><td>{user_text}</td><td>{assistant_text}</td><td>{assistant_expected}</td><td>{scores_html}</td></tr>"
            else:
                html += f"<tr><td>{user_text}</td><td>{assistant_text}</td><td>{scores_html}</td></tr>"

        html += "</tbody></table></details>"

    html = html.replace("{passed}", str(passed_cases))
    html = html.replace("{failed}", str(total_cases - passed_cases))
    html += "</div></body></html>"
    return html

def generate_simulation_report(
        results: list,
        threshold: float = 0.8,
        title: str = "Comprehensive Simulation Report",
        description: str = "",
        execution_time: float = 0.0,
        save_path: Union[str, Path] = "simulation_report.html"
):
    html = _render_report_html(
        title=title,
        description=description,
        results=results,
        threshold=threshold,
        execution_time=execution_time,
        is_chat_eval=True,
        strict_step_failures=False
    )

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✅ Simulation report saved to: {save_path}")


def generate_dataset_report(
        results: list,
        threshold: float = 0.5,
        title: str = "Comprehensive Dataset Report",
        description: str = "",
        execution_time: float = 0.0,
        save_path: Union[str, Path] = "dataset_report.html"
):
    html = _render_report_html(
        title=title,
        description=description,
        results=results,
        threshold=threshold,
        execution_time=execution_time,
        is_chat_eval=False,
        strict_step_failures=True
    )

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✅ Dataset report saved to: {save_path}")
