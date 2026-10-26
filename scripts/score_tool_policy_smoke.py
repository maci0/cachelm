#!/usr/bin/env python3
"""Smoke: print reward scores for a few hand examples (ll3m.rewards)."""

from __future__ import annotations

from ll3m.rewards import score_calc_episode, score_completion, score_nocall_episode


def main() -> None:
    cases = [
        ("nocall_ok", score_nocall_episode("Hello! How can I help?")),
        (
            "nocall_spam",
            score_nocall_episode("Hi <call>lookup capital france</call>"),
        ),
        (
            "calc_ok",
            score_calc_episode(
                "<call>calc add 5 5</call>\nThe answer is 10.",
                op="add",
                a="5",
                b="5",
                result="10",
            ),
        ),
        (
            "calc_miss",
            score_calc_episode("ten", op="add", a="5", b="5", result="10"),
        ),
        (
            "unknown",
            score_completion("<call>the weather</call>", expect_tool=None),
        ),
    ]
    for name, r in cases:
        print(f"{name:12} total={r.total:+.2f}  {r.as_dict()}")


if __name__ == "__main__":
    main()
