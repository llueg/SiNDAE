# Contributing to SiNDAE

This is a short guide to submitting changes. The aim is that every pull request is
easy to review and safe to merge. If something here is unclear or out of date, say
so in your PR.

## Branches and scope

- Work on a branch, never directly on `main`. Name it after what it does, for
  example `fix/feral-singular-guard` or `docs/pr-conventions`.
- Keep one PR to one topic. If the description needs the word "also", consider
  splitting it into two PRs.

## Setting up the environment

SiNDAE's solver stack (POUNCE, FERAL, and PyNumero's grey-box path) needs more than
a plain `pip install`. Use the conda environment:

```bash
conda env create -f environment.yml   # creates the `sindae` environment
conda activate sindae
pip install -e .
```

POUNCE and FERAL are required dependencies and install automatically. cyipopt and
mpi4py are optional — they are only needed for the cyipopt backend and the MPI
examples. Install them with `pip install -e ".[full]"` (conda is preferred for
cyipopt, whose wheels are platform-dependent).

## Before you open a PR

Run the fast test suite and the linter from the repository root:

```bash
pytest -m "not slow"     # exactly what CI runs
ruff check .             # style and error check
```

- The fast suite must pass. CI runs `pytest -m "not slow"` and nothing more.
- Long-running tests are marked `slow` and are skipped by CI. If your change touches
  solver behavior, training, or inference, run the relevant slow tests locally too
  (`pytest -m slow -k <name>`), because CI will not.
- Any test that runs for more than a few seconds, or that needs HSL/MA27 or an ipopt
  binary, should be marked `@pytest.mark.slow` and should skip cleanly when its
  solver is not installed — not error.

We use pre-commit to run ruff. Install it once with `pre-commit install`; after that
it runs on each commit.

## Writing the PR

Fill in the pull request template. Keep the description factual:

- Say what changed and why, in plain terms.
- List the concrete changes.
- Show how you tested it: the command and the result.
- Point out anything a reviewer should look at closely — trade-offs, decisions you
  were unsure about, and follow-up work you are leaving for later.

## Commits

- Start the summary line with a verb in the imperative: "Add ...", "Fix ...",
  "Remove ...". Use the body to explain *why* when it is not obvious from the change.
- Several commits per PR is fine; you do not need to squash before review.

## What CI checks

Every push to `main` and every pull request runs two jobs:

- **Lint** — ruff. Real errors (syntax errors, undefined names) block the merge. The
  full style check is advisory for now, so it reports issues without failing.
- **Tests** — builds the conda solver stack, compiles PyNumero's ASL extension, and
  runs `pytest -m "not slow"`.

If CI is red, fix it before asking for review.
