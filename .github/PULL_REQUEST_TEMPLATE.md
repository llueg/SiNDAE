<!--
  This template becomes the starting text of every new pull request.
  Fill in each section and delete the comment lines. Keep it factual and short.
-->

## Summary

<!-- One or two sentences: what does this PR change, and why? -->

## Motivation

<!-- What problem does this solve? Link the issue if there is one (e.g. "Closes #12"). -->

## Changes

<!-- The concrete changes, as a list. -->
-

## Testing

<!-- How you verified this works. Paste the commands you ran and what happened —
     not just "tested locally". -->

- [ ] `pytest -m "not slow"` passes locally in the `sindae` conda env
- [ ] `ruff check .` reports no new errors
- [ ] Ran the relevant `slow` / end-to-end tests if this touches solver, training, or inference code (CI does not run them):

## Notes for reviewers

<!-- Anything a reviewer should look at closely: trade-offs, decisions you were
     unsure about, follow-up work you are deliberately leaving for later. -->
