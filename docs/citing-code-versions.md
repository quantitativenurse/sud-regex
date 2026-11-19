# Citing code versions in papers

How the lab points a paper at the exact code that produced its results, so someone else can rerun it. Last checked: November 2026.

## The options

A branch name (like `main`) isn't one of them, because it moves with every commit and never names a single version.

| Option | What a reader gets | Pros | Cons |
|---|---|---|---|
| **Commit hash** (e.g. `62f884b`) | One exact snapshot of the repo | Free and automatic. Always exact. | Hard to read and says nothing about the version. Only lives on GitHub, so it breaks if the repo is made private or deleted. Changes if history is rewritten. Not citable. |
| **Git tag** (e.g. `v0.2.0`) | A readable name for one commit | Free. Can match the version number on PyPI. | Anyone with write access can move or delete it. Still GitHub-only. Not citable. |
| **GitHub Release** (made from a tag) | The tag plus release notes and a source zip | Gives a place to say what changed. With GitHub's release immutability setting turned on, the tag and files can't be changed afterward. Publishing a release is what triggers Zenodo. | Still GitHub-only. No DOI. |
| **Archived release with a DOI** (Zenodo) | A permanent copy of the release with its own DOI | Lives outside GitHub (Zenodo is run by CERN). Files can't be changed after publishing without contacting Zenodo support. A DOI is what journals and reference managers expect. GitHub's own docs recommend Zenodo for this. | The repo has to be public. Needs a one-time setup, and an org owner may need to approve the Zenodo app. Only releases published after setup are archived automatically. |

Also: people usually install from PyPI (`pip install sudregex==X.Y.Z`). A PyPI version can't be re-uploaded with different files, but it doesn't record which commit it came from. A matching tag fixes that.

**Bottom line:** these aren't either/or. Tag the commit, publish a GitHub Release from it, and Zenodo archives it and mints the DOI. One release covers all four. Zenodo gives each release a **version DOI** (that exact release) plus one **concept DOI** (all versions, resolves to the newest). Zenodo says to cite the version DOI, and papers should.

## Lab convention

**When to tag**
1. From now on, every version uploaded to PyPI gets a GitHub Release with tag `vX.Y.Z`, made from the exact commit that was built. Build and upload from that commit, not from uncommitted changes. (Older versions: see the sudregex appendix.)
2. Any analysis that goes into a paper runs on a released version (`pip install sudregex==X.Y.Z`), never on a branch or a local copy. If the analysis needs code that isn't released yet, cut a release first.
3. Repos that aren't on PyPI, like analysis code for a single paper, get a release at submission and another for each revision.
4. Before submitting, check that the version used has a release and a DOI.

**How tags are named**
- `vX.Y.Z`, matching `pyproject.toml` and PyPI exactly. The existing `v0.1.0` and `v0.1.2` already follow this.
- Bump the middle number (`0.1.x` to `0.2.0`) whenever a change can change the output, even for a bug fix. Bump the last number only when outputs stay the same.
- Make the tag with git first (`git tag -a`, which records who tagged and when), push it, then make the Release from that tag. A tag created in GitHub's release form doesn't record that.
- Never move, delete or reuse a tag. Turn on release immutability for lab repos that papers cite.
- Don't add paper-specific tags. After a paper comes out, add it to that version's release notes and to the Zenodo record's metadata. Metadata can be edited without changing the DOI.

**Where it goes in the manuscript**
- **Methods:** package name, version and version DOI.
- **Code availability statement:** the GitHub link plus the version DOI.
- **References:** a software citation with the version DOI.
- A commit hash is optional, in the supplement at most. The release tag already points to one exact commit.

**When code changes after submission or during revision**
- Never move a tag or replace a release. Make a new release, numbered by the rule above.
- **Rerun for a revision:** cite the new version and DOI in the revised manuscript. Say in the response to reviewers what changed and why.
- **Changes after acceptance:** the paper keeps citing the version that produced its results. If a later fix changes those results, note it in the new release notes, and the PI decides whether the journal needs to hear about it.

## Sources

- Zenodo, versioning and concept DOIs: https://zenodo.org/help/versioning
- Zenodo, GitHub integration: https://help.zenodo.org/docs/github/enable-repository
- Zenodo, editing published records: https://help.zenodo.org/docs/deposit/manage-records/
- Zenodo, `CITATION.cff` and `.zenodo.json`: https://help.zenodo.org/docs/github/describe-software/citation-file/
- GitHub, referencing and citing content: https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content
- GitHub, CITATION files: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files
- GitHub, immutable releases: https://docs.github.com/en/code-security/supply-chain-security/understanding-your-software-supply-chain/immutable-releases
- PyPI file-name reuse rule (Python forum thread): https://discuss.python.org/t/delete-an-existing-project-on-pypi-and-then-create-one-with-the-same-name-but-got-error-the-filename-has-already-been-used/11628

---

## Appendix: applied to sudregex

As of November 2026: 0.1.0–0.1.8 are on PyPI, but GitHub has only two tags (`v0.1.0`, `v0.1.2`), no Releases, no `CITATION.cff` and no DOI. The README asks people to cite "Version 0.2.0", which is on `main` but not on PyPI or tagged. This matters here because 0.2.0 changes final match counts compared with 0.1.x, so a paper has to say which side of that line it's on.

The full convention applies from 0.2.0 on. Older versions get a tag where a commit can be matched, and a Release + DOI only if a paper used them. Commits below were matched by PyPI upload time, commit time and the version in `pyproject.toml`. Confirm each against the PyPI files before tagging.

| Version | On PyPI | Matching commit | Tag today | What it would need |
|---|---|---|---|---|
| 0.1.0 | 2025-09-25 | `d0c1b40` | `v0.1.0` | Release + DOI if a paper used it |
| 0.1.1 | 2025-09-25 | `c7e50fa` (or `14dc12e`, lint-only difference) | none | Tag; Release + DOI if a paper used it |
| 0.1.2 | 2025-09-25 | `978910f` | `v0.1.2` | Release + DOI if a paper used it |
| 0.1.3 | 2025-10-27 | none, built before the version bump was committed | none | Can't be tagged exactly. Note it in the README. |
| 0.1.4 | 2025-10-27 | probably `61c2d34` (committed after upload) | none | Confirm against the PyPI files, then tag |
| 0.1.5 | 2025-11-25 | `603409f` | none | Tag; Release + DOI if a paper used it |
| 0.1.6 | 2026-03-12 | `f860352` (lint-only after `e61dc0f`) | none | Tag; Release + DOI if a paper used it |
| 0.1.7 | 2026-06-19 | `93cc96f` | none | Tag; Release + DOI if a paper used it |
| 0.1.8 | 2026-07-14 | `7c28933` | none | Tag; Release + DOI if a paper used it |
| 0.2.0 | not yet | `main` (`62f884b`) | none | Release + DOI + PyPI upload, all from the same commit |

**One-time setup, before any release goes out:** switch the repo on in Zenodo (an org owner may need to approve the Zenodo app for the `quantitativenurse` org first). Turn on release immutability. Add a `CITATION.cff` and point the README citation at the DOI. Zenodo's docs don't say whether a new Release on an old tag gets archived, so check on the first one. If it doesn't, Zenodo also takes a manual zip upload.

**Sample methods sentence**

> Notes were processed with sudregex version 0.2.0 (Quantitative Nurse Lab; https://github.com/quantitativenurse/sud-regex), a Python package for regular-expression extraction with negation and false-positive filtering, archived at https://doi.org/10.5281/zenodo.XXXXXXX.

**Sample reference:** Quantitative Nurse Lab. *sudregex* (version 0.2.0) [software]. Zenodo; 2026. doi:10.5281/zenodo.XXXXXXX

Change the version to the one the analysis actually used. The DOI is a placeholder until the release exists.
