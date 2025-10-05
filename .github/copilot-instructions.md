## FACSPy — instructions for AI coding agents

Purpose: give an AI agent the minimal, high-value context to be immediately productive in this repository.

- Repo entrypoints and public API
  - Primary package module: `FACSPy/FACSPy/__init__.py`. This file re-exports the main namespaces used by callers: `dt` (dataset), `tl` (tools), `pl` (plotting), `ml` (models/ml), and `sync` (synchronization). Prefer these public exports when changing user-facing behavior.

- Quick setup (developer/test loop)
  - Recommended: use conda and install locally.
  ```powershell
  conda create -n facspy python=3.10; conda activate facspy
  pip install .
  ```
  - Run tests (fast path, focused file):
  ```powershell
  pytest -q FACSPy/FACSPy/tests/test_io.py
  ```

- Where to look first (key files & folders)
  - `FACSPy/FACSPy/__init__.py` — public API and re-exports.
  - `FACSPy/FACSPy/tools/` — wrappers for dimensionality reduction, clustering, and metrics (e.g. `_umap.py`, `_tsne.py`, `_pca.py`, `_phenograph.py`).
  - `FACSPy/FACSPy/transforms/` — transformation implementations (asinh, logicle, etc.).
  - `FACSPy/FACSPy/io/_io.py` — dataset read/write helpers.
  - `FACSPy/FACSPy/_utils.py` — common helpers used across modules.
  - `FACSPy/FACSPy/_resources/` — test `.fcs`/`.wsp` fixtures and XSDs used by tests.
  - `FACSPy/FACSPy/tests/` — pytest suite; tests include baseline images.
  - `pyproject.toml` — dependency pins and extras (docs, dev, r_env).

- Project-specific conventions and gotchas
  - Short namespace aliases (`fp.dt`, `fp.pl`, `fp.tl`, etc.) are the intended public surface. Keep backwards compatible names in `__init__.py` when refactoring.
  - Underscored modules (files beginning with `_`) are internal. Prefer changing public helpers unless you're explicitly adding internal helpers.
  - Many tests assert on baseline images. Small plotting default changes (figure sizes, default colormaps, DPI) can break tests — run affected tests locally.
  - Heavy/fragile dependencies: `scanpy`, `phenograph`, `parc`, and `flowsom` (installed from a git reference). R interop uses optional extras (`rpy2`, `anndata2ri`). Document any added native/compiled dependencies.

- Example snippets (use these when adding or testing features)
  - Create dataset (from README, canonical):
  ```python
  import FACSPy as fp
  dataset = fp.dt.create_dataset(metadata=fp.dt.Metadata('metadata.csv'), panel=fp.dt.Panel('panel.csv'), workspace=fp.dt.FlowJoWorkspace('workspace.wsp'))
  ```
  - Transform and plot:
  ```python
  fp.dt.calculate_cofactors(dataset)
  fp.dt.transform(dataset, transform='asinh', cofactor_table=dataset.uns['cofactors'], key_added='transformed', layer='compensated')
  fp.pl.biax(dataset, sample_identifier='2', marker='CD38')
  ```

- Editing guidance (where to change behavior safely)
  - Change public behavior: update `FACSPy/FACSPy/__init__.py` exports and the top-level modules under `FACSPy/FACSPy` (e.g. `tools/`, `transforms/`, `io/`).
  - Add new tools: put them under `FACSPy/FACSPy/tools/` with a leading underscore for internal helpers and expose a stable wrapper in the `tl` namespace.
  - Add tests alongside features in `FACSPy/FACSPy/tests/` and include small resource fixtures in `_resources/` when needed.

- CI & docs
  - CI test workflow: `.github/workflows/pytest.yml` — follow its matrix if you modify tests.
  - Docs: Sphinx + nbsphinx; dev extra `.[docs]` in `pyproject.toml` lists required packages.

- Quick triage checklist for PRs
  1. Run the focused pytest(s) for changed modules.
 2. If plotting code changed, run the image-baseline tests in `tests/baseline_images`.
 3. Update `pyproject.toml` only if dependency change is necessary and note the reason in the PR description.

If anything here is unclear or you want the same guidance adapted into a different format (shorter, or more verbose with file links), tell me which section to expand and I'll iterate.
