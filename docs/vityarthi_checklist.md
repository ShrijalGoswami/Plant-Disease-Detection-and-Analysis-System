# VITyarthi Requirement Checklist

Each row maps a requirement from the "Build Your Own Project" guideline PDF to the
place in this repository where it is satisfied.

## Section 2.1 Functional requirements

| Requirement | Where |
|-------------|-------|
| Three major functional modules | Image processing (`src/preprocessing.py`), Disease detection (`src/model.py`, `src/prediction.py`), Analysis and reporting (`src/analytics.py`). Each has its own section in the app and its own test file. |
| Clear input / output structure | Input: one JPG or PNG file. Output: crop, condition, status, confidence, top 3, history table, CSV and TXT export. Documented in `docs/report_content.md` section 4. |
| Logical workflow of user interaction | Workflow diagram in `docs/diagrams.md` section 2; the app follows Step 1 (processing) then Step 2 (detection) then the analytics tab. |

## Section 2.2 Non-functional requirements (at least four)

| Requirement | Where |
|-------------|-------|
| Six NFRs: performance, usability, reliability and error handling, maintainability, resource efficiency, reproducibility and logging | `docs/report_content.md` section 5, each with how it is met and, where measurable, the measured value in `docs/results.md` |

## Section 3 Technical expectations

| Requirement | Where |
|-------------|-------|
| Proper architectural design | `docs/report_content.md` section 6 and the architecture diagram in `docs/diagrams.md` |
| Correct application of subject concepts | Image decoding, colour conversion, resizing, normalisation, augmentation, CNN, transfer learning, softmax confidence, precision / recall / F1, confusion matrix. Listed with reasons in `README.md` (Computer Vision approach) |
| Modular and clean implementation | 7 modules in `src/`, 3 scripts in `training/`, 1 config file, 1 app file. The longest file is `app.py` at 277 lines; every other file is under 135 lines. pyflakes reports no unused imports or undefined names. |
| Appropriate documentation and comments | Module docstrings explain purpose; comments explain non obvious choices (why INTER_AREA, why augmentation before normalisation, why last_recorded is kept). |
| Validation and error handling | `validate_upload`, `decode_image`, `ImageValidationError`, `FileNotFoundError` for missing model, all caught in `app.py` and covered by tests |
| Version control usage (Git) | `.gitignore` prepared; commits, tags and push are done by the student |
| Minimum 5 to 10 meaningful modules / files | 17 Python files, not counting the two empty `__init__.py`: `app.py`, `config.py`, 7 in `src/`, 3 in `training/`, 5 in `tests/` (4 test files plus `conftest.py`) |
| Proper folder structure | `src/`, `training/`, `tests/`, `models/`, `assets/`, `docs/`, `data/` |
| Testing | 22 pytest tests in `tests/test_*.py` plus 8 browser level checks in `tests/browser_check.py`, results in `docs/test_results.md` |

## Section 4 Design and documentation

| Requirement | Where |
|-------------|-------|
| Problem statement | `statement.md`, `docs/report_content.md` section 3 |
| Objectives | `docs/report_content.md` section 3 |
| Functional requirements | `docs/report_content.md` section 4 (FR1.1 to FR3.7) |
| Non-functional requirements | `docs/report_content.md` section 5 |
| System architecture diagram | `docs/diagrams.md` section 1, rendered at `assets/diagrams/01_system_architecture_diagram.png` |
| Process flow / workflow diagram | `docs/diagrams.md` sections 2 and 3, rendered as `02_...png` and `03_...png` |
| Use case diagram | `docs/diagrams.md` section 4, rendered as `04_use_case_diagram.png` |
| Class / component diagram | `docs/diagrams.md` section 6, rendered as `06_class_component_diagram.png` |
| Sequence diagram | `docs/diagrams.md` section 5, rendered as `05_sequence_diagram_single_prediction.png` |
| ER diagram / schema (if applicable) | Not applicable: no database. History is an in memory list. Explained in `docs/diagrams.md` and `docs/report_content.md` section 6. |
| Dataset description | `README.md` (Dataset), `docs/results.md` (counts per split), `docs/report_content.md` section 9 |
| Model selection rationale | `docs/report_content.md` section 8, `src/model.py` docstring |
| Evaluation methodology | Held out stratified test split, accuracy plus macro precision / recall / F1 and confusion matrix. `src/evaluation.py`, `training/evaluate.py`, `docs/results.md` |

## Section 5 GitHub repository

| Requirement | Where |
|-------------|-------|
| README.md with title, overview, features, technologies, install and run steps, testing instructions, screenshots | `README.md`, all sections present, screenshots in `assets/screenshots/` |
| statement.md with problem statement, scope, target users, high level features | `statement.md` |
| Source code, data files, scripts, assets, configuration | Source in `src/`, `training/`, `app.py`; config in `config.py`; split CSVs in `data/splits/`; weights and metrics in `models/`; samples and screenshots in `assets/`. The raw dataset is excluded by `.gitignore` because of its size, with download steps in the README. |

## Section 6 Project report sections

| Report section | Content source |
|----------------|----------------|
| 1 Cover page | `docs/report_content.md` section 1 (student details to be filled in) |
| 2 Introduction | section 2 |
| 3 Problem statement | section 3 |
| 4 Functional requirements | section 4 |
| 5 Non-functional requirements | section 5 |
| 6 System architecture | section 6 |
| 7 Design diagrams | PNG files in `assets/diagrams/`, source in `docs/diagrams.md` |
| 8 Design decisions and rationale | section 8 |
| 9 Implementation details | section 9 |
| 10 Screenshots / results | `docs/results.md`, `assets/screenshots/` |
| 11 Testing approach | `docs/test_results.md` |
| 12 Challenges faced | `docs/results.md` (Challenges) |
| 13 Learnings and key takeaways | `docs/results.md` (Learnings) |
| 14 Future enhancements | section 14 |
| 15 References | section 15 |

## Honesty checks

* All metrics come from `models/metrics.json` and `models/training_history.json`, produced on 11 September 2026 by the scripts in this repository.
* All screenshots were taken from the running app with the trained model.
* Test output in `docs/test_results.md` is copied from the actual pytest run.
* No em dashes are used anywhere in the code or documentation (checked with grep).
