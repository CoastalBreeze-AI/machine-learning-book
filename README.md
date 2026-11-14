# Machine Learning: Foundations, Algorithms, and Practice — Code

The code for *Machine Learning: Foundations, Algorithms, and Practice*, by Kevin A. Lee, PhD, Book 2 of the Computer Science Series. The book teaches machine learning by building a small library, `tinyml`, from scratch in Python and NumPy, from linear models and trees to neural networks, Transformers, reinforcement learning, and production systems.

This repository matches the **first printing, 2026**. Corrections are listed in [ERRATA.md](ERRATA.md).

## What's here

| Item | Contents |
|---|---|
| `tinyml/` | The complete library: 24 modules and an empty `__init__.py`. Each module is the concatenation of its listings, in chapter order (Appendix A, Table A.1) |
| `tests/` | The library's tests, one file per chapter from 3 to 16 (Appendix A, Section A.4) |
| `labs/` | Each chapter's lab listings that aren't module code, as one runnable script per chapter, generated from the chapter. `ch13_sort_lab.py` is the script the chapter calls `sort_lab.py` |
| `CHECKSUMS.txt` | The SHA-256 hash of every file in `tinyml/` |
| `ERRATA.md` | Corrections to the book and the code |

**Building it yourself is the point.** `tinyml/` here is the finished library. The labs teach by having you type each module from its listings, run the tests, and break things on purpose to see what fails. Use this copy to check your work or to unblock yourself, not as a substitute (Appendix A, Section A.3).

## Quick start

You need **Python 3.10 or newer** (the book was tested on Python 3.14.6) and **NumPy** (tested with 2.5.2). Nothing needs a GPU, a network connection, or a dataset download.

```text
git clone https://github.com/coastalbreeze-ai/machine-learning-book.git
cd machine-learning-book
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install numpy
```

**Run the tests** from this folder:

```text
python -m unittest discover -s tests      # all 213 tests
python -m unittest tests.test_ch05        # one chapter's tests
```

**Run a lab script** from this folder or from `labs/`:

```text
python labs/ch03_leakage_lab.py
```

Each lab script puts this folder on Python's path itself, so `tinyml` imports either way. Your own scripts should live in this folder, next to `tinyml`, or set `PYTHONPATH` to it (Appendix A, Section A.5).

Printed numbers should match the book exactly, except fit and prediction times and, on a different machine or BLAS library, the last digits of a few ill-conditioned results and the model hash in Chapter 15 (Appendix A, Section A.6).

To check that your copy of the library is unchanged, run `sha256sum -c CHECKSUMS.txt` (macOS: `shasum -a 256 -c CHECKSUMS.txt`). The repository's `.gitattributes` stops Git from converting line endings, which would change the hashes.

## What's not here

The solutions manual, the graded labs and their reference solutions, exam materials, slides, and other instructor materials are not in this repository. Instructors can request them by email at [instructors@coastalbreeze.ai](mailto:instructors@coastalbreeze.ai).

## License

The code in this repository is released under the [MIT License](LICENSE), copyright 2026 Kevin A. Lee. The book's text is not covered by that license; it remains under the book's copyright, all rights reserved.
