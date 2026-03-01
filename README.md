# beam-tools

> Symbolic Euler–Bernoulli beam analysis using **Macaulay singularity functions** and **SymPy**.

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

---

## 📦 Python Library

You can use `beam_tools` directly as a Python library:

```bash
pip install -e .
```

### Example — simply supported beam with uniform load

```python
import sympy as sp
from beam_tools import Beam, Support, SupportType, UniformLoad, LoadOrientation

L, w = sp.symbols("L w", positive=True)

beam = Beam(
    length=L,
    supports=[
        Support(0, SupportType.HINGED, "A"),
        Support(L, SupportType.ROLLER, "B"),
    ],
    loads=[
        UniformLoad(w=w, start=0, end=L, orientation=LoadOrientation.VERTICAL),
    ],
)

print("Reactions:", beam.reactions)

V, M = beam.get_shear_force_and_bending_moment_equations()
theta, y = beam.get_slope_and_deflection_equations()
```

See [`examples/`](examples/) for more — simply supported, hyperstatic, and visualisation examples.

---

## ✨ Library Features

| Feature | Description |
|---|---|
| **Singularity functions** | Macaulay bracket notation ⟨x − a⟩ⁿ for piecewise-continuous expressions |
| **Symbolic analysis** | All results are SymPy expressions — substitute numbers whenever you want |
| **Hyperstatic beams** | Automatically detects indeterminate structures |
| **Multiple load types** | Point loads, uniform loads, triangular loads, applied moments |
| **Multiple support types** | Fixed, hinged (pinned), and roller |
| **Deflection & slope** | Integrates the bending-moment equation and applies boundary conditions |

---

## 🗂 Project Structure

```
beam-tools/
├── src/beam_tools/        # Python library source
├── examples/              # Usage examples
├── beam-web/
│   ├── backend/           # FastAPI backend (solver API)
│   └── frontend/          # React + TypeScript frontend
├── docker-compose.yml     # One-command deployment
├── Makefile               # Convenience commands
└── .env.example           # Environment variable template
```

---

## 📄 License

MIT — see [LICENSE](LICENSE).
