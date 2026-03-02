"""
Singularity (Macaulay) bracket notation for symbolic beam analysis.

Provides a custom SymPy Piecewise subclass that renders using angle-bracket
notation ⟨x − a⟩ⁿ in plain text, LaTeX, and code generation.
"""

import sympy as sp

x_symb = sp.symbols('x')


class SingularityBracket(sp.Piecewise):
    """Custom Piecewise that prints using Macaulay bracket notation."""

    def __new__(cls, *args, **options):
        return super().__new__(cls, *args, **options)

    def _sympystr(self, printer):
        cond, expr = self.args

        expr = expr[0]
        cond: sp.LessThan = cond[1]

        if isinstance(expr, sp.Pow):
            base = expr.base
            exp = int(expr.exp)
            return "[⟨{}⟩^{}]".format(printer.doprint(base), printer.doprint(exp))

        # Always extract 'a' from the condition (x < a) so the position
        # is never lost — even when n=0 causes (x-a)^0 to simplify to 1.
        a = cond.args[1] if cond.args[0] == x_symb else cond.args[0]
        x = cond.args[0] if cond.args[0] == x_symb else cond.args[1]

        if expr == 1:
            n = 0
        else:
            n = 1

        return "[⟨{}-{}⟩^{}]".format(printer.doprint(x), printer.doprint(a), n)

    def _latex(self, printer):
        cond, expr = self.args

        expr = expr[0]
        cond: sp.LessThan = cond[1]

        if isinstance(expr, sp.Pow):
            base = expr.base
            exp = int(expr.exp)
            return "\\langle {} \\rangle^{}".format(
                printer.doprint(base), printer.doprint(exp)
            )

        # Always extract 'a' from the condition — same fix as _sympystr.
        a = cond.args[1] if cond.args[0] == x_symb else cond.args[0]
        x = cond.args[0] if cond.args[0] == x_symb else cond.args[1]

        if expr == 1:
            n = 0
        else:
            n = 1

        return "\\langle {} - {} \\rangle^{}".format(
            printer.doprint(x), printer.doprint(a), n
        )

    def _pprint(self, printer, *args, **kwargs):
        return self._sympystr(printer)

    def _eval_code(self, printer):
        cond, expr = self.args
        expr = expr[0]
        cond = cond[1]
        return "numpy.where(not {}, {}, 0)".format(
            printer.doprint(cond), printer.doprint(expr)
        )

    def _numpycode(self, printer):
        return self._eval_code(printer)

    def _ccode(self, printer):
        return self._eval_code(printer)


def symb_singular_function(a, n):
    """
    Return a symbolic expression for ⟨x − a⟩ⁿ using Macaulay bracket notation.

    Parameters
    ----------
    a : sympy expression
        The offset value in the singularity function.
    n : int or float
        The exponent of the singularity function.

    Returns
    -------
    SingularityBracket
        A custom Piecewise expression: 0 for x < a, (x − a)ⁿ for x ≥ a.
    """
    return SingularityBracket(
        (0, x_symb < a),
        ((x_symb - a) ** n, x_symb >= a),
    )

