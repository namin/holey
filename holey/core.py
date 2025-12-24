from typing import Any, Optional, List, Type
from dataclasses import dataclass
from contextlib import contextmanager
from .backend import default_backend, Backend

@dataclass 
class SymbolicTracer:
    def __init__(self, backend=None, llm_solver=None):
        self.backend = backend or default_backend()
        self.llm_solver = llm_solver
        self.path_conditions = []
        self.forall_conditions = []
        self.branch_counter = 0
        self.current_branch_exploration = []
        self.remaining_branch_explorations = []

    def driver(self, thunk):
        while True:
            result = thunk()
            self.add_constraint(result)
            if self.remaining_branch_explorations == []:
                return
            else:
                self.current_branch_exploration = self.remaining_branch_explorations.pop()
                self.branch_counter = 0
                self.path_conditions = []

    def branch(self, condition):
        """Handle branching with optional LLM guidance"""
        if len(self.current_branch_exploration) > self.branch_counter:
            branch_val = self.current_branch_exploration[self.branch_counter]
        else:
            # Use LLM guidance if available
            if self.llm_solver:
                branch_val = self.llm_solver.get_branch_guidance(
                    condition, self.path_conditions
                )
            else:
                branch_val = True
                
            self.remaining_branch_explorations.append(
                self.current_branch_exploration + [not branch_val]
            )
            self.current_branch_exploration += [branch_val]

        self.branch_counter += 1
        condition = self.ensure_symbolic(condition)
        self.path_conditions.append(
            condition.z3_expr if branch_val
            else self.backend.Not(condition.z3_expr)
        )
        return branch_val

    def add_constraint(self, constraint):
        """Add constraint with optional LLM refinement"""
        constraint = truthy(constraint)
        if hasattr(constraint, 'z3_expr'):
            constraint = constraint.z3_expr
            
        if self.llm_solver:
            self.llm_solver.add_constraint_refinements(constraint, self.backend)

        if self.path_conditions:
            constraint = self.backend.Implies(
                self.backend.And(*self.path_conditions),
                constraint
            )

        if self.forall_conditions:
            for var_expr, bounds_expr in self.forall_conditions:
                var_decl = self.backend.Int(var_expr.decl().name())
                constraint = self.backend.ForAll(
                    [var_decl],
                    self.backend.Implies(
                        bounds_expr,
                        constraint
                    )
                )

        self.backend.solver.add(constraint)
    
    def check(self):
        """Check satisfiability including path conditions"""
        result = self.backend.check()
        return result
    
    def model(self):
        return self.backend.solver.model()

    def solution(self):
        result = self.check()
        if self.backend.is_sat(result):
            return self.model()
        return None

    def solution_var(self, model, var):
        # Handle BoundedSymbolicList - extract individual variables
        if isinstance(var, BoundedSymbolicList):
            result = []
            for i in range(var.size):
                var_name = f"{var.name}_e{i}"
                if var_name in model:
                    result.append(model[var_name])
                else:
                    # Variable not in model - use default or raise
                    print(f"Warning: {var_name} not in model")
                    return None
            return result
        return model[var.name]
    
    def ensure_symbolic(self, other):
        if isinstance(other, bool):
            return SymbolicBool(other, tracer=self)
        if isinstance(other, int):
            return SymbolicInt(other, tracer=self)
        if isinstance(other, float):
            return SymbolicFloat(other, tracer=self)
        if isinstance(other, str):
            return SymbolicStr(other, tracer=self)
        if isinstance(other, list):
            typ = find_element_type_of(other)
            print("typ", typ, "for", other)
            return SymbolicList(other, elementTyp=typ, tracer=self)
        if isinstance(other, SymbolicSlice):
            return other.get_slice()
        return other

def find_element_type_of(xs):
    if not xs:
        return None
    x = xs[0]
    if isinstance(x, SymbolicInt):
        return int
    if isinstance(x, SymbolicStr):
        return str
    return type(x)

class SymbolicBool:
    """Wrapper class for symbolic boolean expressions"""
    def __init__(self, value: Optional[bool] = None, name: Optional[str] = None, tracer: Optional[SymbolicTracer] = None):
        self.tracer = tracer or SymbolicTracer()
        self.concrete = None
        # TODO: weird that we need this
        if str(value) in ['true', 'false']:
            value = str(value)!='false'
        if name is not None:
            self.z3_expr = self.tracer.backend.Bool(name)
        elif isinstance(value, bool):
            self.concrete = value
            self.z3_expr = self.tracer.backend.BoolVal(value)
        else:
            self.z3_expr = value
        self.name = name

    def __bool__(self):
        if self.concrete is not None:
            return self.concrete
        return self.tracer.branch(self)

    def __and__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicBool(self.tracer.backend.And(self.z3_expr, truthy(other).z3_expr), tracer=self.tracer)
    
    def __or__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicBool(self.tracer.backend.Or(self.z3_expr, truthy(other).z3_expr), tracer=self.tracer)
    
    def __not__(self):
        if self.concrete is not None:
            return SymbolicBool(not self.concrete, tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.Not(self.z3_expr), tracer=self.tracer)

    def __eq__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if not isinstance(other, SymbolicBool):
            other = truthy(other)
        if self.concrete is not None and hasattr(other, 'concrete') and other.concrete is not None:
            return SymbolicBool(self.concrete == other.concrete, tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.Eq(self.z3_expr, other.z3_expr), tracer=self.tracer)

    def __ne__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if not isinstance(other, SymbolicBool):
            other = truthy(other)
        if self.concrete is not None and hasattr(other, 'concrete') and other.concrete is not None:
            return SymbolicBool(self.concrete != other.concrete, tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.Not(self.tracer.backend.Eq(self.z3_expr, other.z3_expr)), tracer=self.tracer)

    def __add__(self, other):
        as_int = SymbolicInt(
            int(self.concrete) if self.concrete is not None else
            self.tracer.backend.If(
                self.z3_expr,
                self.tracer.backend.IntVal(1),
                self.tracer.backend.IntVal(0)
            ),
            tracer=self.tracer
        )
        if isinstance(other, SymbolicBool):
            other = SymbolicInt(
                int(other.concrete) if other.concrete is not None else
                self.tracer.backend.If(
                    other.z3_expr,
                    self.tracer.backend.IntVal(1),
                    self.tracer.backend.IntVal(0)
                ),
                tracer=self.tracer
            )
        return as_int + other
        
    def __radd__(self, other):
        return self.__add__(other)

class SymbolicInt:
    """Wrapper class for symbolic integer expressions"""
    def __init__(self, value: Optional[Any] = None, name: Optional[str] = None, tracer: Optional[SymbolicTracer] = None):
        self.tracer = tracer or SymbolicTracer()
        self.concrete = None
        if name is not None:
            self.z3_expr = self.tracer.backend.Int(name)
        elif isinstance(value, int):
            self.concrete = value
            self.z3_expr = self.tracer.backend.IntVal(value)
        else:
            self.z3_expr = value
        self.name = name

    def __int__(self):
        if self.concrete is not None:
            return self.concrete
        raise ValueError("SymbolincInt cannot be concretized.")

    def __str__(self):
        return f"SymbolicInt({self.name})"

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if isinstance(other, SymbolicFloat):
            return SymbolicFloat(self.tracer.backend.Add(self.z3_expr, other.z3_expr), tracer=self.tracer)
        if self.concrete is not None and other.concrete is not None:
            return SymbolicInt(self.concrete + other.concrete, tracer=self.tracer)
        if isinstance(other, SymbolicBool):
            as_int = SymbolicInt(
                int(other.concrete) if other.concrete is not None else
                self.tracer.backend.If(
                    other.z3_expr,
                    self.tracer.backend.IntVal(1),
                    self.tracer.backend.IntVal(0)
                ),
                tracer=self.tracer
            )
            return SymbolicInt(self.tracer.backend.Add(self.z3_expr, as_int.z3_expr), tracer=self.tracer)
        return SymbolicInt(self.tracer.backend.Add(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __sub__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if isinstance(other, SymbolicFloat):
            return SymbolicFloat(self.tracer.backend.Sub(self.z3_expr, other.z3_expr), tracer=self.tracer)
        if self.concrete is not None and other.concrete is not None:
            return SymbolicInt(self.concrete - other.concrete, tracer=self.tracer)
        return SymbolicInt(self.tracer.backend.Sub(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __mul__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if isinstance(other, SymbolicFloat):
            return SymbolicFloat(self.tracer.backend.Mul(self.z3_expr, other.z3_expr), tracer=self.tracer)
        if self.concrete is not None and other.concrete is not None:
            return SymbolicInt(self.concrete * other.concrete, tracer=self.tracer)
        if isinstance(other, SymbolicStr):
            return SymbolicStr(self.tracer.backend.StrMul(other.z3_expr, self.z3_expr), tracer=self.tracer)
        return SymbolicInt(self.tracer.backend.Mul(self.z3_expr, other.z3_expr), tracer=self.tracer)
        
    def __eq__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            return SymbolicBool(self.concrete == other.concrete, tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.Eq(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __lt__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            return SymbolicBool(self.concrete < other.concrete, tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.LT(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __gt__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            return SymbolicBool(self.concrete > other.concrete, tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.GT(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicInt(self.z3_expr / other.z3_expr, tracer=self.tracer)
    
    def __rtruediv__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicInt(other.z3_expr / self.z3_expr, tracer=self.tracer)
    
    def _add_nonzero_constraint(self, divisor):
        """Add constraint that divisor is not zero, unless it's concrete non-zero or a quantified variable."""
        # Skip if we know the concrete value is non-zero
        if divisor.concrete is not None and divisor.concrete != 0:
            return

        backend = self.tracer.backend
        is_quantified = False
        if hasattr(divisor.z3_expr, 'decl') and callable(divisor.z3_expr.decl):
            try:
                var_name = str(divisor.z3_expr.decl().name())
                is_quantified = var_name in backend.quantified_vars
            except:
                pass

        if not is_quantified:
            self.tracer.add_constraint(backend.Not(backend.Eq(divisor.z3_expr, backend.IntVal(0))))

    def _python_floor_div(self, dividend, divisor):
        """Helper to implement Python's floor division semantics.
        Python: rounds toward negative infinity
        SMT-LIB div: Euclidean (remainder always non-negative)

        To convert from Euclidean to floor division:
        - If divisor < 0 and remainder != 0, subtract 1 from the quotient
        """
        self._add_nonzero_constraint(divisor)

        backend = self.tracer.backend
        euclidean_div = backend.UDiv(dividend.z3_expr, divisor.z3_expr)
        euclidean_mod = backend.Mod(dividend.z3_expr, divisor.z3_expr)

        # Adjust when divisor < 0 and remainder != 0
        divisor_negative = backend.LT(divisor.z3_expr, backend.IntVal(0))
        has_remainder = backend.Not(backend.Eq(euclidean_mod, backend.IntVal(0)))
        needs_adjustment = backend.And(divisor_negative, has_remainder)

        return backend.If(needs_adjustment,
                         backend.Sub(euclidean_div, backend.IntVal(1)),
                         euclidean_div)

    def __floordiv__(self, other):
        other = self.tracer.ensure_symbolic(other)
        result = self._python_floor_div(self, other)
        return SymbolicInt(result, tracer=self.tracer)

    def __rfloordiv__(self, other):
        other = self.tracer.ensure_symbolic(other)
        result = self._python_floor_div(other, self)
        return SymbolicInt(result, tracer=self.tracer)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            other = SymbolicInt(self.tracer.backend.IntVal(other), tracer=self.tracer)
        return SymbolicInt(self.tracer.backend.Sub(other.z3_expr, self.z3_expr), tracer=self.tracer)
    
    def __hash__(self):
        return hash(str(self.z3_expr))

    def __abs__(self):
        if self.concrete is not None:
            return abs(self.concrete)
        return SymbolicInt(self.tracer.backend.If(self.z3_expr >= 0, 
                                                  self.z3_expr, 
                                                  -self.z3_expr), 
                           tracer=self.tracer)

    def _python_mod(self, dividend, divisor):
        """Helper to implement Python's modulo semantics.
        Python: result has same sign as divisor
        SMT-LIB mod: result has same sign as dividend

        For now, we just use SMT-LIB's mod directly.
        The full Python semantics would require complex logic that causes timeouts.
        Most puzzles use positive numbers where Python and SMT-LIB agree.
        """
        self._add_nonzero_constraint(divisor)
        return self.tracer.backend.Mod(dividend.z3_expr, divisor.z3_expr)

    def __mod__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            return SymbolicInt(self.concrete % other.concrete, tracer=self.tracer)

        result = self._python_mod(self, other)
        return SymbolicInt(result, tracer=self.tracer)

    def __rmod__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            return SymbolicInt(other.concrete % self.concrete, tracer=self.tracer)

        result = self._python_mod(other, self)
        return SymbolicInt(result, tracer=self.tracer)


    def __pow__(self, other, mod=None):
        other = self.tracer.ensure_symbolic(other)
        if mod is not None:
            # Modular exponentiation: (self ** other) % mod
            mod = self.tracer.ensure_symbolic(mod)
            if self.concrete is not None and other.concrete is not None and mod.concrete is not None:
                return SymbolicInt(pow(self.concrete, other.concrete, mod.concrete), tracer=self.tracer)
            # For symbolic: compute (base ^ exp) mod m
            pow_result = self.tracer.backend.Pow(self.z3_expr, other.z3_expr)
            return SymbolicInt(self.tracer.backend.Mod(pow_result, mod.z3_expr), tracer=self.tracer)
        if isinstance(other, SymbolicFloat):
            return SymbolicFloat(self.tracer.backend.Pow(self.z3_expr, other.z3_expr), tracer=self.tracer)
        if self.concrete is not None and other.concrete is not None:
            result = self.concrete ** other.concrete
            if isinstance(result, float):
                return SymbolicFloat(result, tracer=self.tracer)
            return SymbolicInt(result, tracer=self.tracer)
        return SymbolicInt(self.tracer.backend.Pow(self.z3_expr, other.z3_expr), tracer=self.tracer)

    def __rpow__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if isinstance(other, SymbolicFloat):
            return SymbolicFloat(self.tracer.backend.Pow(other.z3_expr, self.z3_expr), tracer=self.tracer)
        return SymbolicInt(self.tracer.backend.Pow(other.z3_expr, self.z3_expr), tracer=self.tracer)


    def __le__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicBool(self.tracer.backend.LE(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __ge__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            return SymbolicBool(self.concrete >= other.concrete, tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.GE(self.z3_expr, other.z3_expr), tracer=self.tracer)

    def __ne__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            return SymbolicBool(self.concrete != other.concrete, tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.Not(self.tracer.backend.Eq(self.z3_expr, other.z3_expr)), tracer=self.tracer)

    def __divmod__(self, other):
        other = self.tracer.ensure_symbolic(other)
        q = SymbolicInt(self.z3_expr / other.z3_expr, tracer=self.tracer)
        r = SymbolicInt(self.z3_expr % other.z3_expr, tracer=self.tracer)
        return (q, r)

    def __rdivmod__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return divmod(other, self)

    def __lshift__(self, other):
        other = self.tracer.ensure_symbolic(other)
        two = self.tracer.backend.IntVal(2)
        return SymbolicInt(self.tracer.backend.Mul(self.z3_expr, self.tracer.backend.Pow(two, other.z3_expr)), tracer=self.tracer)

    def __rlshift__(self, other):
        # other << self  =>  other * (2 ** self)
        other = self.tracer.ensure_symbolic(other)
        two = self.tracer.backend.IntVal(2)
        return SymbolicInt(self.tracer.backend.Mul(other.z3_expr, self.tracer.backend.Pow(two, self.z3_expr)), tracer=self.tracer)

    def __rshift__(self, other):
        other = self.tracer.ensure_symbolic(other)
        two = self.tracer.backend.IntVal(2)
        return SymbolicInt(self.tracer.backend.Div(self.z3_expr, self.tracer.backend.Pow(two, other.z3_expr)), tracer=self.tracer)

    def __rrshift__(self, other):
        # other >> self  =>  other / (2 ** self)
        other = self.tracer.ensure_symbolic(other)
        two = self.tracer.backend.IntVal(2)
        return SymbolicInt(self.tracer.backend.UDiv(other.z3_expr, self.tracer.backend.Pow(two, self.z3_expr)), tracer=self.tracer)

    def __neg__(self):
        if self.concrete is not None:
            return SymbolicInt(-self.concrete, tracer=self.tracer)
        return SymbolicInt(-self.z3_expr, tracer=self.tracer)

    def is_integer(self):
        """Always returns True since SymbolicInt is always an integer"""
        return SymbolicBool(True, tracer=self.tracer)

    def __and__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicBool(self.tracer.backend.And(truthy(self).z3_expr, truthy(other).z3_expr), tracer=self.tracer)
    
    def __or__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicInt(self.tracer.backend.If(truthy(self).z3_expr, self.z3_expr, other.z3_expr), tracer=self.tracer)

    def __xor__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicInt(self.tracer.backend.Xor(self.z3_expr, other.z3_expr), tracer=self.tracer)

    def __rand__(self, other):
        return self.__and__(other)

    def __ror__(self, other):
        return self.__or__(other)

    def __rxor__(self, other):
        return self.__xor__(other)

    def __index__(self):
        if self.concrete is not None:
            return self.concrete
        raise ValueError("Cannot convert symbolic integer to index")

    def __not__(self):
        if self.concrete is not None:
            return SymbolicBool(not self.concrete, tracer=self.tracer)
        return self != 0

class SymbolicFloat:
    def __init__(self, value: Optional[Any] = None, name: Optional[str] = None, tracer: Optional[SymbolicTracer] = None):
        self.tracer = tracer or SymbolicTracer()
        self.concrete = None
        if name is not None:
            self.z3_expr = self.tracer.backend.Real(name)
        elif isinstance(value, (int, float)):
            self.concrete = value
            self.z3_expr = self.tracer.backend.RealVal(float(value))
        else:
            self.z3_expr = value
        self.name = name

    def __sub__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicFloat(self.tracer.backend.Sub(self.z3_expr, other.z3_expr), tracer=self.tracer)

    def __abs__(self):
        result = SymbolicFloat(
            self.tracer.backend.If(self.z3_expr >= 0,
                                   self.z3_expr,
                                   -self.z3_expr),
            tracer=self.tracer)
        return result

    def __le__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicBool(self.z3_expr <= other.z3_expr, tracer=self.tracer)

    def __eq__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicBool(self.tracer.backend.Eq(self.z3_expr, other.z3_expr), tracer=self.tracer)

    def is_integer(self):
        """Returns True if the number equals its floor"""
        return SymbolicBool(
            self.tracer.backend.Eq(
                self.z3_expr,
                self.tracer.backend.ToInt(self.z3_expr)
            ),
            tracer=self.tracer
        )
        
    def __rsub__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicFloat(self.tracer.backend.Sub(other.z3_expr, self.z3_expr), tracer=self.tracer)
        
    def __add__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            return SymbolicFloat(self.concrete + other.concrete, tracer=self.tracer)
        return SymbolicFloat(self.tracer.backend.Add(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            return SymbolicFloat(self.concrete * other.concrete, tracer=self.tracer)
        return SymbolicFloat(self.tracer.backend.Mul(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __pow__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            return SymbolicFloat(self.concrete ** other.concrete, tracer=self.tracer)
        # For small non-negative integer exponents, expand to multiplication
        # Z3's ^ operator doesn't work well with Real base and Int exponent
        if hasattr(other, 'concrete') and other.concrete is not None:
            n = other.concrete
            if isinstance(n, int) and 0 <= n <= 10:
                if n == 0:
                    return SymbolicFloat(1.0, tracer=self.tracer)
                result = self.z3_expr
                for _ in range(n - 1):
                    result = self.tracer.backend.Mul(result, self.z3_expr)
                return SymbolicFloat(result, tracer=self.tracer)
        return SymbolicFloat(self.tracer.backend.Pow(self.z3_expr, other.z3_expr), tracer=self.tracer)

    def __rpow__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicFloat(self.tracer.backend.Pow(other.z3_expr, self.z3_expr), tracer=self.tracer)
    
    def __gt__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            return SymbolicBool(self.concrete > other.concrete, tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.GT(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __lt__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            return SymbolicBool(self.concrete < other.concrete, tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.LT(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __ge__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            return SymbolicBool(self.concrete >= other.concrete, tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.GE(self.z3_expr, other.z3_expr), tracer=self.tracer)

    def __truediv__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicFloat(self.z3_expr / other.z3_expr, tracer=self.tracer)
    
    def __rtruediv__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicFloat(other.z3_expr / self.z3_expr, tracer=self.tracer)

    def __str__(self):
        if self.concrete is not None:
            return str(self.concrete)
        return f"SymbolicFloat({self.name})"
    
    def __repr__(self):
        return self.__str__()

class SymbolicListIterator:
    _counter = 0
    
    def __init__(self, sym_list):
        self.tracer = sym_list.tracer
        self.sym_list = sym_list
        self.elementTyp = sym_list.elementTyp
        self.length = sym_list.__len__()
        self.used = False
        # Store the initial forall conditions count to restore later
        self.initial_forall_count = len(self.tracer.forall_conditions)
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.used:
            # Clear the forall conditions we added when iteration completes
            self.tracer.forall_conditions = self.tracer.forall_conditions[:self.initial_forall_count]
            raise StopIteration
            
        # Create position variable directly as a variable in forall
        pos_name = f'list_pos_{SymbolicListIterator._counter}'
        SymbolicListIterator._counter += 1
        self.tracer.backend.quantified_vars.add(pos_name)        
        pos_var = SymbolicInt(name=pos_name, tracer=self.tracer)
        
        # Add bounds constraints for position
        bounds = (pos_var.z3_expr >= 0) & (pos_var.z3_expr < self.length.z3_expr)
        # Get element at current position
        element_expr = self.tracer.backend.ListGet(self.sym_list.z3_expr, pos_var.z3_expr, self.tracer.backend.Type(self.elementTyp))
        result = make_symbolic_value(self.elementTyp, element_expr, tracer=self.tracer)
        
        # Add position variable to forall condition
        self.tracer.forall_conditions.append((pos_var.z3_expr, bounds))
        
        self.used = True
        return result

class SymbolicList:
    def __init__(self, value, elementTyp, name: Optional[str] = None, tracer: Optional[SymbolicTracer] = None):
        self.tracer = tracer or SymbolicTracer()
        self.elementTyp = elementTyp
        self.concrete = None
        if name is not None:
            self.z3_expr = self.tracer.backend.List(name, self.tracer.backend.Type(elementTyp))
        elif isinstance(value, list):
            self.concrete = value
            converted_values = []
            for elem in value:
                if hasattr(elem, 'z3_expr'):
                    converted_values.append(elem.z3_expr)
                else:
                    converted_values.append(elem)
            self.z3_expr = self.tracer.backend.ListVal(converted_values, self.tracer.backend.Type(elementTyp))
        else:
            self.z3_expr = value
        self.name = name

    def __eq__(self, other):
        if isinstance(other, (str, SymbolicList)):
            other = self.tracer.ensure_symbolic(other)
            if self.concrete is not None and other.concrete is not None:
                result = self.concrete == other.concrete
            else:
                result = self.tracer.backend.Eq(self.z3_expr, other.z3_expr)
            return SymbolicBool(result, tracer=self.tracer)
        else:
            return SymbolicBool(self.tracer.backend.BoolVal(False), tracer=self.tracer)

    def __ne__(self, other):
        eq = self.__eq__(other)
        if isinstance(eq, SymbolicBool):
            if eq.concrete is not None:
                return SymbolicBool(not eq.concrete, tracer=self.tracer)
            return SymbolicBool(self.tracer.backend.Not(eq.z3_expr), tracer=self.tracer)
        return not eq

    def __contains__(self, item):
        item = self.tracer.ensure_symbolic(item)
        if self.concrete is not None:
            if item.concrete is not None:
                return item.concrete in self.concrete
            else:
                return SymbolicBool(self.tracer.backend.Or(*[(item == x).z3_expr for x in self.concrete]), tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.ListContains(self.z3_expr, item.z3_expr, self.tracer.backend.Type(self.elementTyp)), tracer=self.tracer)


    def contains(self, item):
        return self.tracer.ensure_symbolic(self.__contains__(item))

    def __getitem__(self, key):
        if self.concrete is not None:
            if isinstance(key, slice):
                if (not isinstance(key.start, SymbolicInt) and 
                    not isinstance(key.stop, SymbolicInt) and 
                    not isinstance(key.step, SymbolicInt)):
                    return SymbolicList(self.concrete[key], self.elementTyp, tracer=self.tracer)
                return SymbolicSlice(self.concrete, key.start, key.stop, key.step, tracer=self.tracer)
            elif isinstance(key, SymbolicInt):
                # If we have a concrete value, use it directly
                if key.concrete is not None:
                    return self.concrete[key.concrete]

                # Add bounds check
                n = len(self)
                self.tracer.add_constraint(key < n)
                self.tracer.add_constraint(key >= -n)

                # Build an If expression to select the right value
                result = None
                for i, item in enumerate(self.concrete):
                    item = self.tracer.ensure_symbolic(item)
                    if result is None:
                        result = item
                    else:
                        result = self.tracer.ensure_symbolic(result)
                        result = make_symbolic_value(self.elementTyp,
                            self.tracer.backend.If(
                                self.tracer.backend.Or(key.z3_expr == i, key.z3_expr == -n+i),
                                item.z3_expr,
                                result.z3_expr
                            ),
                            tracer=self.tracer
                        )
                return result
            return self.concrete[key]
        if isinstance(key, slice):
            start = self.tracer.ensure_symbolic(key.start or 0)
            stop = self.tracer.ensure_symbolic(key.stop or -1)
            step = self.tracer.ensure_symbolic(key.step or 1)
            return SymbolicList(self.tracer.backend.ListSlice(self.z3_expr, start.z3_expr, stop.z3_expr, step.z3_expr, self.tracer.backend.Type(self.elementTyp)), self.elementTyp, tracer=self.tracer)
        else:
            key = self.tracer.ensure_symbolic(key)
            # Add length constraint to ensure the list has enough elements
            list_len = self.__len__()
            if hasattr(key, 'concrete') and key.concrete is not None:
                if key.concrete >= 0:
                    self.tracer.add_constraint(list_len > key)
                else:
                    self.tracer.add_constraint(list_len >= -key.concrete)
            else:
                # For symbolic index, add constraint for positive case
                # (list.get handles negative indices internally)
                self.tracer.add_constraint(list_len > key)
            return make_symbolic_value(self.elementTyp, self.tracer.backend.ListGet(self.z3_expr, key.z3_expr, self.tracer.backend.Type(self.elementTyp)), tracer=self.tracer)

    def __iter__(self):
        if self.concrete is not None:
            return iter(self.concrete)
        return SymbolicListIterator(self)

    def __len__(self):
        if self.concrete is not None:
            return len(self.concrete)
        return SymbolicInt(self.tracer.backend.ListLength(self.z3_expr, self.tracer.backend.Type(self.elementTyp)), tracer=self.tracer)

    def __add__(self, other):
        # Handle BoundedSymbolicList: return SymbolicList with concrete elements
        if isinstance(other, (BoundedSymbolicList, BoundedSymbolicSlice)):
            if self.concrete is not None:
                return SymbolicList(self.concrete + list(other), self.elementTyp, tracer=self.tracer)
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            return SymbolicList(self.concrete + other.concrete, self.elementTyp, tracer=self.tracer)
        return SymbolicList(self.tracer.backend.ListAppend(self.z3_expr, other.z3_expr, self.tracer.backend.Type(self.elementTyp)), self.elementTyp, tracer=self.tracer)

    def __radd__(self, other):
        # Handle BoundedSymbolicList: return SymbolicList with concrete elements
        if isinstance(other, (BoundedSymbolicList, BoundedSymbolicSlice)):
            if self.concrete is not None:
                return SymbolicList(list(other) + self.concrete, self.elementTyp, tracer=self.tracer)
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            return SymbolicList(other.concrete + self.concrete, self.elementTyp, tracer=self.tracer)
        return SymbolicList(self.tracer.backend.ListAppend(other.z3_expr, self.z3_expr, self.tracer.backend.Type(self.elementTyp)), self.elementTyp, tracer=self.tracer)

    def index(self, item):
        if self.concrete is None:
            item = self.tracer.ensure_symbolic(item)
            return SymbolicInt(self.tracer.backend.ListIndex(self.z3_expr, item.z3_expr, self.tracer.backend.Type(self.elementTyp)), tracer=self.tracer)
        # Return first index where item appears as SymbolicInt
        
        conditions = []
        for i, x in enumerate(self.concrete):
            eq = (x == item)  # This gives us a SymbolicBool
            conditions.append((eq.z3_expr, i))
        
        # Default result is -1 (not found)
        result_expr = self.tracer.backend.IntVal(-1)
        
        for condition, index in reversed(conditions):
            result_expr = self.tracer.backend.If(
                condition,
                self.tracer.backend.IntVal(index),
                result_expr
            )
        
        return SymbolicInt(result_expr, tracer=self.tracer)

    def count(self, item):
        """Count occurrences of item in list"""
        if self.concrete is not None:
            count = 0
            for x in self.concrete:
                eq = (x == item)  # This gives us a SymbolicBool
                if eq.concrete is not None:  # If we can evaluate it concretely
                    count += int(eq.concrete)
                else:
                    # For now, just return a symbolic int for the whole count
                    # This could be made more precise later
                    result = None
                    for x in self.concrete:
                        eq = (x == item)
                        if result is None:
                            result = SymbolicInt(
                                self.tracer.backend.If(
                                    eq.z3_expr,
                                    self.tracer.backend.IntVal(1),
                                    self.tracer.backend.IntVal(0)
                                ),
                                tracer=self.tracer
                            )
                        else:
                            result = result + SymbolicInt(
                                self.tracer.backend.If(
                                    eq.z3_expr,
                                    self.tracer.backend.IntVal(1),
                                    self.tracer.backend.IntVal(0)
                                ),
                                tracer=self.tracer
                            )
                    return result
            return SymbolicInt(count, tracer=self.tracer)
        item = self.tracer.ensure_symbolic(item)
        return SymbolicInt(self.tracer.backend.ListCount(self.z3_expr, item.z3_expr, self.tracer.backend.Type(self.elementTyp)), tracer=self.tracer)

    def __not__(self):
        if self.concrete is not None:
            return SymbolicBool(not self.concrete, tracer=self.tracer)
        return self != []

    def append(self, item):
        """Append item to list (mutates in place for concrete lists)"""
        if self.concrete is not None:
            self.concrete.append(self.tracer.ensure_symbolic(item))
        else:
            raise NotImplementedError("Cannot append to fully symbolic list")

    def pop(self, index=-1):
        """Remove and return item at index (mutates in place for concrete lists)"""
        if self.concrete is not None:
            return self.concrete.pop(index)
        else:
            raise NotImplementedError("Cannot pop from fully symbolic list")

    def remove(self, item):
        """Remove first occurrence of item (mutates in place for concrete lists)"""
        if self.concrete is not None:
            # Need custom removal since symbolic == returns SymbolicBool
            for i, elem in enumerate(self.concrete):
                eq = elem == item
                # Check if equality is concretely true
                if hasattr(eq, 'concrete') and eq.concrete:
                    self.concrete.pop(i)
                    return
                elif isinstance(eq, bool) and eq:
                    self.concrete.pop(i)
                    return
            raise ValueError("list.remove(x): x not in list")
        else:
            raise NotImplementedError("Cannot remove from fully symbolic list")


class BoundedSymbolicList:
    """A symbolic list with known size, using individual variables instead of recursive (List Int).

    This encoding is much faster for SMT solvers because it avoids quantifiers over list indices
    and recursive list helper functions.
    """
    def __init__(self, size: int, elementTyp, name: str, tracer: Optional['SymbolicTracer'] = None):
        self.tracer = tracer or SymbolicTracer()
        self.elementTyp = elementTyp
        self.size = size
        self.name = name
        self.concrete = None
        self.z3_expr = None  # Compatibility with SymbolicList (not used, but needed for type checks)

        # Create bounded list with individual variables
        self.bounded_vars = self.tracer.backend.BoundedList(
            name, size, self.tracer.backend.Type(elementTyp)
        )

    def __eq__(self, other):
        if isinstance(other, (list, SymbolicList)):
            other_list = self.tracer.ensure_symbolic(other) if isinstance(other, list) else other
            if hasattr(other_list, 'concrete') and other_list.concrete is not None:
                # Compare element by element
                if len(other_list.concrete) != self.size:
                    return SymbolicBool(False, tracer=self.tracer)
                constraints = []
                for i, item in enumerate(other_list.concrete):
                    item = self.tracer.ensure_symbolic(item)
                    constraints.append(self.tracer.backend.Eq(
                        self.bounded_vars.get(i),
                        item.z3_expr
                    ))
                if constraints:
                    return SymbolicBool(self.tracer.backend.And(*constraints), tracer=self.tracer)
                return SymbolicBool(True, tracer=self.tracer)
        return SymbolicBool(False, tracer=self.tracer)

    def __ne__(self, other):
        eq = self.__eq__(other)
        if isinstance(eq, SymbolicBool):
            if eq.concrete is not None:
                return SymbolicBool(not eq.concrete, tracer=self.tracer)
            return SymbolicBool(self.tracer.backend.Not(eq.z3_expr), tracer=self.tracer)
        return not eq

    def __contains__(self, item):
        item = self.tracer.ensure_symbolic(item)
        # Build OR of equality checks for each element
        checks = [self.tracer.backend.Eq(self.bounded_vars.get(i), item.z3_expr)
                  for i in range(self.size)]
        if checks:
            return SymbolicBool(self.tracer.backend.Or(*checks), tracer=self.tracer)
        return SymbolicBool(False, tracer=self.tracer)

    def contains(self, item):
        return self.tracer.ensure_symbolic(self.__contains__(item))

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Handle slicing
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else self.size
            step = key.step if key.step is not None else 1

            # If all indices are concrete, return a BoundedSymbolicSlice
            if all(isinstance(x, int) for x in [start, stop, step]):
                return BoundedSymbolicSlice(self, start, stop, step)
            # For symbolic indices, return a SymbolicSlice-like object
            return BoundedSymbolicSlice(self, start, stop, step)

        # Handle indexing
        if isinstance(key, int):
            # Concrete index
            if key < 0:
                key = self.size + key
            return make_symbolic_value(self.elementTyp, self.bounded_vars.get(key), tracer=self.tracer)
        elif isinstance(key, SymbolicInt):
            if key.concrete is not None:
                idx = key.concrete
                if idx < 0:
                    idx = self.size + idx
                return make_symbolic_value(self.elementTyp, self.bounded_vars.get(idx), tracer=self.tracer)
            else:
                # Symbolic index - use ITE chain
                return make_symbolic_value(self.elementTyp, self.bounded_vars.get(key.z3_expr), tracer=self.tracer)
        else:
            raise ValueError(f"Unsupported index type: {type(key)}")

    def __iter__(self):
        # Return an iterator over the elements
        for i in range(self.size):
            yield make_symbolic_value(self.elementTyp, self.bounded_vars.get(i), tracer=self.tracer)

    def __len__(self):
        return SymbolicInt(self.size, tracer=self.tracer)

    def __add__(self, other):
        # For now, adding bounded lists creates a regular symbolic list
        other = self.tracer.ensure_symbolic(other)
        my_elems = [make_symbolic_value(self.elementTyp, self.bounded_vars.get(i), tracer=self.tracer)
                    for i in range(self.size)]
        if hasattr(other, 'concrete') and other.concrete is not None:
            return SymbolicList(my_elems + list(other.concrete), self.elementTyp, tracer=self.tracer)
        raise NotImplementedError("Cannot add bounded list to fully symbolic list")

    def __radd__(self, other):
        other = self.tracer.ensure_symbolic(other)
        my_elems = [make_symbolic_value(self.elementTyp, self.bounded_vars.get(i), tracer=self.tracer)
                    for i in range(self.size)]
        if isinstance(other, list):
            # Raw list on left side
            return SymbolicList(list(other) + my_elems, self.elementTyp, tracer=self.tracer)
        if hasattr(other, 'concrete') and other.concrete is not None:
            return SymbolicList(list(other.concrete) + my_elems, self.elementTyp, tracer=self.tracer)
        raise NotImplementedError("Cannot add fully symbolic list to bounded list")

    def index(self, item):
        item = self.tracer.ensure_symbolic(item)
        # Build nested ITE: if e0 == item then 0 else if e1 == item then 1 else ...
        result = self.tracer.backend.IntVal(-1)  # Not found
        for i in range(self.size - 1, -1, -1):
            result = self.tracer.backend.If(
                self.tracer.backend.Eq(self.bounded_vars.get(i), item.z3_expr),
                self.tracer.backend.IntVal(i),
                result
            )
        return SymbolicInt(result, tracer=self.tracer)

    def count(self, item):
        item = self.tracer.ensure_symbolic(item)
        # Sum of (1 if ei == item else 0) for all i
        terms = []
        for i in range(self.size):
            term = self.tracer.backend.If(
                self.tracer.backend.Eq(self.bounded_vars.get(i), item.z3_expr),
                self.tracer.backend.IntVal(1),
                self.tracer.backend.IntVal(0)
            )
            terms.append(term)
        if terms:
            return SymbolicInt(self.tracer.backend.Add(*terms), tracer=self.tracer)
        return SymbolicInt(0, tracer=self.tracer)

    def __not__(self):
        return SymbolicBool(self.size == 0, tracer=self.tracer)


class BoundedSymbolicSlice:
    """A slice of a BoundedSymbolicList - used for prefix sums etc."""
    def __init__(self, bounded_list, start, stop, step=1):
        self.bounded_list = bounded_list
        self.tracer = bounded_list.tracer
        self.start = start
        self.stop = stop
        self.step = step

    def sum(self):
        """Sum elements in the slice - optimized for prefix sums"""
        # Handle concrete start
        start_val = self.start if isinstance(self.start, int) else (
            self.start.concrete if hasattr(self.start, 'concrete') and self.start.concrete is not None else None
        )

        if start_val is not None and start_val == 0:
            # This is a prefix sum li[:stop]
            stop = self.stop
            if isinstance(stop, int):
                return SymbolicInt(self.bounded_list.bounded_vars.prefix_sum(stop), tracer=self.tracer)
            elif isinstance(stop, SymbolicInt):
                if stop.concrete is not None:
                    return SymbolicInt(self.bounded_list.bounded_vars.prefix_sum(stop.concrete), tracer=self.tracer)
                else:
                    return SymbolicInt(self.bounded_list.bounded_vars.prefix_sum(stop.z3_expr), tracer=self.tracer)

        # General case: build sum with conditions
        terms = []
        for i in range(self.bounded_list.size):
            # Check if index i is in range [start, stop)
            in_range = self.tracer.backend.And(
                self.tracer.backend.GE(
                    self.tracer.backend.IntVal(i),
                    self.start.z3_expr if isinstance(self.start, SymbolicInt) else self.tracer.backend.IntVal(self.start)
                ),
                self.tracer.backend.LT(
                    self.tracer.backend.IntVal(i),
                    self.stop.z3_expr if isinstance(self.stop, SymbolicInt) else self.tracer.backend.IntVal(self.stop)
                )
            )
            term = self.tracer.backend.If(
                in_range,
                self.bounded_list.bounded_vars.get(i),
                self.tracer.backend.IntVal(0)
            )
            terms.append(term)

        if terms:
            return SymbolicInt(self.tracer.backend.Add(*terms), tracer=self.tracer)
        return SymbolicInt(0, tracer=self.tracer)

    def __iter__(self):
        # Iterate over elements in the slice
        start = self.start if isinstance(self.start, int) else 0
        stop = self.stop if isinstance(self.stop, int) else self.bounded_list.size
        step = self.step if isinstance(self.step, int) else 1

        for i in range(start, min(stop, self.bounded_list.size), step):
            yield make_symbolic_value(
                self.bounded_list.elementTyp,
                self.bounded_list.bounded_vars.get(i),
                tracer=self.tracer
            )

    def get_slice(self):
        """Return the slice as a SymbolicList (for compatibility)"""
        elements = list(self)
        return SymbolicList(elements, self.bounded_list.elementTyp, tracer=self.tracer)

    def __add__(self, other):
        """Concatenate two slices or a slice with another list"""
        my_elems = list(self)
        if isinstance(other, BoundedSymbolicSlice):
            other_elems = list(other)
            return SymbolicList(my_elems + other_elems, self.bounded_list.elementTyp, tracer=self.tracer)
        if isinstance(other, BoundedSymbolicList):
            other_elems = list(other)
            return SymbolicList(my_elems + other_elems, self.bounded_list.elementTyp, tracer=self.tracer)
        if isinstance(other, list):
            return SymbolicList(my_elems + other, self.bounded_list.elementTyp, tracer=self.tracer)
        if hasattr(other, 'concrete') and other.concrete is not None:
            return SymbolicList(my_elems + list(other.concrete), self.bounded_list.elementTyp, tracer=self.tracer)
        raise NotImplementedError(f"Cannot add BoundedSymbolicSlice to {type(other)}")

    def __radd__(self, other):
        """Handle other + slice"""
        my_elems = list(self)
        if isinstance(other, list):
            return SymbolicList(other + my_elems, self.bounded_list.elementTyp, tracer=self.tracer)
        if hasattr(other, 'concrete') and other.concrete is not None:
            return SymbolicList(list(other.concrete) + my_elems, self.bounded_list.elementTyp, tracer=self.tracer)
        raise NotImplementedError(f"Cannot add {type(other)} to BoundedSymbolicSlice")


class SymbolicStrIterator:
    _counter = 0
    
    def __init__(self, sym_str):
        self.tracer = sym_str.tracer
        self.sym_str = sym_str
        self.length = sym_str.__len__()
        self.used = False
        # Store the initial forall conditions count to restore later
        self.initial_forall_count = len(self.tracer.forall_conditions)
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.used:
            # Clear the forall conditions we added when iteration completes
            self.tracer.forall_conditions = self.tracer.forall_conditions[:self.initial_forall_count]
            raise StopIteration
            
        # Create position variable directly as a Z3 variable in forall
        pos_name = f'str_pos_{SymbolicStrIterator._counter}'
        SymbolicStrIterator._counter += 1
        self.tracer.backend.quantified_vars.add(pos_name)        
        pos_var = SymbolicInt(name=pos_name, tracer=self.tracer)
        
        # Add bounds constraints for position
        bounds = (pos_var.z3_expr >= 0) & (pos_var.z3_expr < self.length.z3_expr)
        
        # Get character at current position
        result = SymbolicStr(
            self.tracer.backend.StrIndex(self.sym_str.z3_expr, pos_var.z3_expr),
            tracer=self.tracer
        )
        
        # Add position variable to forall condition
        self.tracer.forall_conditions.append((pos_var.z3_expr, bounds))
        
        self.used = True
        return result

class SymbolicStr:
    def __init__(self, value: Optional[Any] = None, name: Optional[str] = None, tracer: Optional[SymbolicTracer] = None):
        self.tracer = tracer
        self.concrete = None
        if name is not None:
            self.z3_expr = self.tracer.backend.String(name)
        elif isinstance(value, str):
            self.concrete = value
            self.z3_expr = self.tracer.backend.StringVal(value)
        else:
            self.z3_expr = value
        self.name = name

    def index(self, sub, start=0):
        [sub, start] = [self.tracer.ensure_symbolic(x) for x in [sub, start]]
        if all(x.concrete is not None for x in [self, sub, start]):
            return SymbolicInt(self.concrete.index(sub.concrete, start.concrete))
        return SymbolicInt(self.tracer.backend.StrIndexOf(self.z3_expr, sub.z3_expr, start.z3_expr), tracer=self.tracer)

    def join(self, ss):
        """Join a list of strings with this string as separator"""
        # Handle case where ss is a SymbolicList
        if isinstance(ss, SymbolicList):
            if ss.concrete is not None:
                # If list has concrete values, use those
                elements = [self.tracer.ensure_symbolic(x) for x in ss.concrete]
                element_exprs = [elem.z3_expr for elem in elements]
            else:
                # For fully symbolic list, we need a special handling
                # This will depend on the backend implementation details
                return SymbolicStr(
                    self.tracer.backend.StrJoin(self.z3_expr, ss.z3_expr),
                    tracer=self.tracer
                )
        else:
            # Try to handle iterable objects that aren't SymbolicList
            try:
                elements = [self.tracer.ensure_symbolic(x) for x in ss]
                element_exprs = [elem.z3_expr for elem in elements]
            except Exception as e:
                raise ValueError(f"Cannot join items in {type(ss)}: {e}")

        # Create a StrList and then join it
        str_list = self.tracer.backend.StrList(element_exprs)
        return SymbolicStr(
            self.tracer.backend.StrJoin(self.z3_expr, str_list),
            tracer=self.tracer
        )

    def __lt__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            return SymbolicBool(self.concrete < other.concrete, tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.StrLT(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __gt__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            return SymbolicBool(self.concrete > other.concrete, tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.StrLT(other.z3_expr, self.z3_expr), tracer=self.tracer)

    def __le__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return self.__lt__(other).__or__(self.__eq__(other))
    
    def __ge__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            return SymbolicBool(self.concrete >= other.concrete, tracer=self.tracer)
        return other.__le__(self)

    def __hash__(self):
        if self.concrete is not None:
            return hash(self.concrete)
        raise ValueError("Symbolic hash not yet implemented")

    def split(self, sep=None):
        """Split string into list of strings"""
        if self.concrete is not None:
            sep_c = sep
            if isinstance(sep, SymbolicStr):
                if sep.concrete is not None:
                    sep_c = sep.concrete
            if sep_c is None or isinstance(sep_c, str):
                parts = self.concrete.split(sep_c)
                return SymbolicList([SymbolicStr(p, tracer=self.tracer) for p in parts], str, tracer=self.tracer)
        
        if sep is None:
            sep = " "  # Default separator is whitespace
        sep = self.tracer.ensure_symbolic(sep)
        result = self.tracer.backend.StrSplit(self.z3_expr, sep.z3_expr)
        return SymbolicList(result, str, tracer=self.tracer)

    def __contains__(self, item):
        item = self.tracer.ensure_symbolic(item)
        if self.concrete is not None and item.concrete is not None:
            return item.concrete in self.concrete
        return self.tracer.backend.StrContains(self.z3_expr, item.z3_expr)

    def contains(self, item):
        return SymbolicBool(self.__contains__(item), tracer=self.tracer)

    def __getitem__(self, key):
        if self.concrete is None:
            if isinstance(key, slice):
                step = key.step or 1
                if step == -1 and key.start is None and key.stop is None:
                    # special case
                    return SymbolicStr(self.tracer.backend.StrReverse(self.z3_expr), tracer=self.tracer)
                elif step == 1:
                    start = key.start if key.start else 0
                    stop = key.stop if key.stop else self.__len__()
                    start = self.tracer.ensure_symbolic(start)
                    stop = self.tracer.ensure_symbolic(stop)
                    return SymbolicStr(self.tracer.backend.StrSubstr(self.z3_expr, start.z3_expr, stop.z3_expr), tracer=self.tracer)
                else:
                    raise ValueError("Slicing on symbolic strings not fully implemented.")
            key = self.tracer.ensure_symbolic(key)
            # Add bounds check: for positive index, len > index; for negative, len >= -index
            str_len = self.__len__()
            if hasattr(key, 'concrete') and key.concrete is not None:
                if key.concrete >= 0:
                    self.tracer.add_constraint(str_len > key)
                else:
                    self.tracer.add_constraint(str_len >= -key.concrete)
            else:
                # Symbolic index - add constraint for both cases
                self.tracer.add_constraint(str_len > key)
            return SymbolicStr(self.tracer.backend.StrIndex(self.z3_expr, key.z3_expr), tracer=self.tracer)
        if isinstance(key, slice):
            if (not isinstance(key.start, SymbolicInt) and 
                not isinstance(key.stop, SymbolicInt) and 
                not isinstance(key.step, SymbolicInt)):
                return SymbolicStr(self.concrete[key], tracer=self.tracer)
            return SymbolicSlice(self.concrete, key.start, key.stop, key.step, tracer=self.tracer)
        elif isinstance(key, SymbolicInt):
            return SymbolicSlice(self.concrete, key, key+1, None, tracer=self.tracer)
        return SymbolicStr(self.concrete[key], tracer=self.tracer)

    def __iter__(self):
        if self.concrete is not None:
            return iter(self.concrete)
        return SymbolicStrIterator(self)

    def __len__(self):
        if self.concrete is not None:
            return SymbolicInt(len(self.concrete), tracer=self.tracer)
        else:
            return SymbolicInt(self.tracer.backend.StrLen(self.z3_expr), tracer=self.tracer)

    def count(self, sub: str) -> 'SymbolicInt':
        """Count occurrences of substring in string"""
        if not isinstance(sub, (str, SymbolicStr)):
            raise TypeError(f"Can't count occurrences of {type(sub)}")
            
        if isinstance(sub, str):
            sub = SymbolicStr(sub, tracer=self.tracer)
            
        # Create a symbolic integer for the count
        result = SymbolicInt(self.tracer.backend.StrCount(self.z3_expr, sub.z3_expr), tracer=self.tracer)
        
        # Add constraint that count is non-negative
        self.tracer.add_constraint(result.z3_expr >= 0)
        
        return result

    def __str__(self):
        return self.concrete

    def __repr__(self):
        return f"SymbolicStr({self.concrete!r}, tracer={self.tracer!r})"

    def __add__(self, other):
        if isinstance(other, SymbolicStr):
            return SymbolicStr(
                self.tracer.backend.StrConcat(self.z3_expr, other.z3_expr),
                tracer=self.tracer
            )
        elif isinstance(other, str):
            return SymbolicStr(
                self.tracer.backend.StrConcat(
                    self.z3_expr,
                    self.tracer.backend.StringVal(other)
                ),
                tracer=self.tracer
            )
        elif isinstance(other, SymbolicSlice):
            return self + other.get_slice()
        raise ValueError("Not implemented: __add__")

    def __radd__(self, other):
        if isinstance(other, str):
            return SymbolicStr(
                self.tracer.backend.StrConcat(
                    self.tracer.backend.StringVal(other),
                    self.z3_expr
                ),
                tracer=self.tracer
            )
        raise ValueError("Not implemented: __radd__")

    def __mul__(self, other):
        """String repetition: str * n"""
        other = self.tracer.ensure_symbolic(other)
        return SymbolicStr(self.tracer.backend.StrMul(self.z3_expr, other.z3_expr), tracer=self.tracer)

    def __rmul__(self, other):
        """String repetition: n * str"""
        return self.__mul__(other)

    # For comparison operations
    def __eq__(self, other):
        if isinstance(other, (str, SymbolicStr)):
            other = self.tracer.ensure_symbolic(other)
            if self.concrete is not None and other.concrete is not None:
                result = self.concrete == other.concrete
            else:
                result = self.tracer.backend.Eq(self.z3_expr, other.z3_expr)
            return SymbolicBool(result, tracer=self.tracer)
        else:
            return SymbolicBool(self.tracer.backend.BoolVal(False), tracer=self.tracer)

    def __ne__(self, other):
        eq = self.__eq__(other)
        if isinstance(eq, SymbolicBool):
            if eq.concrete is not None:
                return SymbolicBool(not eq.concrete, tracer=self.tracer)
            return SymbolicBool(self.tracer.backend.Not(eq.z3_expr), tracer=self.tracer)
        return not eq

    def __and__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicBool(self.tracer.backend.And(truthy(self).z3_expr, truthy(other).z3_expr), tracer=self.tracer)
    
    def __or__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicBool(self.tracer.backend.Or(truthy(self).z3_expr, truthy(other).z3_expr), tracer=self.tracer)

    def startswith(self, prefix):
        prefix = self.tracer.ensure_symbolic(prefix)
        return SymbolicBool(self.tracer.backend.StrPrefixOf(prefix.z3_expr, self.z3_expr), tracer=self.tracer)

    def isupper(self):
        if self.concrete is not None:
            return SymbolicBool(self.concrete.isupper(), tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.IsUpper(self.z3_expr), tracer=self.tracer)

    def islower(self):
        if self.concrete is not None:
            return SymbolicBool(self.concrete.islower(), tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.IsLower(self.z3_expr), tracer=self.tracer)

    def upper(self):
        if self.concrete is not None:
            return SymbolicStr(self.concrete.upper(), tracer=self.tracer)
        return SymbolicStr(self.tracer.backend.StrUpper(self.z3_expr), tracer=self.tracer)

    def lower(self):
        if self.concrete is not None:
            return SymbolicStr(self.concrete.lower(), tracer=self.tracer)
        return SymbolicStr(self.tracer.backend.StrLower(self.z3_expr), tracer=self.tracer)

    def isdigit(self):
        if self.concrete is not None:
            return SymbolicBool(self.concrete.isdigit(), tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.StrIsDigit(self.z3_expr), tracer=self.tracer)

    def isalpha(self):
        if self.concrete is not None:
            return SymbolicBool(self.concrete.isalpha(), tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.StrIsAlpha(self.z3_expr), tracer=self.tracer)

    def swapcase(self):
        if self.concrete is not None:
            return SymbolicStr(self.concrete.swapcase(), tracer=self.tracer)
        return SymbolicStr(self.tracer.backend.SwapCase(self.z3_expr), tracer=self.tracer)

    def translate(self, subs):
        if self.concrete is not None:
            concrete_subs = dict([(as_concrete(k), as_concrete(v)) for k,v in subs.items()])
            print('concrete_subs', concrete_subs)
            return SymbolicStr(self.concrete.translate(concrete_subs), tracer=self.tracer)
        raise ValueError("translate not implemented for symbolic strings")

    def replace(self, a, b):
        a = self.tracer.ensure_symbolic(a)
        b = self.tracer.ensure_symbolic(b)
        if self.concrete is not None and a.concrete is not None and b.concrete is not None:
            return SymbolicStr(self.concrete.replace(a.concrete, b.concrete), tracer=self.tracer)
        return SymbolicStr(self.tracer.backend.StrReplace(self.z3_expr, a.z3_expr, b.z3_expr), tracer=self.tracer)

    def __not__(self):
        if self.concrete is not None:
            return SymbolicBool(not self.concrete, tracer=self.tracer)
        return self != ""

class SymbolicSlice:
    def __init__(self, concrete_seq, start, end, step=None, tracer: Optional[SymbolicTracer] = None):
        assert concrete_seq is not None
        self.concrete = concrete_seq  # Can be str or list
        self.start = start
        self.end = end
        self.step = step
        self.tracer = tracer

    def __add__(self, other):
        if isinstance(other, SymbolicSlice):
            other = other.get_slice()
        return self.get_slice() + other

    def __radd__(self, other):
        return other + self.get_slice()

    def __and__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicBool(self.tracer.backend.And(truthy(self).z3_expr, truthy(other).z3_expr), tracer=self.tracer)
    
    def __or__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicBool(self.tracer.backend.Or(truthy(self).z3_expr, truthy(other).z3_expr), tracer=self.tracer)

    def __getitem__(self, key):
        """Handle indexing into the slice"""
        if isinstance(key, SymbolicInt):
            # Build an If expression to select the right value
            result = None
            for i, item in enumerate(self.concrete):
                if result is None:
                    result = SymbolicInt(
                        self.tracer.backend.If(
                            key.z3_expr == i,
                            item.z3_expr,
                            self.tracer.backend.IntVal(0)  # default value
                        ),
                        tracer=self.tracer
                    )
                else:
                    result = SymbolicInt(
                        self.tracer.backend.If(
                            key.z3_expr == i,
                            item.z3_expr,
                            result.z3_expr
                        ),
                        tracer=self.tracer
                    )
            return result
        return self.concrete[key]

    def __iter__(self):
        # If all indices are concrete, use concrete slice
        if (isinstance(self.start, (int, type(None))) and 
            isinstance(self.end, (int, type(None))) and 
            isinstance(self.step, (int, type(None)))):
            start = self.start if self.start is not None else 0
            end = self.end if self.end is not None else len(self.concrete)
            step = self.step if self.step is not None else 1
            return iter(self.concrete[start:end:step])
        
        # For symbolic indices, return only up to the symbolic end
        if isinstance(self.end, SymbolicInt):
            # For now, just use the concrete value if available
            if hasattr(self.end, 'concrete') and self.end.concrete is not None:
                return iter(self.concrete[:self.end.concrete])
        
        # Default to full sequence if we can't determine bounds
        return iter(self.concrete)

    def __eq__(self, other):
        return self.get_slice().__eq__(other)

    def __ne__(self, other):
        return self.get_slice().__ne__(other)

    def __lt__(self, other):
        if isinstance(other, SymbolicSlice):
            other = other.get_slice()
        return self.get_slice().__lt__(other)

    def __le__(self, other):
        if isinstance(other, SymbolicSlice):
            other = other.get_slice()
        return self.get_slice().__le__(other)

    def __gt__(self, other):
        if isinstance(other, SymbolicSlice):
            other = other.get_slice()
        return self.get_slice().__gt__(other)

    def __ge__(self, other):
        if isinstance(other, SymbolicSlice):
            other = other.get_slice()
        return self.get_slice().__ge__(other)

    def upper(self):
        return self.get_slice().upper()

    def count(self, sub):
        """Count occurrences in sliced sequence"""
        # Get the sliced sequence
        sliced = self.get_slice()
        # Use the appropriate count method
        return sliced.count(sub)
    
    def sum(self):
        """Sum elements in sliced sequence (for numeric lists)"""
        if isinstance(self.concrete, list):
            # Check if all elements are numeric (int, float, or SymbolicInt)
            all_numeric = all(isinstance(x, (int, float, SymbolicInt)) for x in self.concrete)
            if all_numeric:
                # Build a conditional sum over the range
                start = self.start if self.start is not None else 0
                end = self.end if self.end is not None else len(self.concrete)
                
                start_expr = start.z3_expr if isinstance(start, SymbolicInt) else self.tracer.backend.IntVal(start)
                end_expr = end.z3_expr if isinstance(end, SymbolicInt) else self.tracer.backend.IntVal(end)
                
                # Build nested If expressions for the sum
                result_sum = self.tracer.backend.IntVal(0)
                for i in range(len(self.concrete)):
                    in_range = self.tracer.backend.And(
                        self.tracer.backend.LE(start_expr, self.tracer.backend.IntVal(i)),
                        self.tracer.backend.LT(self.tracer.backend.IntVal(i), end_expr)
                    )
                    # Get the value at index i
                    elem = self.concrete[i]
                    if isinstance(elem, SymbolicInt):
                        elem_expr = elem.z3_expr
                    else:
                        elem_expr = self.tracer.backend.IntVal(elem)
                    
                    result_sum = self.tracer.backend.Add(
                        result_sum,
                        self.tracer.backend.If(
                            in_range,
                            elem_expr,
                            self.tracer.backend.IntVal(0)
                        )
                    )
                return SymbolicInt(result_sum, tracer=self.tracer)
        
        # For non-numeric or non-list, just use get_slice and regular sum
        return sum(self.get_slice())

    def get_slice(self):
        """Convert slice to appropriate symbolic type (SymbolicStr or SymbolicList)"""
        # If all indices are concrete, just return the concrete slice
        if (isinstance(self.start, (int, type(None))) and 
            isinstance(self.end, (int, type(None))) and 
            isinstance(self.step, (int, type(None)))):
            start = self.start if self.start is not None else 0
            end = self.end if self.end is not None else len(self.concrete)
            step = self.step if self.step is not None else 1
            result = self.concrete[start:end:step]
            return (SymbolicStr(result, tracer=self.tracer) if isinstance(self.concrete, str)
                   else SymbolicList(result, str, tracer=self.tracer))
        
        # For symbolic indices, use substring operation
        start = self.start if self.start is not None else 0
        end = self.end if self.end is not None else len(self.concrete)
        
        if isinstance(self.concrete, str):
            # Use Z3's Extract for strings
            start_expr = start.z3_expr if isinstance(start, SymbolicInt) else self.tracer.backend.IntVal(start)
            length_expr = (end.z3_expr if isinstance(end, SymbolicInt) else self.tracer.backend.IntVal(end)) - start_expr
            return SymbolicStr(
                self.tracer.backend.StrSubstr(self.tracer.backend.StringVal(self.concrete), start_expr, length_expr, variant="str.substr"),
                tracer=self.tracer
            )
        else:
            # For lists with symbolic indices, we can't easily create a proper list
            # For now, return self (the SymbolicSlice) which has methods to work with it
            # This is a placeholder - proper symbolic list slicing would require more work
            return self

class SymbolicRangeIterator:
    def __init__(self, sym_range):
        self.tracer = sym_range.tracer
        self.sym_range = sym_range
        # Create fresh variable for the iterator
        var_name = f'i_{SymbolicRange._counter}'
        SymbolicRange._counter += 1
        # Mark as quantified so it's not declared in the header
        self.tracer.backend.quantified_vars.add(var_name)
        self.var = SymbolicInt(name=var_name, tracer=sym_range.tracer)
        self.used = False
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.used:
            raise StopIteration
        self.used = True
        # Don't add to forall_conditions here - let sym_all/sym_any handle it
        return self.var
    
    def get_bounds(self):
        """Get bounds constraints including step if present"""
        bounds = (self.var >= self.sym_range.start).__and__(
                 self.var < self.sym_range.end)
        
        if self.sym_range.step is not None:
            # i = start + k * step for some k >= 0
            k = SymbolicInt(name=f'k_{SymbolicRange._counter}', tracer=self.sym_range.tracer)
            SymbolicRange._counter += 1
            step_constraint = (self.var == self.sym_range.start + k * self.sym_range.step).__and__(
                             k >= 0).__and__(
                             k < (self.sym_range.end - self.sym_range.start) / self.sym_range.step)
            bounds = bounds.__and__(step_constraint)
            
        return bounds

class SymbolicRange:
    _counter = 0

    def __init__(self, start, end, step=None, tracer=None):
        self.start = start if isinstance(start, SymbolicInt) else SymbolicInt(start, tracer=tracer)
        self.end = end if isinstance(end, SymbolicInt) else SymbolicInt(end, tracer=tracer)
        self.step = step if step is None else (step if isinstance(step, SymbolicInt) else SymbolicInt(step, tracer=tracer))
        self.tracer = tracer or self.start.tracer

    def __iter__(self):
        return SymbolicRangeIterator(self)

    def __contains__(self, item):
        item = self.tracer.ensure_symbolic(item)
            
        result = (item >= self.start) and (item < self.end)
        if self.step is not None:
            # item = start + k * step for some k
            k = SymbolicInt('k', tracer=self.tracer)
            result = result and (item == self.start + k * self.step)
        return result

    def contains(self, item):
        return SymbolicBool(self.__contains__(item), tracer=self.tracer)

    def __len__(self):
        if self.step is None or self.step == 1:
            return self.end - self.start
        return (self.end - self.start) / self.step

def fresh_symbolic(var):
    typ = type(var).__name__.lower().replace('symbolic', '')
    return make_symbolic(typ, var.name, var.tracer)

def make_symbolic(typ: Type, name: str, tracer: Optional[SymbolicTracer] = None, size: Optional[int] = None) -> Any:
    """Create a new symbolic variable of given type.

    Args:
        typ: The type of the symbolic variable
        name: Variable name
        tracer: The symbolic tracer
        size: For list types, optional size bound. If provided, uses BoundedSymbolicList
              which is much faster for SMT solving.
    """
    if typ == int or typ == 'int':
        sym = SymbolicInt(name=name, tracer=tracer)
    elif typ == bool or typ == 'bool':
        sym = SymbolicBool(name=name, tracer=tracer)
    elif typ == float or typ == 'float':
        sym = SymbolicFloat(name=name, tracer=tracer)
    elif typ == str or typ == 'str':
        sym = SymbolicStr(name=name, tracer=tracer)
    elif typ == list[str] or typ == 'List[str]':
        if size is not None:
            sym = BoundedSymbolicList(size, str, name=name, tracer=tracer)
        else:
            sym = SymbolicList(None, str, name=name, tracer=tracer)
    elif typ == list[int] or typ == 'List[int]':
        if size is not None:
            sym = BoundedSymbolicList(size, int, name=name, tracer=tracer)
        else:
            sym = SymbolicList(None, int, name=name, tracer=tracer)
    else:
        raise ValueError(f"Unsupported symbolic type: {typ}")
    return sym

def make_symbolic_value(typ: Type, v: Any, tracer: Optional[SymbolicTracer] = None) -> Any:
    """Create a new symbolic value of given type"""
    if typ == int or typ == 'int':
        sym = SymbolicInt(v, tracer=tracer)
    elif typ == bool or typ == 'bool':
        sym = SymbolicBool(v, tracer=tracer)
    elif typ == float or typ == 'float':
        sym = SymbolicFloat(v, tracer=tracer)
    elif typ == str or typ == 'str':
        sym = SymbolicStr(v, tracer=tracer)
    elif typ == list[str] or typ == 'List[str]':
        sym = SymbolicList(v, str, tracer=tracer)
    elif typ == list[int] or typ == 'List[int]':
        sym = SymbolicList(v, int, tracer=tracer)
    else:
        raise ValueError(f"Unsupported symbolic type: {typ}")
    return sym

def truthy(x):
    if isinstance(x, SymbolicBool):
        return x
    elif isinstance(x, SymbolicInt):
        return x != 0
    elif isinstance(x, SymbolicStr):
        return x != ""
    elif isinstance(x, SymbolicSlice):
        return x != "" # TODO for list too
    else:
        return x

def as_concrete(x):
    if hasattr(x, 'concrete') and x.concrete is not None:
        return x.concrete
    else:
        return x
