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

        condition = self.ensure_symbolic(condition)
        self.path_conditions.append(
            condition.z3_expr if branch_val 
            else self.backend.Not(condition.z3_expr)
        )
        return branch_val

    def add_constraint(self, constraint):
        """Add constraint with optional LLM refinement"""
        if isinstance(constraint, (SymbolicInt, SymbolicBool)):
            constraint = truthy(constraint).z3_expr
            
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
        return  model[var.name]
    
    def ensure_symbolic(self, other):
        if isinstance(other, bool):
            return SymbolicBool(other, tracer=self)
        if isinstance(other, int):
            return SymbolicInt(other, tracer=self)
        if isinstance(other, float):
            return SymbolicFloat(other, tracer=self)
        if isinstance(other, str):
            return SymbolicStr(other, tracer=self)
        if isinstance(other, SymbolicSlice):
            return other.get_slice()
        return other

class SymbolicBool:
    """Wrapper class for symbolic boolean expressions"""
    def __init__(self, value, tracer: Optional[SymbolicTracer] = None):
        """Initialize boolean expression with optional tracer"""
        self.tracer = tracer or SymbolicTracer()
        self.z3_expr = value
        if isinstance(value, bool):
            self.concrete = value
        # TODO: weird that we need this
        elif str(value) in ['true', 'false']:
            self.concrete = str(value)!='false'
        else:
            self.concrete = None

    def __bool__(self):
        if self.concrete is not None:
            return self.concrete
        return self.tracer.branch(self)

    def __and__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicBool(self.tracer.backend.And(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __or__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicBool(self.tracer.backend.Or(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __not__(self):
        if self.concrete is not None:
            return SymbolicBool(not self.concrete, tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.Not(self.z3_expr), tracer=self.tracer)

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
            if isinstance(other.concrete, (int, bool)):
                return SymbolicInt(self.concrete + other.concrete, tracer=self.tracer)
            elif isinstance(other.concrete, float):
                return SymbolicFloat(self.concrete + other.concrete, tracer=self.tracer)
            elif isinstance(other.concrete, str):
                return SymbolicStr(str(self.concrete) + other.concrete, tracer=self.tracer)
            else:
                # Default case, use backend Add
                return SymbolicInt(self.tracer.backend.Add(self.z3_expr, other.z3_expr), tracer=self.tracer)
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
            if isinstance(other.concrete, (int, bool)):
                return SymbolicInt(self.concrete - other.concrete, tracer=self.tracer)
            elif isinstance(other.concrete, float):
                return SymbolicFloat(self.concrete - other.concrete, tracer=self.tracer)
            else:
                # Default case, use backend Sub
                return SymbolicInt(self.tracer.backend.Sub(self.z3_expr, other.z3_expr), tracer=self.tracer)
        return SymbolicInt(self.tracer.backend.Sub(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __mul__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if isinstance(other, SymbolicFloat):
            return SymbolicFloat(self.tracer.backend.Mul(self.z3_expr, other.z3_expr), tracer=self.tracer)
        if isinstance(other, SymbolicStr):
            return SymbolicStr(self.tracer.backend.StrMul(other.z3_expr, self.z3_expr), tracer=self.tracer)
        if self.concrete is not None and other.concrete is not None:
            if isinstance(other.concrete, (int, bool)):
                return SymbolicInt(self.concrete * other.concrete, tracer=self.tracer)
            elif isinstance(other.concrete, float):
                return SymbolicFloat(self.concrete * other.concrete, tracer=self.tracer)
            elif isinstance(other.concrete, str):
                return SymbolicStr(other.concrete * self.concrete, tracer=self.tracer)
            else:
                # Default case, use backend Mul
                return SymbolicInt(self.tracer.backend.Mul(self.z3_expr, other.z3_expr), tracer=self.tracer)
        return SymbolicInt(self.tracer.backend.Mul(self.z3_expr, other.z3_expr), tracer=self.tracer)
        
    def __eq__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            return SymbolicBool(self.concrete == other.concrete, tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.Eq(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __lt__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            if isinstance(other.concrete, (int, bool, float)):
                return SymbolicBool(self.concrete < other.concrete, tracer=self.tracer)
            else:
                # Default case, use backend LT
                return SymbolicBool(self.tracer.backend.LT(self.z3_expr, other.z3_expr), tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.LT(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __gt__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            if isinstance(other.concrete, (int, bool, float)):
                return SymbolicBool(self.concrete > other.concrete, tracer=self.tracer)
            else:
                # Default case, use backend GT
                return SymbolicBool(self.tracer.backend.GT(self.z3_expr, other.z3_expr), tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.GT(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            if isinstance(other.concrete, (int, bool, float)):
                return SymbolicFloat(self.concrete / other.concrete, tracer=self.tracer)
            else:
                # Default case, use backend Div
                return SymbolicFloat(self.tracer.backend.Div(self.z3_expr, other.z3_expr), tracer=self.tracer)
        return SymbolicFloat(self.tracer.backend.Div(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __rtruediv__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            if isinstance(other.concrete, (int, bool, float)):
                return SymbolicFloat(other.concrete / self.concrete, tracer=self.tracer)
            else:
                # Default case, use backend Div
                return SymbolicFloat(self.tracer.backend.Div(other.z3_expr, self.z3_expr), tracer=self.tracer)
        return SymbolicFloat(self.tracer.backend.Div(other.z3_expr, self.z3_expr), tracer=self.tracer)
    
    def __floordiv__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicInt(self.tracer.backend.UDiv(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __rfloordiv__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicInt(self.tracer.backend.UDiv(other.z3_expr, self.z3_expr), tracer=self.tracer)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __rsub__(self, other):
        if isinstance(other, int):
            other = SymbolicInt(self.tracer.backend.IntVal(other), tracer=self.tracer)
        elif isinstance(other, float):
            other = SymbolicFloat(self.tracer.backend.RealVal(other), tracer=self.tracer)
        return SymbolicInt(self.tracer.backend.Sub(other.z3_expr, self.z3_expr), tracer=self.tracer)
    
    def __hash__(self):
        return hash(str(self.z3_expr))

    def __abs__(self):
        if self.concrete is not None:
            return SymbolicInt(abs(self.concrete), tracer=self.tracer)
        # Use backend functions for comparison and negation
        return SymbolicInt(self.tracer.backend.If(self.tracer.backend.GE(self.z3_expr, self.tracer.backend.IntVal(0)), 
                                                  self.z3_expr, 
                                                  self.tracer.backend.Mul(self.tracer.backend.IntVal(-1), self.z3_expr)), 
                           tracer=self.tracer)

    def __mod__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            if isinstance(other.concrete, (int, bool)):
                return SymbolicInt(self.concrete % other.concrete, tracer=self.tracer)
            elif isinstance(other.concrete, float):
                return SymbolicFloat(self.concrete % other.concrete, tracer=self.tracer)
            else:
                # Default case, use backend Mod
                return SymbolicInt(self.tracer.backend.Mod(self.z3_expr, other.z3_expr), tracer=self.tracer)
        return SymbolicInt(self.tracer.backend.Mod(self.z3_expr, other.z3_expr), tracer=self.tracer)

    def __rmod__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            if isinstance(other.concrete, (int, bool)):
                return SymbolicInt(other.concrete % self.concrete, tracer=self.tracer)
            elif isinstance(other.concrete, float):
                return SymbolicFloat(other.concrete % self.concrete, tracer=self.tracer)
            else:
                # Default case, use backend Mod
                return SymbolicInt(self.tracer.backend.Mod(other.z3_expr, self.z3_expr), tracer=self.tracer)
        return SymbolicInt(self.tracer.backend.Mod(other.z3_expr, self.z3_expr), tracer=self.tracer)


    def __pow__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if isinstance(other, SymbolicFloat):
            return SymbolicFloat(self.tracer.backend.Pow(self.z3_expr, other.z3_expr), tracer=self.tracer)
        if self.concrete is not None and other.concrete is not None:
            if isinstance(other.concrete, (int, bool)):
                return SymbolicInt(self.concrete ** other.concrete, tracer=self.tracer)
            elif isinstance(other.concrete, float):
                return SymbolicFloat(self.concrete ** other.concrete, tracer=self.tracer)
            else:
                # Default case, use backend Pow
                return SymbolicInt(self.tracer.backend.Pow(self.z3_expr, other.z3_expr), tracer=self.tracer)
        return SymbolicInt(self.tracer.backend.Pow(self.z3_expr, other.z3_expr), tracer=self.tracer)

    def __rpow__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if isinstance(other, SymbolicFloat):
            return SymbolicFloat(self.tracer.backend.Pow(other.z3_expr, self.z3_expr), tracer=self.tracer)
        if self.concrete is not None and other.concrete is not None:
            if isinstance(other.concrete, (int, bool)):
                return SymbolicInt(other.concrete ** self.concrete, tracer=self.tracer)
            elif isinstance(other.concrete, float):
                return SymbolicFloat(other.concrete ** self.concrete, tracer=self.tracer)
            else:
                # Default case, use backend Pow
                return SymbolicInt(self.tracer.backend.Pow(other.z3_expr, self.z3_expr), tracer=self.tracer)
        return SymbolicInt(self.tracer.backend.Pow(other.z3_expr, self.z3_expr), tracer=self.tracer)


    def __le__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicBool(self.tracer.backend.LE(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __ge__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            if isinstance(other.concrete, (int, bool, float)) and isinstance(self.concrete, (int, float, bool)):
                return SymbolicBool(self.concrete >= other.concrete, tracer=self.tracer)
            else:
                # Default case, use backend GE
                return SymbolicBool(self.tracer.backend.GE(self.z3_expr, other.z3_expr), tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.GE(self.z3_expr, other.z3_expr), tracer=self.tracer)

    def __ne__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            return SymbolicBool(self.concrete != other.concrete, tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.Not(self.tracer.backend.Eq(self.z3_expr, other.z3_expr)), tracer=self.tracer)

    def __divmod__(self, other):
        other = self.tracer.ensure_symbolic(other)
        q = SymbolicInt(self.tracer.backend.Div(self.z3_expr, other.z3_expr), tracer=self.tracer)
        r = SymbolicInt(self.tracer.backend.Mod(self.z3_expr, other.z3_expr), tracer=self.tracer)
        return (q, r)

    def __rdivmod__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return divmod(other, self)

    def __lshift__(self, other):
        other = self.tracer.ensure_symbolic(other)
        # Use Mul and Pow from backend instead of direct operators
        power = self.tracer.backend.Pow(self.tracer.backend.IntVal(2), other.z3_expr)
        return SymbolicInt(self.tracer.backend.Mul(self.z3_expr, power), tracer=self.tracer)

    def __rshift__(self, other):
        other = self.tracer.ensure_symbolic(other)
        # Use Div and Pow from backend instead of direct operators
        power = self.tracer.backend.Pow(self.tracer.backend.IntVal(2), other.z3_expr)
        return SymbolicInt(self.tracer.backend.Div(self.z3_expr, power), tracer=self.tracer)

    def __neg__(self):
        if self.concrete is not None:
            return SymbolicInt(-self.concrete, tracer=self.tracer)
        # Use Mul with -1 instead of direct negation
        return SymbolicInt(self.tracer.backend.Mul(self.tracer.backend.IntVal(-1), self.z3_expr), tracer=self.tracer)

    def is_integer(self):
        """Always returns True since SymbolicInt is always an integer"""
        return SymbolicBool(True, tracer=self.tracer)

    def __and__(self, other):
        other = self.tracer.ensure_symbolic(other)
        # Ensure truthy returns a proper SymbolicBool with z3_expr
        t_self = truthy(self)
        t_other = truthy(other)
        if hasattr(t_self, 'z3_expr') and hasattr(t_other, 'z3_expr'):
            return SymbolicBool(self.tracer.backend.And(t_self.z3_expr, t_other.z3_expr), tracer=self.tracer)
        # Fallback to concrete evaluation if possible
        if self.concrete is not None and other.concrete is not None:
            return SymbolicBool(bool(self.concrete) and bool(other.concrete), tracer=self.tracer)
        # Default case - don't try to access z3_expr directly
        return SymbolicBool(False, tracer=self.tracer)
    
    def __or__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicInt(self.tracer.backend.If(truthy(self).z3_expr, self.z3_expr, other.z3_expr), tracer=self.tracer)

    def __xor__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicInt(self.tracer.backend.Xor(self.z3_expr, other.z3_expr), tracer=self.tracer)

    def __index__(self):
        if self.concrete is not None:
            return self.concrete
        raise ValueError("Cannot convert symbolic integer to index")

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
        if self.concrete is not None:
            return SymbolicFloat(abs(self.concrete), tracer=self.tracer)
        # Use backend functions for comparison and negation
        result = SymbolicFloat(
            self.tracer.backend.If(self.tracer.backend.GE(self.z3_expr, self.tracer.backend.RealVal(0.0)),
                                   self.z3_expr,
                                   self.tracer.backend.Mul(self.tracer.backend.RealVal(-1.0), self.z3_expr)),
            tracer=self.tracer)
        return result

    def __le__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            if isinstance(other.concrete, (int, bool, float)) and isinstance(self.concrete, (int, float, bool)):
                return SymbolicBool(self.concrete <= other.concrete, tracer=self.tracer)
            else:
                # Default case, use backend LE
                return SymbolicBool(self.tracer.backend.LE(self.z3_expr, other.z3_expr), tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.LE(self.z3_expr, other.z3_expr), tracer=self.tracer)

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
            if isinstance(other.concrete, (int, bool, float)) and isinstance(self.concrete, (int, float)):
                return SymbolicFloat(self.concrete + other.concrete, tracer=self.tracer)
            else:
                # Default case, use backend Add
                return SymbolicFloat(self.tracer.backend.Add(self.z3_expr, other.z3_expr), tracer=self.tracer)
        return SymbolicFloat(self.tracer.backend.Add(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            if isinstance(other.concrete, (int, bool, float)) and isinstance(self.concrete, (int, float)):
                return SymbolicFloat(self.concrete * other.concrete, tracer=self.tracer)
            else:
                # Default case, use backend Mul
                return SymbolicFloat(self.tracer.backend.Mul(self.z3_expr, other.z3_expr), tracer=self.tracer)
        return SymbolicFloat(self.tracer.backend.Mul(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __pow__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            if isinstance(other.concrete, (int, bool, float)) and isinstance(self.concrete, (int, float)):
                return SymbolicFloat(self.concrete ** other.concrete, tracer=self.tracer)
            else:
                # Default case, use backend Pow
                return SymbolicFloat(self.tracer.backend.Pow(self.z3_expr, other.z3_expr), tracer=self.tracer)
        return SymbolicFloat(self.tracer.backend.Pow(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __rpow__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicFloat(self.tracer.backend.Pow(other.z3_expr, self.z3_expr), tracer=self.tracer)
    
    def __gt__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            if isinstance(other.concrete, (int, bool, float)) and isinstance(self.concrete, (int, float, bool)):
                return SymbolicBool(self.concrete > other.concrete, tracer=self.tracer)
            else:
                # Default case, use backend GT
                return SymbolicBool(self.tracer.backend.GT(self.z3_expr, other.z3_expr), tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.GT(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __lt__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            if isinstance(other.concrete, (int, bool, float)) and isinstance(self.concrete, (int, float, bool)):
                return SymbolicBool(self.concrete < other.concrete, tracer=self.tracer)
            else:
                # Default case, use backend LT
                return SymbolicBool(self.tracer.backend.LT(self.z3_expr, other.z3_expr), tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.LT(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __ge__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            if isinstance(other.concrete, (int, bool, float)) and isinstance(self.concrete, (int, float, bool)):
                return SymbolicBool(self.concrete >= other.concrete, tracer=self.tracer)
            else:
                # Default case, use backend GE
                return SymbolicBool(self.tracer.backend.GE(self.z3_expr, other.z3_expr), tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.GE(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __str__(self):
        if self.concrete is not None:
            return str(self.concrete)
        return f"SymbolicFloat({self.name})"
    
    def __repr__(self):
        return self.__str__()

class SymbolicList:
    def __init__(self, value, name: Optional[str] = None, tracer: Optional[SymbolicTracer] = None):
        self.tracer = tracer or SymbolicTracer()        
        assert name is None       
        assert isinstance(value, list), "Symbolic lists not yet supported: found "+str(value)+" of type "+str(type(value))
        self.concrete = value
        self.z3_expr = None

    def __contains__(self, item):
        item = self.tracer.ensure_symbolic(item)
        if self.concrete is not None:
            if item.concrete is not None:
                return item.concrete in self.concrete
            else:
                return self.tracer.backend.Or(*[(item == x).z3_expr for x in self.concrete])
        return ValueError("Cannot check contains symbolically")

    def contains(self, item):
        return SymbolicBool(self.__contains__(item), tracer=self.tracer)

    def __getitem__(self, key):
        if isinstance(key, slice):
            if (not isinstance(key.start, SymbolicInt) and 
                not isinstance(key.stop, SymbolicInt) and 
                not isinstance(key.step, SymbolicInt)):
                return SymbolicList(self.concrete[key], tracer=self.tracer)
            return SymbolicSlice(self.concrete, key.start, key.stop, key.step, tracer=self.tracer)
        elif isinstance(key, SymbolicInt):
            # If we have a concrete value, use it directly
            if key.concrete is not None:
                return self.concrete[key.concrete]
                
            # Add bounds check
            n = len(self)
            self.tracer.add_constraint(key < n)
            self.tracer.add_constraint(key > -n)

            # Build an If expression to select the right value
            result = None
            for i, item in enumerate(self.concrete):
                if result is None:
                    result = item
                else:
                    result = SymbolicInt(
                        self.tracer.backend.If(
                            self.tracer.backend.Or(key.z3_expr == i, key.z3_expr == -n+i),
                            item.z3_expr,
                            result.z3_expr
                        ),
                        tracer=self.tracer
                    )
            return result
        return self.concrete[key]

    def __iter__(self):
        return iter(self.concrete)

    def __len__(self):
        return len(self.concrete)

    def __add__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            if isinstance(self.concrete, list) and isinstance(other.concrete, list):
                return SymbolicList(self.concrete + other.concrete, tracer=self.tracer)
            else:
                # If we can't concatenate directly, return empty list
                return SymbolicList([], tracer=self.tracer)
        return SymbolicList([], tracer=self.tracer)

    def __radd__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            if isinstance(self.concrete, list) and isinstance(other.concrete, list):
                return SymbolicList(other.concrete + self.concrete, tracer=self.tracer)
            else:
                # If we can't concatenate directly, return empty list
                return SymbolicList([], tracer=self.tracer)
        return SymbolicList([], tracer=self.tracer)

    def index(self, item):
        # Return first index where item appears as SymbolicInt
        result = None
        for i, x in enumerate(self.concrete):
            eq = (x == item)  # This gives us a SymbolicBool
            if result is None:
                result = SymbolicInt(
                    self.tracer.backend.If(
                        eq.z3_expr,  # condition
                        self.tracer.backend.IntVal(i),  # then
                        self.tracer.backend.IntVal(-1)  # else
                    ),
                    tracer=self.tracer
                )
            else:
                result = SymbolicInt(
                    self.tracer.backend.If(
                        eq.z3_expr,  # condition
                        self.tracer.backend.If(
                            result.z3_expr == -1,  # if not found yet
                            self.tracer.backend.IntVal(i),  # use this index
                            result.z3_expr  # keep previous index
                        ),
                        result.z3_expr  # else keep previous result
                    ),
                    tracer=self.tracer
                )
        return result

    def count(self, item):
        """Count occurrences of item in list"""
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

class SymbolicStrIterator:
    _counter = 0
    
    def __init__(self, sym_str):
        self.tracer = sym_str.tracer
        self.sym_str = sym_str
        self.length = sym_str.__len__()
        self.used = False
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.used:
            raise StopIteration
            
        # Create position variable directly as a Z3 variable in forall
        pos_name = f'str_pos_{SymbolicStrIterator._counter}'
        SymbolicStrIterator._counter += 1
        self.tracer.backend.quantified_vars.add(pos_name)        
        pos_var = SymbolicInt(name=pos_name, tracer=self.tracer)
        
        # Add bounds constraints for position
        bounds = self.tracer.backend.And(
            self.tracer.backend.GE(pos_var.z3_expr, self.tracer.backend.IntVal(0)),
            self.tracer.backend.LT(pos_var.z3_expr, self.length.z3_expr)
        )
        
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
        self.tracer = tracer or SymbolicTracer()
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
            if isinstance(self.concrete, str) and isinstance(sub.concrete, str):
                if isinstance(start.concrete, (int, bool)):
                    return SymbolicInt(self.concrete.index(sub.concrete, start.concrete), tracer=self.tracer)
                else:
                    # Default to 0 if start is not a valid index type
                    return SymbolicInt(self.concrete.index(sub.concrete, 0), tracer=self.tracer)
            else:
                # Default case, use backend StrIndexOf
                return SymbolicInt(self.tracer.backend.StrIndexOf(self.z3_expr, sub.z3_expr, start.z3_expr), tracer=self.tracer)
        return SymbolicInt(self.tracer.backend.StrIndexOf(self.z3_expr, sub.z3_expr, start.z3_expr), tracer=self.tracer)

    def join(self, ss):
        underlying = ss.concrete if hasattr(ss, 'concrete') and ss.concrete is not None else list(ss)
        return SymbolicStr(self.tracer.backend.StrJoin(self.z3_expr, self.tracer.backend.StrList([self.tracer.ensure_symbolic(x).z3_expr for x in underlying])), tracer=self.tracer)

    def __lt__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            if isinstance(self.concrete, str) and isinstance(other.concrete, str):
                return SymbolicBool(self.concrete < other.concrete, tracer=self.tracer)
            else:
                # Default case, use backend StrLT
                return SymbolicBool(self.tracer.backend.StrLT(self.z3_expr, other.z3_expr), tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.StrLT(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __gt__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            if isinstance(self.concrete, str) and isinstance(other.concrete, str):
                return SymbolicBool(self.concrete > other.concrete, tracer=self.tracer)
            else:
                # Default case, use backend StrLT (reversed)
                return SymbolicBool(self.tracer.backend.StrLT(other.z3_expr, self.z3_expr), tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.StrLT(other.z3_expr, self.z3_expr), tracer=self.tracer)

    def __le__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return self.__lt__(other).__or__(self.__eq__(other))
    
    def __ge__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            if isinstance(self.concrete, str) and isinstance(other.concrete, str):
                return SymbolicBool(self.concrete >= other.concrete, tracer=self.tracer)
            else:
                # Default case, use backend StrLT (negated)
                return SymbolicBool(self.tracer.backend.Not(self.tracer.backend.StrLT(self.z3_expr, other.z3_expr)), tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.Not(self.tracer.backend.StrLT(self.z3_expr, other.z3_expr)), tracer=self.tracer)

    def __hash__(self):
        if self.concrete is not None:
            return hash(self.concrete)
        raise ValueError("Symbolic hash not yet implemented")

    def split(self, sep=None):
        """Split string into list of strings"""
        if self.concrete is not None:
            # If we have a concrete string, use Python's split
            parts = self.concrete.split(sep)
            return SymbolicList([SymbolicStr(p, tracer=self.tracer) for p in parts], tracer=self.tracer)
        
        # For symbolic strings, we need to use Z3's string operations
        if sep is None:
            sep = " "  # Default separator is whitespace
        sep = self.tracer.ensure_symbolic(sep)
        # Use Z3's string operations to split
        result = self.tracer.backend.StrSplit(self.z3_expr, sep.z3_expr)
        return SymbolicList(result, tracer=self.tracer)

    def __contains__(self, item):
        item = self.tracer.ensure_symbolic(item)
        if self.concrete is not None and item.concrete is not None:
            if isinstance(self.concrete, str) and isinstance(item.concrete, str):
                return item.concrete in self.concrete
            else:
                # Default case, use backend StrContains
                return self.tracer.backend.StrContains(self.z3_expr, item.z3_expr)
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
            
        # Create a symbolic version of the substring if needed
        sub_sym = sub
        if isinstance(sub, str):
            sub_sym = SymbolicStr(sub, tracer=self.tracer)
            
        # Create a symbolic integer for the count
        result = SymbolicInt(self.tracer.backend.StrCount(self.z3_expr, sub_sym.z3_expr), tracer=self.tracer)
        
        # Add constraint that count is non-negative using backend methods
        self.tracer.add_constraint(self.tracer.backend.GE(result.z3_expr, self.tracer.backend.IntVal(0)))
        
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
        # Ensure truthy returns a proper SymbolicBool with z3_expr
        t_self = truthy(self)
        t_other = truthy(other)
        if hasattr(t_self, 'z3_expr') and hasattr(t_other, 'z3_expr'):
            return SymbolicBool(self.tracer.backend.And(t_self.z3_expr, t_other.z3_expr), tracer=self.tracer)
        # Fallback to concrete evaluation if possible
        if self.concrete is not None and other.concrete is not None:
            return SymbolicBool(bool(self.concrete) and bool(other.concrete), tracer=self.tracer)
        # Default case - don't try to access z3_expr directly
        return SymbolicBool(False, tracer=self.tracer)
    
    def __or__(self, other):
        other = self.tracer.ensure_symbolic(other)
        # Ensure truthy returns a proper SymbolicBool with z3_expr
        t_self = truthy(self)
        t_other = truthy(other)
        if hasattr(t_self, 'z3_expr') and hasattr(t_other, 'z3_expr'):
            return SymbolicBool(self.tracer.backend.Or(t_self.z3_expr, t_other.z3_expr), tracer=self.tracer)
        # Fallback to concrete evaluation if possible
        if self.concrete is not None and other.concrete is not None:
            return SymbolicBool(bool(self.concrete) or bool(other.concrete), tracer=self.tracer)
        # Default case - don't try to access z3_expr directly
        return SymbolicBool(True, tracer=self.tracer)

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
            if isinstance(self.concrete, str) and isinstance(a.concrete, str) and isinstance(b.concrete, str):
                return SymbolicStr(self.concrete.replace(a.concrete, b.concrete), tracer=self.tracer)
            else:
                # Default case, use backend StrReplace
                return SymbolicStr(self.tracer.backend.StrReplace(self.z3_expr, a.z3_expr, b.z3_expr), tracer=self.tracer)
        return SymbolicStr(self.tracer.backend.StrReplace(self.z3_expr, a.z3_expr, b.z3_expr), tracer=self.tracer)

class SymbolicSlice:
    def __init__(self, concrete_seq, start, end, step=None, tracer: Optional[SymbolicTracer] = None):
        assert concrete_seq is not None
        self.concrete = concrete_seq  # Can be str or list
        self.start = start
        self.end = end
        self.step = step
        self.tracer = tracer or SymbolicTracer()

    def __add__(self, other):
        return self.get_slice() + other.get_slice()

    def __and__(self, other):
        other = self.tracer.ensure_symbolic(other)
        # Ensure truthy returns a proper SymbolicBool with z3_expr
        t_self = truthy(self)
        t_other = truthy(other)
        if hasattr(t_self, 'z3_expr') and hasattr(t_other, 'z3_expr'):
            return SymbolicBool(self.tracer.backend.And(t_self.z3_expr, t_other.z3_expr), tracer=self.tracer)
        # Fallback to concrete evaluation if possible
        if self.concrete is not None and other.concrete is not None:
            return SymbolicBool(bool(self.concrete) and bool(other.concrete), tracer=self.tracer)
        # Default case - don't try to access z3_expr directly
        return SymbolicBool(False, tracer=self.tracer)
    
    def __or__(self, other):
        other = self.tracer.ensure_symbolic(other)
        # Ensure truthy returns a proper SymbolicBool with z3_expr
        t_self = truthy(self)
        t_other = truthy(other)
        if hasattr(t_self, 'z3_expr') and hasattr(t_other, 'z3_expr'):
            return SymbolicBool(self.tracer.backend.Or(t_self.z3_expr, t_other.z3_expr), tracer=self.tracer)
        # Fallback to concrete evaluation if possible
        if self.concrete is not None and other.concrete is not None:
            return SymbolicBool(bool(self.concrete) or bool(other.concrete), tracer=self.tracer)
        # Default case - don't try to access z3_expr directly
        return SymbolicBool(True, tracer=self.tracer)

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

    def upper(self):
        sliced = self.get_slice()
        if isinstance(sliced, SymbolicStr):
            return sliced.upper()
        # Default case for non-string types
        return sliced

    def count(self, sub):
        """Count occurrences in sliced sequence"""
        # Get the sliced sequence
        sliced = self.get_slice()
        # Use the appropriate count method
        return sliced.count(sub)

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
                   else SymbolicList(result, tracer=self.tracer))
        
        # For symbolic indices, use substring operation
        start = self.start if self.start is not None else 0
        end = self.end if self.end is not None else len(self.concrete)
        
        if isinstance(self.concrete, str):
            # Use Z3's Extract for strings
            start_expr = start.z3_expr if isinstance(start, SymbolicInt) else self.tracer.backend.IntVal(int(start))
            end_expr = end.z3_expr if isinstance(end, SymbolicInt) else self.tracer.backend.IntVal(int(end))
            length_expr = self.tracer.backend.Sub(end_expr, start_expr)
            return SymbolicStr(
                self.tracer.backend.StrSubstr(self.tracer.backend.StringVal(self.concrete), start_expr, length_expr, variant="str.substr"),
                tracer=self.tracer
            )
        else:
            # For lists, we still need to implement this
            raise ValueError("Not implemented: symbolic list slicing")

class SymbolicRangeIterator:
    def __init__(self, sym_range):
        self.tracer = sym_range.tracer
        self.sym_range = sym_range
        # Create fresh variable for the iterator
        self.var = SymbolicInt(name=f'i_{SymbolicRange._counter}', tracer=sym_range.tracer)
        SymbolicRange._counter += 1
        self.used = False
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.used:
            raise StopIteration
        self.used = True
        bounds = self.get_bounds()
        self.tracer.forall_conditions.append((self.var.z3_expr, bounds.z3_expr))
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
        
        # Create individual constraints
        in_range = SymbolicBool(self.tracer.backend.And(
            self.tracer.backend.GE(item.z3_expr, self.start.z3_expr),
            self.tracer.backend.LT(item.z3_expr, self.end.z3_expr)
        ), tracer=self.tracer)
        
        if self.step is not None:
            # item = start + k * step for some k
            k = SymbolicInt('k', tracer=self.tracer)
            step_constraint = SymbolicBool(self.tracer.backend.Eq(
                item.z3_expr, 
                self.tracer.backend.Add(
                    self.start.z3_expr, 
                    self.tracer.backend.Mul(k.z3_expr, self.step.z3_expr)
                )
            ), tracer=self.tracer)
            
            # Combine constraints
            result = SymbolicBool(self.tracer.backend.And(
                in_range.z3_expr, 
                step_constraint.z3_expr
            ), tracer=self.tracer)
            return result
        
        return in_range

    def contains(self, item):
        return SymbolicBool(self.__contains__(item), tracer=self.tracer)

    def __len__(self):
        # Handle concrete values
        if (isinstance(self.step, (int, type(None))) and 
            isinstance(self.start, int) and 
            isinstance(self.end, int)):
            step = 1 if self.step is None else self.step
            if step == 0:
                return 0
            # Calculate length using integer division for concrete values
            diff = self.end - self.start
            if isinstance(diff, (SymbolicInt, SymbolicFloat)):
                return SymbolicInt(self.tracer.backend.UDiv(diff.z3_expr, self.tracer.backend.IntVal(step)), tracer=self.tracer)
            return diff // step if step > 0 else 0
        
        # For symbolic values, create a symbolic expression
        step_expr = self.tracer.backend.IntVal(1) if self.step is None else self.step.z3_expr
        return SymbolicInt(
            self.tracer.backend.UDiv(  # Use UDiv for integer division
                self.tracer.backend.Sub(self.end.z3_expr, self.start.z3_expr),
                step_expr
            ),
            tracer=self.tracer
        )

def fresh_symbolic(var):
    typ = type(var).__name__.lower().replace('symbolic', '')
    if hasattr(var, 'name'):
        return make_symbolic(typ, var.name, var.tracer)
    return make_symbolic(typ, "fresh_var", var.tracer if hasattr(var, 'tracer') else None)

def make_symbolic(typ, name: str, tracer: Optional[SymbolicTracer] = None) -> Any:
    """Create a new symbolic variable of given type"""
    if typ == int or typ == 'int':
        sym = SymbolicInt(name, tracer=tracer)
    elif typ == bool or typ == 'bool':
        sym = SymbolicBool(False, tracer=tracer)  # Default value
    elif typ == float or typ == 'float':
        sym = SymbolicFloat(0.0, tracer=tracer)  # Default value
    elif typ == str or typ == 'str':
        sym = SymbolicStr("", tracer=tracer)  # Default value
    else:
        raise ValueError(f"Unsupported symbolic type: {typ}")
    return sym

def truthy(x):
    """Convert a value to a symbolic boolean"""
    if isinstance(x, SymbolicBool):
        return x
    elif isinstance(x, SymbolicInt):
        return x != 0
    elif isinstance(x, SymbolicStr):
        return x != ""
    elif isinstance(x, SymbolicList):
        # For lists, check if length > 0
        if hasattr(x, 'tracer'):
            return SymbolicBool(len(x) > 0, tracer=x.tracer)
    elif isinstance(x, SymbolicSlice):
        # For slices, treat as non-empty
        if hasattr(x, 'tracer'):
            return SymbolicBool(True, tracer=x.tracer)
    
    # For concrete values or other types
    if hasattr(x, 'tracer'):
        return SymbolicBool(bool(x), tracer=x.tracer)
    
    # Create a new SymbolicBool with no tracer
    return SymbolicBool(bool(x))

def as_concrete(x):
    if hasattr(x, 'concrete') and x.concrete is not None:
        return x.concrete
    else:
        return x
