from typing import Any, Optional, List, Type
from dataclasses import dataclass
from contextlib import contextmanager
from .backend import default_backend, Backend

@dataclass 
class SymbolicTracer:
    def __init__(self, backend=None):
        self.backend = backend or default_backend()
        self.path_conditions = []
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
        """Handle a branching point in execution"""
        if len(self.current_branch_exploration) > self.branch_counter:
            branch_val = self.current_branch_exploration[self.branch_counter]
        else:
            branch_val = True
            self.remaining_branch_explorations.append(self.current_branch_exploration + [False])
            self.current_branch_exploration += [branch_val]

        condition = self.ensure_symbolic(condition)
        self.path_conditions.append(condition.z3_expr if branch_val else self.backend.Not(condition.z3_expr))
        return branch_val

    def add_constraint(self, constraint):
        if isinstance(constraint, (SymbolicInt, SymbolicBool)):
            constraint = truthy(constraint).z3_expr
        if self.path_conditions:
            constraint = self.backend.Implies(
                self.backend.And(*self.path_conditions),
                constraint)
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
    
    def __floordiv__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicInt(self.tracer.backend.UDiv(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __rfloordiv__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicInt(self.tracer.backend.UDiv(other.z3_expr, self.z3_expr), tracer=self.tracer)
    
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

    def __mod__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            return SymbolicInt(self.concrete % other.concrete, tracer=self.tracer)
        return SymbolicInt(self.tracer.backend.Mod(self.z3_expr, other.z3_expr), tracer=self.tracer)

    def __rmod__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            return SymbolicInt(other.concrete % self.concrete, tracer=self.tracer)
        return SymbolicInt(self.tracer.backend.Mod(other.z3_expr, self.z3_expr), tracer=self.tracer)


    def __pow__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if isinstance(other, SymbolicFloat):
            return SymbolicFloat(self.tracer.backend.Pow(self.z3_expr, other.z3_expr), tracer=self.tracer)
        if self.concrete is not None and other.concrete is not None:
            return SymbolicInt(other.concrete ** self.concrete, tracer=self.tracer)
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
        return SymbolicInt(self.z3_expr * (2 ** other.z3_expr), tracer=self.tracer)

    def __rshift__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicInt(self.z3_expr / (2 ** other.z3_expr), tracer=self.tracer)

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

    def __index__(self):
        if self.concrete is not None:
            return self.concrete
        raise ValueError("Cannot convert symbolic integer to index")

class SymbolicFloat:
    def __init__(self, value, tracer=None):
        self.tracer = tracer
        self.z3_expr = value
        self.concrete = value if isinstance(value, (int, float)) else None

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

class SymbolicList:
    def __init__(self, value, name: Optional[str] = None, tracer: Optional[SymbolicTracer] = None):
        self.tracer = tracer        
        assert name is None       
        assert isinstance(value, list), "Symbolic lists not yet supported: found "+str(value)+" of type "+str(type(value))
        self.concrete = value

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
        return SymbolicList(self.concrete + other.concrete, tracer=self.tracer)

    def __radd__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicList(other.concrete + self.concrete, tracer=self.tracer)

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
            
        # Use Z3's string operations to split
        result = self.tracer.backend.StrSplit(self.z3_expr, self.tracer.backend.StringVal(sep))
        return SymbolicList(result, tracer=self.tracer)

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
                result = self.z3_expr == other.z3_expr
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

    def replace(self, a, b):
        a = self.tracer.ensure_symbolic(a)
        b = self.tracer.ensure_symbolic(b)
        if self.concrete is not None and a.concrete is not None and b.concrete is not None:
            return SymbolicStr(self.concrete.replace(a.concrete, b.concrete), tracer=self.tracer)
        return SymbolicStr(self.tracer.backend.StrReplace(self.z3_expr, a.z3_expr, b.z3_expr), tracer=self.tracer)

class SymbolicSlice:
    def __init__(self, concrete_seq, start, end, step=None, tracer: Optional[SymbolicTracer] = None):
        assert concrete_seq is not None
        self.concrete = concrete_seq  # Can be str or list
        self.start = start
        self.end = end
        self.step = step
        self.tracer = tracer

    def __add__(self, other):
        return self.get_slice() + other.get_slice()

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
            start_expr = start.z3_expr if isinstance(start, SymbolicInt) else self.tracer.backend.IntVal(start)
            length_expr = (end.z3_expr if isinstance(end, SymbolicInt) else self.tracer.backend.IntVal(end)) - start_expr
            return SymbolicStr(
                self.tracer.backend.StrSubstr(self.tracer.backend.StringVal(self.concrete), start_expr, length_expr),
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

def make_symbolic(typ: Type, name: str, tracer: Optional[SymbolicTracer] = None) -> Any:
    """Create a new symbolic variable of given type"""
    if typ == int or typ == 'int':
        sym = SymbolicInt(name=name, tracer=tracer)
    elif typ == bool or typ == 'bool':
        sym = SymbolicBool(name=name, tracer=tracer)
    elif typ == float or typ == 'float':
        sym = SymbolicFloat(name=name, tracer=tracer)
    elif typ == str or typ == 'str':
        sym = SymbolicStr(name=name, tracer=tracer)
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
