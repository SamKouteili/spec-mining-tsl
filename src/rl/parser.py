import sys
from pathlib import Path
import re
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# from src.rl.loop import Op, AP, Not, And, Or, Next, Until, Eventually, Always, temporalOp, LogicalOp, neg
from src.bolt.log2tslf import Update, Predicate, BooleanAP

#################################################################
################## SPEC TYPES ###################################
#################################################################

type AP = BooleanAP | Predicate | Update

@dataclass(frozen=True)
class Not:
    op: 'Op'
    def __str__(self) -> str:
        return f"(! {self.op})"

@dataclass(frozen=True)
class And:
    left: 'Op'
    right: 'Op'
    def __str__(self) -> str:
        return f"({self.left} && {self.right})"

@dataclass(frozen=True)
class Or:
    left: 'Op'
    right: 'Op'
    def __str__(self) -> str:
        return f"({self.left} || {self.right})"

type LogicalOp = Not | And | Or

@dataclass(frozen=True)
class Next:
    op: 'Op'
    def __str__(self) -> str:
        return f"(X {self.op})"

@dataclass(frozen=True)
class Until:
    left: 'Op'
    right: 'Op'
    def __str__(self) -> str:
        return f"({self.left} U {self.right})"

@dataclass(frozen=True)
class Eventually:
    op: 'Op'
    def __str__(self) -> str:
        return f"(F {self.op})"

@dataclass(frozen=True)
class Always:
    op: 'Op'
    def __str__(self) -> str:
        return f"(G {self.op})"

type temporalOp = Next | Until | Eventually | Always

type Op = AP | LogicalOp | temporalOp

def op_to_str(op: Op) -> str:
    """Convert an Op to its string representation."""
    return str(op)

def neg(op: Op) -> Op:
    match op:
        case Not(inner):
            return inner
        case And(left, right):
            return Or(neg(left), neg(right))
        case Or(left, right):
            return And(neg(left), neg(right))
        case Always(inner):
            return Eventually(neg(inner))
        case Eventually(inner):
            return Always(neg(inner))
        case Until(left, right):
            return Or(And(neg(right), Always(neg(left))), Eventually(neg(right)))
        case Next(inner):
            return Next(neg(inner))
        case _:
            return Not(op)

#################################################################
################## TSL_f PARSER #################################
#################################################################

class TSLTokenizer:
    """Tokenizer for TSL_f formulas."""

    # Token patterns
    TOKENS = [
        ('LPAREN', r'\('),
        ('RPAREN', r'\)'),
        ('LBRACKET', r'\['),
        ('RBRACKET', r'\]'),
        ('IFF', r'<->'),
        ('IMPLIES', r'->'),
        ('ARROW', r'<-'),
        ('AND', r'&&|&'),
        ('OR', r'\|\||\|'),
        ('NOT', r'!'),
        ('NEXT', r'X\[!\]|X(?![a-zA-Z0-9_])'),  # X[!] or standalone X
        ('UNTIL', r'U(?![a-zA-Z0-9_])'),  # U not followed by identifier chars
        ('EVENTUALLY', r'F'),
        ('ALWAYS', r'G'),
        ('NUMBER', r'\d+'),  # For array indices
        ('IDENT', r'[a-zA-Z_][a-zA-Z0-9_]*'),
        ('SKIP', r'[ \t]+'),
    ]

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.tokens: list[tuple[str, str]] = []
        self._tokenize()
        self.idx = 0

    def _tokenize(self):
        pattern = '|'.join(f'(?P<{name}>{regex})' for name, regex in self.TOKENS)
        for m in re.finditer(pattern, self.text):
            kind = m.lastgroup
            value = m.group()
            if kind != 'SKIP':
                self.tokens.append((kind, value))

    def peek(self) -> tuple[str, str] | None:
        if self.idx < len(self.tokens):
            return self.tokens[self.idx]
        return None

    def consume(self) -> tuple[str, str]:
        tok = self.tokens[self.idx]
        self.idx += 1
        return tok

    def expect(self, kind: str) -> str:
        tok = self.peek()
        if tok is None or tok[0] != kind:
            raise ValueError(f"Expected {kind}, got {tok}")
        return self.consume()[1]


class TSLParser:
    """
    Recursive descent parser for TSL_f formulas.

    Grammar (precedence from lowest to highest):
        expr     ::= until
        until    ::= iff (U iff)*
        iff      ::= implies (<-> implies)*
        implies  ::= or (-> or)*
        or       ::= and (| and)*
        and      ::= unary (& unary)*
        unary    ::= ! unary | F unary | G unary | X[!] unary | primary
        primary  ::= ( expr ) | update | predicate | booleanAP
        update   ::= [ ident <- term ]
        term     ::= ident+
    """

    def __init__(self, text: str):
        self.tokenizer = TSLTokenizer(text)

    def parse(self) -> Op:
        result = self._parse_until()
        if self.tokenizer.peek() is not None:
            raise ValueError(f"Unexpected token after parsing: {self.tokenizer.peek()}")
        return result

    def _parse_until(self) -> Op:
        left = self._parse_iff()
        while self.tokenizer.peek() and self.tokenizer.peek()[0] == 'UNTIL':
            self.tokenizer.consume()
            right = self._parse_iff()
            left = Until(left, right)
        return left

    def _parse_iff(self) -> Op:
        left = self._parse_implies()
        while self.tokenizer.peek() and self.tokenizer.peek()[0] == 'IFF':
            self.tokenizer.consume()
            right = self._parse_implies()
            # a <-> b = (a & b) | (!a & !b)
            left = Or(And(left, right), And(Not(left), Not(right)))
        return left

    def _parse_implies(self) -> Op:
        left = self._parse_or()
        while self.tokenizer.peek() and self.tokenizer.peek()[0] == 'IMPLIES':
            self.tokenizer.consume()
            right = self._parse_or()
            # a -> b = !a | b
            left = Or(Not(left), right)
        return left

    def _parse_or(self) -> Op:
        left = self._parse_and()
        while self.tokenizer.peek() and self.tokenizer.peek()[0] == 'OR':
            self.tokenizer.consume()
            right = self._parse_and()
            left = Or(left, right)
        return left

    def _parse_and(self) -> Op:
        left = self._parse_unary()
        while self.tokenizer.peek() and self.tokenizer.peek()[0] == 'AND':
            self.tokenizer.consume()
            right = self._parse_unary()
            left = And(left, right)
        return left

    def _parse_unary(self) -> Op:
        tok = self.tokenizer.peek()
        if tok is None:
            raise ValueError("Unexpected end of input")

        if tok[0] == 'NOT':
            self.tokenizer.consume()
            return Not(self._parse_unary())
        elif tok[0] == 'EVENTUALLY':
            self.tokenizer.consume()
            return Eventually(self._parse_unary())
        elif tok[0] == 'ALWAYS':
            self.tokenizer.consume()
            return Always(self._parse_unary())
        elif tok[0] == 'NEXT':
            self.tokenizer.consume()
            return Next(self._parse_unary())
        else:
            return self._parse_primary()

    def _parse_primary(self) -> Op:
        tok = self.tokenizer.peek()
        if tok is None:
            raise ValueError("Unexpected end of input")

        if tok[0] == 'LPAREN':
            self.tokenizer.consume()
            expr = self._parse_until()
            self.tokenizer.expect('RPAREN')
            return expr
        elif tok[0] == 'LBRACKET':
            return self._parse_update()
        else:
            # Could be a predicate or boolean AP
            return self._parse_predicate_or_boolean()

    def _parse_indexed_ident(self) -> str:
        """Parse an identifier, optionally with array index: ident or ident[N]"""
        name = self.tokenizer.expect('IDENT')
        # Check for array index
        if self.tokenizer.peek() and self.tokenizer.peek()[0] == 'LBRACKET':
            self.tokenizer.consume()  # consume [
            idx = self.tokenizer.expect('NUMBER')
            self.tokenizer.expect('RBRACKET')
            return f"{name}[{idx}]"
        return name

    def _parse_term(self) -> str:
        """Parse a term: ident, ident[N], or ident() (nullary function call)"""
        tok = self.tokenizer.peek()
        if tok is None:
            raise ValueError("Unexpected end of input in term")

        if tok[0] == 'IDENT':
            name = self.tokenizer.consume()[1]
            # Check for array index or function call
            next_tok = self.tokenizer.peek()
            if next_tok and next_tok[0] == 'LBRACKET':
                self.tokenizer.consume()  # consume [
                idx = self.tokenizer.expect('NUMBER')
                self.tokenizer.expect('RBRACKET')
                return f"{name}[{idx}]"
            elif next_tok and next_tok[0] == 'LPAREN':
                self.tokenizer.consume()  # consume (
                self.tokenizer.expect('RPAREN')
                return f"{name}()"
            return name
        elif tok[0] == 'NUMBER':
            return self.tokenizer.consume()[1]
        else:
            raise ValueError(f"Expected term, got {tok}")

    def _parse_update(self) -> Update:
        """Parse [var <- func args] or [var <- var]

        Handles:
        - Simple: [x <- y]
        - Indexed: [player[0] <- player[1]]
        - Function calls: [x <- add x i1()]
        """
        self.tokenizer.expect('LBRACKET')

        # Parse LHS (possibly indexed)
        var_name = self._parse_indexed_ident()

        self.tokenizer.expect('ARROW')

        # Collect RHS terms until RBRACKET
        rhs_parts = []
        while self.tokenizer.peek() and self.tokenizer.peek()[0] not in ('RBRACKET',):
            tok = self.tokenizer.peek()
            if tok[0] in ('IDENT', 'NUMBER'):
                rhs_parts.append(self._parse_term())
            else:
                break

        self.tokenizer.expect('RBRACKET')

        # Build Update from string
        rhs_str = ' '.join(rhs_parts)
        update_str = f"[{var_name} <- {rhs_str}]"
        return Update.from_string(update_str)

    def _parse_predicate_or_boolean(self) -> Predicate | BooleanAP:
        """Parse a predicate (like 'eqC x y') or boolean AP (like 'END')."""
        first = self.tokenizer.expect('IDENT')

        # Look ahead to see if there are more identifiers (predicate arguments)
        args = []
        while self.tokenizer.peek() and self.tokenizer.peek()[0] == 'IDENT':
            # Check if it might be a keyword or operator
            next_tok = self.tokenizer.peek()[1]
            if next_tok in ('U', 'F', 'G', 'END'):
                break
            args.append(self.tokenizer.consume()[1])

        if args:
            # It's a predicate
            pred_str = f"{first} {' '.join(args)}"
            return Predicate.from_string(pred_str)
        else:
            # It's a boolean AP
            return BooleanAP.from_string(first)


def parse_tsl(text: str) -> Op:
    """Parse a TSL_f formula string into an Op."""
    return TSLParser(text).parse()


def parse_tsl_file(path: Path) -> list[Op]:
    """Parse a TSL file containing one formula per line."""
    ops = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                ops.append(parse_tsl(line))
    return ops


def arg_parse():
    import argparse

    parser = argparse.ArgumentParser(description="TSL_f Parser")
    parser.add_argument("tsl_file", type=Path, help="Path to the TSL_f file to parse")
    return parser.parse_args()

if __name__ == "__main__":
    # Example usage
    args = arg_parse()
    ops = parse_tsl_file(args.tsl_file)
    print(ops)