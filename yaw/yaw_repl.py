"""
yaw REPL: Interpreter for algebraic quantum programming

Minimal REPL that can be incrementally upgraded to support
full context management, braiding, and advanced features.
"""

from yaw_prototype import *
from sympy import sqrt
import sys
import traceback
import os

# Import readline for command history and line editing
# This provides backspace, arrow keys, history navigation, etc.
try:
    import readline
    READLINE_AVAILABLE = True
except ImportError:
    # Windows doesn't have readline by default
    try:
        import pyreadline3 as readline
        READLINE_AVAILABLE = True
    except ImportError:
        READLINE_AVAILABLE = False
        print("Warning: readline not available. History and line editing disabled.")

class VariableStore:
    """Abstract interface for variable storage.
    
    This interface allows the REPL to be upgraded from simple dict storage
    to full context management without changing REPL code.
    
    Future implementations can add:
    - Context graphs (linking variables)
    - Scoping rules (local/global context)
    - Generator/relation extraction
    """
    
    def get(self, name):
        """Retrieve variable value."""
        raise NotImplementedError
    
    def set(self, name, value):
        """Store variable value."""
        raise NotImplementedError
    
    def has(self, name):
        """Check if variable exists."""
        raise NotImplementedError
    
    def list_vars(self):
        """List all variables."""
        raise NotImplementedError

class DictVariableStore(VariableStore):
    """Simple dictionary-based variable storage.
    
    This is the minimal implementation. Later, this can be replaced
    with a Context class that adds linking, scoping, etc.
    """
    
    def __init__(self):
        self._vars = {}
    
    def get(self, name):
        return self._vars.get(name)
    
    def set(self, name, value):
        self._vars[name] = value
    
    def has(self, name):
        return name in self._vars
    
    def list_vars(self):
        return list(self._vars.keys())

class GenFam:
    """Helper for creating generator names with subscript syntax.
    
    Allows writing E_{a,b} in comprehensions which becomes E[(a,b)]
    after preprocessing, and this class handles the subscript to generate
    the proper name string 'E_{a,b}'.
    
    Example:
        E = Gen('E')
        E[(0,1)]  # Returns 'E_{0,1}'
        
    This enables:
        gens = [E[(a,b)] for a in range(3) for b in range(3)]
    Which with preprocessing becomes:
        gens = [E_{a,b} for a in range(3) for b in range(3)]
    """
    def __init__(self, base_name='E'):
        self.base_name = base_name
    
    def __getitem__(self, index):
        """Generate generator name from subscript.
        
        Args:
            index: Can be a single value or tuple
            
        Returns:
            String like 'E_{0,1}' or 'E_{a,b}'
        """
        if isinstance(index, tuple):
            # Multiple indices: E[(a,b)] → 'E_{a,b}'
            indices_str = ','.join(str(i) for i in index)
            return f'{self.base_name}_{{{indices_str}}}'
        else:
            # Single index: E[(a,)] or E[a] → 'E_{a}'
            return f'{self.base_name}_{{{index}}}'
    
    def __repr__(self):
        return f"GenFam('{self.base_name}')"

class YawREPL:
    """Interactive REPL for yaw quantum programming.
    
    Features:
    - Variable assignment and retrieval
    - Algebra definition
    - Expression evaluation
    - Pretty printing of results
    
    Architecture allows future upgrades:
    - VariableStore can be swapped for Context
    - Parser can be improved from eval to proper lexer/parser
    - Commands can be extended
    """
    def __init__(self):
        """Initialize REPL with Context."""
        from yaw_prototype import Context
        self.variables = Context()
        self.algebra = None
        self.verbose = False
        self.statement_buffer = []
        self.in_statement = False
        self.current_algebra = None

        self.variables.unlink('$gens', '_gens')
        self.variables.unlink('$rels', '_rels')
        
        # Set up readline for history and line editing
        self._setup_readline()
    
    def _setup_readline(self):
        """Set up readline for command history and line editing.
        
        Features:
        - Command history (up/down arrows)
        - Line editing (backspace, left/right arrows, etc.)
        - History persistence to ~/.yaw_history
        - Tab completion for yaw keywords
        """
        if not READLINE_AVAILABLE:
            return
        
        # Set up history file
        self.history_file = os.path.expanduser('~/.yaw_history')
        
        # Load existing history
        if os.path.exists(self.history_file):
            try:
                readline.read_history_file(self.history_file)
            except Exception as e:
                # Silently ignore history load errors
                pass
        
        # Set history length (number of commands to remember)
        readline.set_history_length(1000)
        
        # Set up tab completion
        readline.parse_and_bind('tab: complete')
        
        # Set up completer function
        readline.set_completer(self._tab_completer)
        
        # Configure delimiters for completion
        # This determines what characters break words for completion
        readline.set_completer_delims(' \t\n=()[]{}@|><+-*/,;:')
    
    def _tab_completer(self, text, state):
        """Tab completion function for yaw REPL.
        
        Completes:
        - yaw keywords and functions
        - Variable names
        - Common operators
        
        Args:
            text: Current text to complete
            state: Completion state (0 for first call, increments)
        
        Returns:
            Completion string or None
        """
        if state == 0:
            # First call: generate list of completions
            
            # Common yaw keywords and functions
            keywords = [
                'char', 'proj', 'qft', 'ctrl', 'tensor',
                'opChannel', 'stChannel', 'opMeasure', 'stMeasure',
                'opBranches', 'stBranches',
                'comm', 'acomm', 'spec', 'minimal_poly',
                'help', 'vars', 'algebra', 'links', 'verbose', 'exit', 'quit',
                'for', 'while', 'if', 'else', 'elif', 'def', 'class',
                'range', 'len', 'print', 'sqrt', 'MixedState', 'mixed'
            ]
            
            # Add variable names from context
            var_names = self.variables.list_vars()
            
            # Combine all possible completions
            all_completions = keywords + var_names
            
            # Filter to those matching the current text
            self.completion_matches = [
                cmd for cmd in all_completions 
                if cmd.startswith(text)
            ]
        
        # Return the next match, or None if no more matches
        try:
            return self.completion_matches[state]
        except IndexError:
            return None
    
    def _save_history(self):
        """Save command history to file."""
        if not READLINE_AVAILABLE:
            return
        
        try:
            readline.write_history_file(self.history_file)
        except Exception as e:
            # Silently ignore history save errors
            pass

    def is_statement(self, line):
        """Check if line starts a Python statement (not expression)."""
        statement_keywords = ['for', 'while', 'if', 'elif', 'else', 'def', 
                             'class', 'with', 'try', 'except', 'finally']
        stripped = line.lstrip()
        return any(stripped.startswith(kw + ' ') or stripped.startswith(kw + ':') 
                   for kw in statement_keywords)

    def _preprocess_unicode(self, line):
        """Replace Unicode operators with ASCII equivalents.
        
        Transforms:
            « → <<  (state conjugation: A « psi)
            » → >>  (operator conjugation: A » B)
            
        This allows using prettier Unicode symbols in tutorials and LaTeX
        without triggering unwanted formatting in editors.
        """
        line = line.replace('«', '<<')
        line = line.replace('»', '>>')
        return line

    def _preprocess_commutators(self, line):
        """Replace [[A, B]] with comm(A, B) and {{A, B}} with acomm(A, B).

        Double brackets/braces avoid conflicts with Python syntax.

        Transforms:
            [[X, Z]] → comm(X, Z)
            {{X, Z}} → acomm(X, Z)
        """
        import re

        # Replace [[...]] with comm(...)
        # Use non-greedy match to handle nested cases
        line = re.sub(r'\[\[(.*?)\]\]', r'comm(\1)', line)

        # Replace {{...}} with acomm(...)
        line = re.sub(r'\{\{(.*?)\}\}', r'acomm(\1)', line)

        return line
    
    def _preprocess_tensor_powers(self, line):
        """Replace (@n) expr with explicit tensor product chain.
        
        Transforms:
            (@3) X → X @ X @ X
            (@5) char(Z, 0) → char(Z, 0) @ char(Z, 0) @ char(Z, 0) @ char(Z, 0) @ char(Z, 0)
            init = (@3) X → init = X @ X @ X
            
        This provides convenient shorthand for tensor powers.
        Creates explicit @ chains for consistency across states and operators.
        """
        import re
        
        # Pattern: (@n) followed by rest of line
        match = re.search(r'\(@(\d+)\)\s+(.+)', line)
        
        if match:
            n = int(match.group(1))
            rest = match.group(2).strip()
            
            # Check if there's an assignment before (@n)
            assignment_match = re.match(r'(.+?=)\s*\(@\d+\)\s+(.+)', line)
            if assignment_match:
                # Handle: var = (@n) expr
                lhs = assignment_match.group(1)
                expr = assignment_match.group(2)
                # Build explicit chain: expr @ expr @ expr @ ...
                tensor_chain = ' @ '.join([f'({expr})'] * n)
                return f"{lhs} {tensor_chain}"
            else:
                # Handle: (@n) expr (no assignment)
                # Build explicit chain: expr @ expr @ expr @ ...
                tensor_chain = ' @ '.join([f'({rest})'] * n)
                return tensor_chain
        
        return line
    
        def replace_tensor_power(match):
            n = match.group(1)
            # Return empty string - we'll append ** n at the end
            return ''
        
        # Check if line contains (@n) pattern
        if re.search(r'\(@\d+\)', line):
            # Extract the number
            match = re.search(r'\(@(\d+)\)', line)
            if match:
                n = match.group(1)
                # Remove the (@n) prefix and add ** n suffix
                line = re.sub(r'\(@\d+\)\s*', '', line)
                # Add ** n to the expression
                # Handle case where line already has other operations
                line = f'({line}) ** {n}'
        
        return line
    
    def is_continuation(self, line):

        """Check if line is a continuation of current statement."""
        stripped = line.strip()

        # Empty lines continue the block
        if not stripped:
            return True

        # Lines starting with whitespace are continuations
        if line and line[0] in (' ', '\t'):
            return True

        # Else/elif/except/finally/elif continue the block
        if (stripped.startswith('else:') or 
            stripped.startswith('elif ') or
            stripped.startswith('except:') or
            stripped.startswith('except ') or
            stripped.startswith('finally:')):
            return True

        return False

    def is_block_continuation(self, line):
        """Check if line continues a block (else, elif, except, finally)."""
        continuation_keywords = ['else', 'elif', 'except', 'finally']
        stripped = line.lstrip()
        return any(stripped.startswith(kw + ':') or stripped.startswith(kw + ' ')
               for kw in continuation_keywords)
    
    def _parse_subscripts(self, line):
        """Parse subscript notation: var_{expr} -> var[(expr)]
        
        Transforms mathematical subscript notation into valid Python syntax.
        
        Examples:
            E_{0,1} -> E[(0,1)]
            E_{a,b} -> E[(a,b)]
            P_{k} -> P[(k)]
        
        This allows subscript syntax to work with:
        - Dictionary access: E[(a,b)] where E is a dict
        - Callable objects: E[(a,b)] where E has __getitem__
        - Generator helpers: GenFam instances
        """
        import re
        # Pattern: word_{content}
        # Replace: word[(content)]
        return re.sub(r'(\w+)_{([^}]+)}', r'\1[(\2)]', line)
    
    def _parse_list_comprehension_assignment(self, line):
        """Parse list comprehension with assignment.
        
        Transforms:
            [var_{k} = expr for k in iterable]
        Into:
            [_assign_and_return('var_{k}', expr) for k in iterable]
        
        Where _assign_and_return stores the variable and returns the value.
        """
        import re
        
        # Pattern: [something = something for ...]
        # This is a heuristic - list comprehensions normally can't have =
        if '[' in line and '=' in line and 'for' in line and ']' in line:
            # Check if it's really a list comp with assignment (not comparison ==)
            # Look for pattern: [ var = expr for var in iterable ]
            pattern = r'\[([^=]+)=([^\]]+for[^\]]+)\]'
            match = re.search(pattern, line)
            
            if match:
                var_part = match.group(1).strip()
                rest_part = match.group(2).strip()
                
                # Replace the assignment with function call
                new_comp = f"[_assign_and_return('{var_part}', {var_part} = {rest_part}]"
                
                # Actually, this won't work because we can't have = in the comprehension
                # Better approach: transform to loop
                return None  # Signal that we need special handling
        
        return line
    
    def _execute_comprehension_with_assignment(self, line):
        """Execute list comprehension with assignment as a for loop.

        Transforms:
            [P_{k} = proj(Z, k) for k in range(2)]
        Into a loop that evaluates subscripts dynamically.
        """
        import re
        
        # Extract the comprehension pattern
        # [var_expr = value_expr for loop_var in iterable]
        pattern = r'\[([^=]+)=([^f]+)for\s+(\w+)\s+in\s+([^\]]+)\]'
        match = re.search(pattern, line)

        if not match:
            return None

        var_template = match.group(1).strip()  # e.g., "P_{k}"
        value_expr = match.group(2).strip()     # e.g., "proj(Z, k)"
        loop_var = match.group(3).strip()       # e.g., "k"
        iterable_expr = match.group(4).strip()  # e.g., "range(2)"

        try:
            namespace = self._build_namespace()

            # Evaluate iterable
            iterable = eval(iterable_expr, namespace, namespace)

            result_list = []

            for loop_value in iterable:
                # Set loop variable in namespace
                namespace[loop_var] = loop_value

                # Substitute subscripts in variable name using current namespace
                var_name = self._substitute_subscripts_in_string(var_template, namespace)

                # Evaluate value expression
                value = eval(value_expr, namespace, namespace)

                # Store variable
                namespace[var_name] = value
                self.variables.set(var_name, value)

                # Append to result
                result_list.append(value)

            return result_list

        except Exception as e:
            return f"Error in list comprehension: {e}"

    def _substitute_subscripts_in_string(self, text, namespace):
        """Substitute subscripts in a string using given namespace.

        Converts P_{k} to P_0 when k=0 in namespace.
        """
        import re
        
        pattern = r'(\w+)_{([^}]+)}'

        def replace_subscript(match):
            var_name = match.group(1)
            subscript_expr = match.group(2)

            try:
                # Evaluate subscript expression with given namespace
                subscript_value = eval(subscript_expr, namespace, namespace)
                return f"{var_name}_{subscript_value}"
            except:
                # If evaluation fails, leave as-is
                return match.group(0)

        return re.sub(pattern, replace_subscript, text)
    
    def _has_subscripts(self, text):
        """Check if text contains subscript notation."""
        return '_{' in text and '}' in text

    def _execute_list_comprehension(self, line):
        """Execute list comprehension with dynamic subscript evaluation.

        Handles:
            [P_{k} for k in range(2)]  → [P_0, P_1]
            [expr(P_{k}) for k in range(3)]
        """
        import re
        
        # Preprocess subscripts first: E_{a,b} -> E[(a,b)]
        line = self._parse_subscripts(line)
        
        # Try direct eval first (for simple cases with GeneratorNameHelper)
        try:
            namespace = self._build_namespace()
            result = eval(line, namespace, namespace)
            if isinstance(result, list):
                return result
        except:
            pass
        
        # Fallback: manual parsing
        # Pattern: [expr for var in iterable]
        pattern = r'\[(.+?)\s+for\s+(\w+)\s+in\s+([^\]]+)\]'
        match = re.search(pattern, line)

        if not match:
            return None

        expr_template = match.group(1).strip()
        loop_var = match.group(2).strip()
        iterable_expr = match.group(3).strip()

        try:
            namespace = self._build_namespace()

            # Evaluate iterable
            iterable = eval(iterable_expr, namespace, namespace)

            result_list = []

            for loop_value in iterable:
                # Set loop variable in namespace
                namespace[loop_var] = loop_value

                # Substitute subscripts in expression
                expr = self._substitute_subscripts_in_string(expr_template, namespace)

                # Evaluate expression
                value = eval(expr, namespace, namespace)

                result_list.append(value)

            return result_list

        except Exception as e:
            return f"Error in list comprehension: {e}"

    def eval_line(self, line):
        """Evaluate a line of yaw code."""
        import re

        line_stripped = line.strip()

        # Preprocess Unicode operators (« → <<, » → >>)
        line_stripped = self._preprocess_unicode(line_stripped)

        if line_stripped == 'help':
            return self._show_help()

        if line_stripped == 'credits':
            return self._show_credits()

        line_stripped = self._preprocess_commutators(line_stripped)
        line_stripped = self._preprocess_tensor_powers(line_stripped)

        # *** CRITICAL: Handle multi-line statements FIRST ***
        if self.in_statement:
            # Empty line ends the statement
            if not line_stripped:
                return self._exec_statement()
            # Block continuation keywords (check on stripped)
            elif self.is_block_continuation(line_stripped):
                # Store original line to preserve indentation
                self.statement_buffer.append(line.rstrip())
                return None
            # *** BUG FIX: Check continuation on ORIGINAL line (has indentation) ***
            elif self.is_continuation(line):  # NOT line_stripped!
                # Store original line to preserve indentation
                self.statement_buffer.append(line.rstrip())
                return None
            else:
                # Dedented line
                result = self._exec_statement()
                new_result = self.eval_line(line)
                if result and new_result:
                    return f"{result}\n{new_result}"
                return result or new_result

        # Check if this line STARTS a new multi-line statement
        if self.is_statement(line_stripped):
            self.in_statement = True
            self.statement_buffer = [line_stripped]
            return None

        # Check for list comprehension with assignment
        if '[' in line_stripped and '=' in line_stripped and 'for' in line_stripped and ']' in line_stripped:
            # Check if this is an outer assignment: var = [inner = expr for ...]
            outer_assign_match = re.match(r'(\w+)\s*=\s*(.+)', line_stripped)

            if outer_assign_match:
                outer_var = outer_assign_match.group(1).strip()
                comprehension_part = outer_assign_match.group(2).strip()

                # Try to execute the comprehension with assignment
                result = self._execute_comprehension_with_assignment(comprehension_part)

                if result is not None:
                    # Store the result in the outer variable
                    self.variables.set(outer_var, result)
                    return None  # Suppress output for assignments
            else:
                # No outer assignment, just execute comprehension
                result = self._execute_comprehension_with_assignment(line_stripped)
                if result is not None:
                    return result

        # Check for list comprehension with subscripts (no assignment)
        if '[' in line_stripped and 'for' in line_stripped and ']' in line_stripped and self._has_subscripts(line_stripped):
            # Check if there's an outer assignment
            outer_assign_match = re.match(r'(\w+)\s*=\s*(.+)', line_stripped)

            if outer_assign_match:
                outer_var = outer_assign_match.group(1).strip()
                comprehension_part = outer_assign_match.group(2).strip()
                result = self._execute_list_comprehension(comprehension_part)
                if result is not None:
                    self.variables.set(outer_var, result)
                    return None  # Suppress output for assignments
            else:
                result = self._execute_list_comprehension(line_stripped)
                if result is not None:
                    return result

        # Now parse subscripts for regular expressions
        line_stripped = self._parse_subscripts(line_stripped)

        # Handle algebra definition: $alg = <...> OR $alg = qudit(d) etc.
        if line_stripped.startswith('$'):
            match = re.match(r'\$(\w+)\s*=\s*(.+)', line_stripped)
            if match:
                var_name = match.group(1)
                rhs = match.group(2).strip()

                # Case 1: Angle bracket notation <gens | rels>
                if rhs.startswith('<') and rhs.endswith('>'):
                    try:
                        result = self._define_algebra(rhs)
                        if result:
                            self.variables.set(f'${var_name}', result)
                            return f"Created algebra: {rhs}"
                    except Exception as e:
                        return f"Error creating algebra: {e}"

                # Case 2: Function call that returns Algebra (e.g., qudit(2))
                else:
                    try:
                        namespace = self._build_namespace()
                        result = eval(rhs, namespace, namespace)

                        # Check if result is an Algebra
                        if isinstance(result, Algebra):
                            # Set the algebra's name for display purposes
                            result.name = var_name
                            
                            self.variables.set(f'${var_name}', result)

                            # Set as current algebra
                            self.current_algebra = result

                            # Make generators AND identity available
                            for gen_name, gen_op in result.generators.items():
                                self.variables.set(gen_name, gen_op)

                            # Add identity
                            self.variables.set('I', result.I)

                            # Display the algebra with its original relation specs
                            gens = ', '.join(result.generator_names)
                            rels_str = ', '.join(result.relation_specs)
                            return f"Created algebra: <{gens} | {rels_str}>"

                        else:
                            return f"Error: {rhs} did not return an Algebra object"

                    except Exception as e:
                        return f"Error evaluating algebra: {e}"

        # Skip empty lines and comments
        if not line_stripped or line_stripped.startswith('#'):
            return None

        # Check for local context operator !
        if '!' in line_stripped and not any(line_stripped.startswith(cmd) for cmd in ['verbose', 'links']):
            return self._eval_with_local_context(line_stripped)

        # Command: link variables
        if line_stripped.startswith('link('):
            try:
                args_str = line_stripped[5:-1]
                args = [arg.strip() for arg in args_str.split(',')]
                if len(args) == 2:
                    return self.variables.link(args[0], args[1])
                else:
                    return "Error: link requires 2 arguments"
            except Exception as e:
                return f"Error in link: {e}"

        # Command: unlink variables
        if line_stripped.startswith('unlink('):
            try:
                args_str = line_stripped[7:-1]
                args = [arg.strip() for arg in args_str.split(',')]
                if len(args) == 2:
                    return self.variables.unlink(args[0], args[1])
                else:
                    return "Error: unlink requires 2 arguments"
            except Exception as e:
                return f"Error in unlink: {e}"

        # Command: show context graph
        if line_stripped == 'links':
            links_str = []
            for parent, children in self.variables.links.items():
                if children:
                    children_str = ', '.join(children)
                    links_str.append(f"  {parent} <- {children_str}")
            if links_str:
                return "Context links:\n" + "\n".join(links_str)
            else:
                return "No links"

        # Command: set verbose mode
        if line_stripped == 'verbose on':
            self.verbose = True
            return "Verbose mode enabled"

        if line_stripped == 'verbose off':
            self.verbose = False
            return "Verbose mode disabled"

        # Command: list variables
        if line_stripped == 'vars':
            vars_list = self.variables.list_vars()
            if not vars_list:
                return "No variables defined"
            return "Variables: " + ", ".join(vars_list)

        # Command: show algebra
        if line_stripped == 'algebra':
            if self.algebra:
                gens = ", ".join(self.algebra.generator_names)
                rels = ", ".join(self.algebra.relation_specs)
                return f"Algebra: <{gens} | {rels}>"
            else:
                return "No algebra defined"

        # Pattern: Algebra definition (old style)
        if line_stripped.startswith('$alg') and '=' in line_stripped:
            return self._define_algebra(line_stripped)

        # Command: Set operations ($gens += _gens)
        if '+=' in line_stripped:
            try:
                target, source = line_stripped.split('+=', 1)
                target = target.strip()
                source = source.strip()

                target_val = self.variables.get(target)
                source_val = self._eval_expr(source)

                if target_val is None:
                    target_val = set()

                if isinstance(target_val, set) and isinstance(source_val, set):
                    new_val = set(target_val)
                    new_val.update(source_val)
                    self.variables.set(target, new_val)
                    return f"Updated {target}: " + "{" + ", ".join(str(x) for x in sorted(new_val)) + "}"
                else:
                    return "Error: += only works with sets"
            except Exception as e:
                return f"Error in set operation: {e}"

        # Pattern: Variable assignment
        # Check if this is a top-level assignment (not inside parentheses)
        if '=' in line_stripped and not any(op in line_stripped for op in ['==', '<=', '>=']):
            # Check if the = is at top level (not inside parentheses)
            if self._is_top_level_assignment(line_stripped):
                return self._assign_variable(line_stripped)

        # Pattern: Expression evaluation
        return self._eval_expr(line_stripped)
    
    def _is_top_level_assignment(self, line):
        """Check if line contains a top-level assignment (= not inside parens/brackets)."""
        depth = 0
        for i, char in enumerate(line):
            if char in '([':
                depth += 1
            elif char in ')]':
                depth -= 1
            elif char == '=' and depth == 0:
                # Check it's not ==, <=, >=
                if i > 0 and line[i-1] in '=<>!':
                    continue
                if i < len(line) - 1 and line[i+1] == '=':
                    continue
                return True
        return False

    def _exec_statement(self):
        """Execute buffered multi-line statement."""
        self.in_statement = False

        # Join all lines preserving original formatting
        full_statement = '\n'.join(self.statement_buffer)
        self.statement_buffer = []
        
        # Preprocess: Replace $alg = ... with alg = ... and inject I, X, Z
        full_statement = self._preprocess_alg_assignments(full_statement)

        # Build namespace
        namespace = self._build_namespace()

        try:
            # Execute the statement
            # CRITICAL: Pass namespace as BOTH globals and locals
            # This allows defined functions to access REPL variables
            exec(full_statement, namespace, namespace)

            # Update variables with any new definitions
            # Extract new variables that were created
            for key, value in namespace.items():
                # Skip builtins and existing items
                if not key.startswith('_') and key not in ['__builtins__']:
                    # Check if this is a new definition or update
                    try:
                        existing = self.variables.get(key)
                        if existing is None or existing != value:
                            self.variables.set(key, value)
                    except:
                        # New variable
                        self.variables.set(key, value)
            
            # Special handling for $alg: inject I, X, Z as globals
            if '$alg' in namespace:
                alg = namespace['$alg']
                if hasattr(alg, 'I'):
                    self.variables.set('I', alg.I)
                if hasattr(alg, 'X') and hasattr(alg.generators, '__getitem__'):
                    if 'X' in alg.generators:
                        self.variables.set('X', alg.generators['X'])
                    elif 'X' in alg.generator_names:
                        self.variables.set('X', alg.X)
                if hasattr(alg, 'Z') and hasattr(alg.generators, '__getitem__'):
                    if 'Z' in alg.generators:
                        self.variables.set('Z', alg.generators['Z'])
                    elif 'Z' in alg.generator_names:
                        self.variables.set('Z', alg.Z)

            return None  # Multi-line statements don't return values

        except Exception as e:
            self.statement_buffer = []
            return f"Error evaluating expression: {e}"
    
    def _is_builtin_name(self, name):
        """Check if name is a built-in function/class we added."""
        builtins = {
            'sqrt', 'tensor', 'tensor_power', 'char', 'conj_op', 'conj_state',
            'TensorState', 'OpChannel', 'opChannel', 'StChannel', 'stChannel',
            'OpMeasurement', 'opMeasure', 'StMeasurement', 'stMeasure',
            'OpBranches', 'opBranches', 'StBranches', 'stBranches',
            'CollapsedState', 'TransformedState', 'QFT', 'qft', 'proj',
            'compose_st_branches', 'compose_op_branches',
            'norm', 'measure',  # Fourier sampling helpers
            'len', 'sum', 'abs', 'max', 'min', 'range', 'enumerate', 'zip',
            'I', 'Encoding', 'MixedState', 'mixed'
        }
        
        # Also check if it's an algebra generator
        if self.algebra and name in self.algebra.generator_names:
            return True
        
        return name in builtins
    
    def _preprocess_alg_assignments(self, code):
        """Preprocess $alg = ... assignments to inject I, X, Z.
        
        Replaces:
            $alg = qudit(d)
        With:
            alg = qudit(d)
            I, X, Z = alg.I, alg.X, alg.Z
        
        This allows the user's desired syntax to work inside functions.
        """
        import re
        
        # Pattern: $alg = ... (possibly with $some_name)
        pattern = r'(\s*)\$(\w+)\s*=\s*(.+)'
        
        def replace_alg(match):
            indent = match.group(1)
            var_name = match.group(2)
            rhs = match.group(3).strip()
            
            # Generate replacement code
            lines = [
                f"{indent}{var_name} = {rhs}",
                f"{indent}I, X, Z = {var_name}.I, {var_name}.X, {var_name}.Z"
            ]
            return '\n'.join(lines)
        
        # Replace all $alg assignments
        result = re.sub(pattern, replace_alg, code)
        return result
    
    def _build_namespace(self):
        """Build namespace for eval/exec (extracted for reuse)."""
        namespace = {}

        # Import yaw functions
        from yaw_prototype import (
            YawOperator, Algebra, State, EigenState, TensorState,
            tensor, tensor_power, char, conj_op, conj_state,
            OpChannel, opChannel, StChannel, stChannel,
            OpMeasurement, opMeasure, StMeasurement, stMeasure,
            OpBranches, opBranches, StBranches, stBranches,
            CollapsedState, TransformedState,
            QFT, qft, proj, proj_algebraic,
            ctrl, ctrl_single,
            qudit, qubit, matrix_units,
            norm, measure,  # Fourier sampling convenience functions
            Encoding, rep, comm, acomm, gnsVec, gnsMat, spec, minimal_poly,
            MixedState, mixed,
            is_projector, is_unitary, is_hermitian  # Operator property checks
        )

        # Add current algebra if exists
        if self.algebra is not None:
            for name, op in self.algebra.generators.items():
                namespace[name] = op
            if hasattr(self.algebra, 'I'):
                namespace['I'] = self.algebra.I

        # *** FIXED: Add user variables from all three Context dictionaries ***
        # Combine temp, global_, and user variables
        for var_dict in [self.variables.temp, self.variables.global_, self.variables.user]:
            for name, value in var_dict.items():
                # Store with original name
                namespace[name] = value

                # If it's a $ variable, also add without $ for Python expressions
                if isinstance(name, str) and name.startswith('$'):
                    namespace[name[1:]] = value

        # SymPy imports
        from sympy import sqrt, exp, pi, I as sympy_I, symbols, Symbol
        namespace['sqrt'] = sqrt
        namespace['exp'] = exp
        namespace['pi'] = pi
        namespace['symbols'] = symbols
        namespace['Symbol'] = Symbol

        # Python builtins
        namespace['len'] = len
        namespace['sum'] = sum
        namespace['abs'] = abs
        namespace['max'] = max
        namespace['min'] = min
        namespace['range'] = range
        namespace['enumerate'] = enumerate
        namespace['zip'] = zip
        namespace['type'] = type
        namespace['list'] = list
        namespace['dict'] = dict
        namespace['set'] = set
        namespace['None'] = None
        namespace['True'] = True
        namespace['False'] = False
        
        # Random module for classical randomness
        import random
        namespace['random'] = random

        # Yaw functions
        namespace['tensor'] = tensor
        namespace['tensor_power'] = tensor_power
        namespace['char'] = char
        namespace['conj_op'] = conj_op
        namespace['conj_state'] = conj_state
        namespace['OpChannel'] = OpChannel
        namespace['opChannel'] = opChannel
        namespace['StChannel'] = StChannel
        namespace['stChannel'] = stChannel
        namespace['OpMeasurement'] = OpMeasurement
        namespace['opMeasure'] = opMeasure
        namespace['StMeasurement'] = StMeasurement
        namespace['stMeasure'] = stMeasure
        namespace['OpBranches'] = OpBranches
        namespace['opBranches'] = opBranches
        namespace['StBranches'] = StBranches
        namespace['stBranches'] = stBranches
        namespace['QFT'] = QFT
        namespace['qft'] = qft
        namespace['tensor_qft'] = tensor_qft
        namespace['proj'] = proj
        namespace['proj_algebraic'] = proj_algebraic
        namespace['ctrl'] = ctrl
        namespace['ctrl_single'] = ctrl_single
        namespace['qudit'] = qudit
        namespace['qubit'] = qubit
        namespace['matrix_units'] = matrix_units
        namespace['norm'] = norm  # Fourier sampling helper
        namespace['measure'] = measure  # Fourier sampling helper
        namespace['Encoding'] = Encoding
        namespace['rep'] = rep
        namespace['comm'] = comm
        namespace['acomm'] = acomm
        namespace['gnsVec'] = gnsVec
        namespace['gnsMat'] = gnsMat
        namespace['spec'] = spec
        namespace['minimal_poly'] = minimal_poly
        namespace['MixedState'] = MixedState
        namespace['mixed'] = mixed
        namespace['is_projector'] = is_projector
        namespace['is_unitary'] = is_unitary
        namespace['is_hermitian'] = is_hermitian
        
        # Local context helper for use inside functions
        # Allows: local(expr, Z=I, X=something) instead of (expr ! Z = I, X = something)
        def local(expr, **bindings):
            """Evaluate expression with local variable bindings.
            
            Usage inside functions:
                def equiv(A):
                    return local(A, Z=I)
            
            This is equivalent to the REPL syntax: A ! Z = I
            
            For operators, this performs substitution of generators.
            For other expressions, this evaluates with temporary variable bindings.
            """
            from yaw_prototype import YawOperator
            import sympy
            
            # If expr is a YawOperator, substitute generators
            if isinstance(expr, YawOperator):
                # Get the symbolic expression
                expr_sympy = expr._expr
                
                # Build substitution dict: generator symbol -> replacement expression
                subs_dict = {}
                for name, value in bindings.items():
                    # Convert generator name to SymPy symbol for substitution
                    from sympy.physics.quantum import Operator
                    gen_symbol = Operator(name)
                    
                    # The binding value should be an operator
                    if isinstance(value, YawOperator):
                        subs_dict[gen_symbol] = value._expr
                    else:
                        # Try to convert to sympy expression
                        subs_dict[gen_symbol] = value
                
                # Perform substitution
                new_expr = expr_sympy.subs(subs_dict)
                
                # Create new YawOperator with the algebra (use the expr's algebra if available)
                algebra = expr.algebra if hasattr(expr, 'algebra') else None
                result = YawOperator(new_expr, algebra)
                
                # Normalize using the algebra
                if algebra:
                    return algebra.normalize(result, force=True)
                else:
                    return result
            
            # For string expressions, evaluate with bindings
            elif isinstance(expr, str):
                # Build namespace with current context + bindings  
                eval_namespace = dict(namespace)
                eval_namespace.update(bindings)
                result = eval(expr, eval_namespace)
                return result
            
            # Otherwise just return as-is
            return expr
        
        namespace['local'] = local
        
        # Numerical evaluation shortcut
        def N(expr, precision=15):
            """Numerically evaluate expression.
            
            Usage:
                N(expr)              # Evaluate to ~15 decimal places
                N(expr, precision=6) # Evaluate to 6 decimal places
            
            This is a shortcut for complex(expr._expr.evalf()).
            Returns a Python complex number (or float if imaginary part is 0).
            
            Examples:
                *> mySum = local(sum(w**k for k in range(10)), w=exp(2*I*pi/10))
                *> N(mySum)
                0.0
            """
            from yaw_prototype import YawOperator
            
            # Handle YawOperator
            if isinstance(expr, YawOperator):
                # Get the underlying SymPy expression
                sympy_expr = expr._expr
                # Evaluate it numerically
                evaled = sympy_expr.evalf(precision)
                # Try to convert to complex
                try:
                    result = complex(evaled)
                except (TypeError, ValueError):
                    # If it can't be converted to complex, just return the evaled expression
                    return evaled
            # Handle SymPy expressions that have .evalf()
            elif hasattr(expr, 'evalf'):
                try:
                    evaled = expr.evalf(precision)
                    result = complex(evaled)
                except (TypeError, ValueError):
                    return evaled
            # Handle regular numbers
            elif isinstance(expr, (int, float, complex)):
                return expr
            else:
                # Try to convert directly
                try:
                    return complex(expr)
                except:
                    return expr
            
            # Clean up the result
            # Round very small values to exactly 0
            threshold = 10**(-precision + 2)
            
            real_part = result.real
            imag_part = result.imag
            
            if abs(real_part) < threshold:
                real_part = 0.0
            if abs(imag_part) < threshold:
                imag_part = 0.0
            
            # Return float if imaginary part is 0, otherwise complex
            if imag_part == 0:
                return real_part
            else:
                return complex(real_part, imag_part)
        
        namespace['N'] = N
        
        # Tolerance-based equality for numerical comparisons
        def isclose(a, b, rtol=1e-09, atol=1e-12):
            """Check if two values are approximately equal.
            
            Uses numpy.isclose() under the hood for element-wise comparison.
            Works with scalars, arrays, and yaw objects.
            
            Parameters:
                a, b: Values to compare
                rtol: Relative tolerance (default: 1e-09)
                atol: Absolute tolerance (default: 1e-12)
            
            Returns:
                Boolean or array of booleans
            
            Examples:
                *> isclose(1.0, 1.0000000000000002)
                True
                
                *> [isclose(f(k), f(k+p)) for k in range(d-p)]
                [True, True, True, True]
            """
            import numpy as np
            from yaw_prototype import State, YawOperator
            
            # Extract numerical values if they're yaw objects
            if isinstance(a, State):
                a = a.val
            if isinstance(b, State):
                b = b.val
            if isinstance(a, YawOperator):
                # For operators, compare matrix representations if possible
                if hasattr(a, 'matrix') and hasattr(b, 'matrix'):
                    a = a.matrix()
                    b = b.matrix()
                else:
                    # Fall back to expression comparison
                    return a == b
            
            # Use numpy's isclose for numerical comparison
            return np.isclose(a, b, rtol=rtol, atol=atol)
        
        namespace['isclose'] = isclose
        
        # Generator name helpers for comprehensions
        # These enable syntax like: gens = [E[(a,b)] for a in range(3) for b in range(3)]
        # which with preprocessing becomes: gens = [E_{a,b} for ...]
        namespace['GenFam'] = GenFam  # The class itself
        namespace['E'] = GenFam('E')
        namespace['F'] = GenFam('F')
        namespace['G'] = GenFam('G')

        return namespace
    
    def _eval_with_local_context(self, line):
        """Evaluate expression with local context: expr ! context

        Examples:
            A**3 ! pow(3)
            B*A ! <A, B | anti, herm, unit>
            X + Y ! X = 2, Y = 3
        """
        try:
            # Split on ! (only first occurrence)
            parts = line.split('!', 1)
            if len(parts) != 2:
                return "Error: Invalid local context syntax"

            expr_str = parts[0].strip()
            context_str = parts[1].strip()

            # Parse context specification
            if context_str.startswith('<') and context_str.endswith('>'):
                # Full algebra spec: <A, B | anti, herm, unit>
                return self._eval_with_algebra_spec(expr_str, context_str)
            elif self._is_bindings_context(context_str):
                # Variable bindings: X = 2, Y = 3
                return self._eval_with_bindings(expr_str, context_str)
            else:
                # Single relation or comma-separated relations: pow(3), anti
                return self._eval_with_relations(expr_str, context_str)

        except Exception as e:
            return f"Error in local context: {e}"
    
    def _is_bindings_context(self, context_str):
        """Check if context string is variable bindings vs relations.
        
        Variable bindings: X = 2, Y = 3
        Relations: pow(3), anti, E_{0,1}.d = E_{1,0}
        
        Heuristic: If all items are simple assignments (identifier = value),
        it's bindings. If any item has operators/functions on LHS or is a
        known relation keyword, it's relations.
        """
        if '=' not in context_str:
            return False
        
        # Split by comma to get individual items
        items = [item.strip() for item in context_str.split(',')]
        
        for item in items:
            if '=' not in item:
                # Items like 'pow(3)', 'herm' without '=' are relations, not bindings
                return False
            
            lhs, _ = item.split('=', 1)
            lhs = lhs.strip()
            
            # Simple identifier pattern (letters, digits, underscore only)
            # If LHS has operators like *, ., etc., it's a relation
            if any(op in lhs for op in ['*', '+', '-', '/', '.', '(', ')', '[', ']', '{', '}', '^', '†']):
                return False
        
        return True
    
    def _eval_with_bindings(self, expr_str, bindings_str):
        """Evaluate expression with local variable bindings.
        
        Example: X + Y ! X = 2, Y = 3
        
        This creates a temporary namespace where the specified variables
        have the given values, evaluates the expression, then discards
        the bindings.
        
        For YawOperators, performs actual substitution using local().
        """
        # Parse bindings (comma-separated)
        bindings = {}
        for binding in bindings_str.split(','):
            binding = binding.strip()
            if '=' not in binding:
                return f"Error: Invalid binding '{binding}' - expected format 'var = value'"
            
            var_name, value_str = binding.split('=', 1)
            var_name = var_name.strip()
            value_str = value_str.strip()
            
            # Evaluate the value expression
            try:
                value = self._eval_expr(value_str)
                if isinstance(value, str) and value.startswith("Error"):
                    return value
                bindings[var_name] = value
            except Exception as e:
                return f"Error evaluating binding '{var_name} = {value_str}': {e}"
        
        # First, try to evaluate the expression
        expr_result = self._eval_expr(expr_str)
        
        # If result is an error, return it
        if isinstance(expr_result, str) and expr_result.startswith("Error"):
            return expr_result
        
        # If the result is a YawOperator, use local() for substitution
        from yaw_prototype import YawOperator
        if isinstance(expr_result, YawOperator):
            # Use local() function from namespace
            namespace = self._build_namespace()
            local_func = namespace['local']
            try:
                result = local_func(expr_result, **bindings)
                return result
            except Exception as e:
                return f"Error applying bindings: {e}"
        
        # Otherwise, build namespace with bindings and evaluate
        namespace = self._build_namespace()
        namespace.update(bindings)
        
        # Preprocess subscripts in expression
        expr_str = self._parse_subscripts(expr_str)
        
        # Evaluate expression
        try:
            result = eval(expr_str, namespace, namespace)
            return result
        except Exception as e:
            return f"Error evaluating expression: {e}"

    def _eval_with_relations(self, expr_str, rels_str):
        """Evaluate expression with local relations.

        Example: A**3 ! pow(3)
        """
        # Extract generators from expression
        parsed_gens = self._parse_generators_from_string(expr_str)

        if not parsed_gens:
            return "Error: No generators found in expression"

        # Parse relations (comma-separated)
        rels = [r.strip() for r in rels_str.split(',')]

        # Create temporary algebra
        temp_algebra = Algebra(
            gens=sorted(parsed_gens),  # Sort for consistency
            rels=rels
        )

        # Build namespace with temporary algebra generators
        namespace = {}
        namespace.update(temp_algebra.generators)
        namespace['I'] = temp_algebra.I

        # Add built-in functions
        namespace['sqrt'] = sqrt
        namespace['len'] = len
        namespace['sum'] = sum
        namespace['abs'] = abs
        namespace['max'] = max
        namespace['min'] = min
        namespace['range'] = range
        namespace['enumerate'] = enumerate
        namespace['zip'] = zip
        
        namespace['tensor'] = tensor
        namespace['tensor_power'] = tensor_power
        namespace['char'] = char
        namespace['conj_op'] = conj_op
        namespace['conj_state'] = conj_state
        namespace['TensorState'] = TensorState
        namespace['OpChannel'] = OpChannel
        namespace['opChannel'] = opChannel
        namespace['StChannel'] = StChannel
        namespace['stChannel'] = stChannel
        namespace['OpMeasurement'] = OpMeasurement
        namespace['opMeasure'] = opMeasure
        namespace['StMeasurement'] = StMeasurement
        namespace['stMeasure'] = stMeasure
        namespace['CollapsedState'] = CollapsedState
        namespace['OpBranches'] = OpBranches
        namespace['opBranches'] = opBranches
        namespace['StBranches'] = StBranches
        namespace['stBranches'] = stBranches
        namespace['compose_st_branches'] = compose_st_branches
        namespace['compose_op_branches'] = compose_op_branches
        namespace['QFT'] = QFT
        namespace['qft'] = qft
        namespace['tensor_qft'] = tensor_qft
        namespace['proj'] = proj
        namespace['ctrl'] = ctrl
        namespace['ctrl_single'] = ctrl_single
        
        # Evaluate expression
        result = eval(expr_str, namespace, namespace)

        # Normalize with temporary algebra
        if isinstance(result, YawOperator):
            result = result.normalize(verbose=self.verbose)

        return result

    def _eval_with_algebra_spec(self, expr_str, algebra_str):
        """Evaluate expression with full algebra specification.

        Example: B*A ! <A, B | anti, herm, unit>
        """
        # Parse algebra spec: <A, B | anti, herm, unit>
        if not algebra_str.startswith('<') or not algebra_str.endswith('>'):
            return "Error: Algebra spec must be in format <gens | rels>"

        spec = algebra_str[1:-1]  # Remove < >

        if '|' not in spec:
            return "Error: Algebra must have format <gens | rels>"

        gens_part, rels_part = spec.split('|', 1)

        # Parse generators and relations
        gens = [g.strip() for g in gens_part.split(',')]
        rels = [r.strip() for r in rels_part.split(',')]

        # Create temporary algebra
        temp_algebra = Algebra(gens=gens, rels=rels)

        # Build namespace with temporary algebra generators
        namespace = {}
        namespace.update(temp_algebra.generators)
        namespace['I'] = temp_algebra.I

        # Add built-in functions
        namespace['sqrt'] = sqrt
        namespace['len'] = len
        namespace['sum'] = sum
        namespace['abs'] = abs
        namespace['max'] = max
        namespace['min'] = min
        namespace['range'] = range
        namespace['enumerate'] = enumerate
        namespace['zip'] = zip
        
        namespace['tensor'] = tensor
        namespace['tensor_power'] = tensor_power
        namespace['char'] = char
        namespace['conj_op'] = conj_op
        namespace['conj_state'] = conj_state
        namespace['TensorState'] = TensorState
        namespace['OpChannel'] = OpChannel
        namespace['opChannel'] = opChannel
        namespace['StChannel'] = StChannel
        namespace['stChannel'] = stChannel
        namespace['OpMeasurement'] = OpMeasurement
        namespace['opMeasure'] = opMeasure
        namespace['StMeasurement'] = StMeasurement
        namespace['stMeasure'] = stMeasure
        namespace['CollapsedState'] = CollapsedState
        namespace['OpBranches'] = OpBranches
        namespace['opBranches'] = opBranches
        namespace['StBranches'] = StBranches
        namespace['stBranches'] = stBranches
        namespace['compose_st_branches'] = compose_st_branches
        namespace['compose_op_branches'] = compose_op_branches
        namespace['QFT'] = QFT
        namespace['qft'] = qft
        namespace['tensor_qft'] = tensor_qft
        namespace['proj'] = proj
        namespace['ctrl'] = ctrl
        namespace['ctrl_single'] = ctrl_single
        
        # Evaluate expression
        result = eval(expr_str, namespace, namespace)

        # Normalize with temporary algebra
        if isinstance(result, YawOperator):
            result = result.normalize(verbose=self.verbose)

        return result
    
    def _parse_algebra_component(self, component_str, component_type):
        """Parse a component of algebra spec (generators or relations).
        
        Supports two formats:
            1. Comma-separated list: "X, Z, W"
            2. List comprehension: "[f'E_{{{a},{b}}}' for a in range(3) for b in range(3)]"
        
        Args:
            component_str: String to parse
            component_type: "generators" or "relations" (for error messages)
            
        Returns:
            List of strings, or error message string
        """
        component_str = component_str.strip()
        
        # Check if it's a comprehension (starts with '[')
        if component_str.startswith('['):
            # It's a comprehension - evaluate it
            try:
                # Preprocess subscripts: E_{a,b} → E[(a,b)]
                # This allows natural syntax with GenFam
                preprocessed = self._parse_subscripts(component_str)
                
                # Build namespace for evaluation
                namespace = self._build_namespace()
                
                # Evaluate the comprehension
                result = eval(preprocessed, namespace, namespace)
                
                # Check that result is a list
                if not isinstance(result, list):
                    return f"Error: {component_type} comprehension must evaluate to a list, got {type(result).__name__}"
                
                # Convert all elements to strings
                return [str(item) for item in result]
                
            except Exception as e:
                return f"Error evaluating {component_type} comprehension: {e}"
        else:
            # Traditional comma-separated list
            return [g.strip() for g in component_str.split(',')]
    
    def _define_algebra(self, line):
        """Parse and define algebra from line like: $alg = <X, Z | herm, unit, anti>"""
        try:
            # Extract the <...> part
            if '<' not in line or '>' not in line:
                return "Error: Algebra must be in format <gens | rels>"

            start = line.index('<')
            end = line.index('>')
            spec = line[start+1:end]

            # Split on |
            if '|' not in spec:
                return "Error: Algebra must have format <gens | rels>"

            gens_part, rels_part = spec.split('|', 1)
            gens_part = gens_part.strip()
            rels_part = rels_part.strip()

            # FEATURE 2: Parse generators (support comprehensions)
            gens = self._parse_algebra_component(gens_part, "generators")
            if isinstance(gens, str) and gens.startswith("Error"):
                return gens
            
            # FEATURE 2: Parse relations (support comprehensions)
            rels = self._parse_algebra_component(rels_part, "relations")
            if isinstance(rels, str) and rels.startswith("Error"):
                return rels

            # Create algebra
            self.algebra = Algebra(gens, rels)
            self.algebra.name = '$alg'  # Set name for display purposes

            # Store generators in variable space
            for name in gens:
                self.variables.set(name, self.algebra.generators[name])

            # Store identity
            self.variables.set('I', self.algebra.I)

            # *** CHANGE 2: Reset _gens and _rels (don't accumulate) ***
            _gens_set = set(gens)
            _rels_set = set(rels)

            self.variables.set('_gens', _gens_set)
            self.variables.set('_rels', _rels_set)

            # *** CHANGE 1: Automatically set $gens and $rels ***
            self.variables.set('$gens', set(_gens_set))  # Copy
            self.variables.set('$rels', set(_rels_set))  # Copy

            # Store algebra reference
            self.variables.set('$alg', self.algebra)

            return f"Created algebra: <{', '.join(gens)} | {', '.join(rels)}>"

        except Exception as e:
            return f"Error defining algebra: {e}"
    
    def _assign_variable(self, line):
        """Parse and execute variable assignment."""
        try:
            # Split on first =
            if '=' not in line:
                return "Error: No assignment operator"

            name_part, expr = line.split('=', 1)
            name_part = name_part.strip()
            expr = expr.strip()

            # Evaluate expression
            result = self._eval_expr(expr)

            # Check if result is an error message
            if isinstance(result, str) and result.startswith("Error"):
                return result

            # *** NEW: Handle tuple unpacking ***
            if ',' in name_part:
                # Tuple unpacking: a, b = (x, y)
                names = [n.strip() for n in name_part.split(',')]

                # Check if result is a tuple
                if not isinstance(result, tuple):
                    return f"Error: Cannot unpack non-tuple to {len(names)} variables"

                if len(names) != len(result):
                    return f"Error: Cannot unpack {len(result)} values to {len(names)} variables"

                # Store each variable
                for name, value in zip(names, result):
                    # If assigning an Algebra, set its name for display purposes
                    if isinstance(value, Algebra):
                        value.name = name
                    self.variables.set(name, value)

                # Return None to suppress output (like Python REPL)
                return None
            else:
                # Simple assignment: a = x
                name = name_part

                # If assigning an Algebra, set its name for display purposes
                if isinstance(result, Algebra):
                    result.name = name
                
                # Store in variables
                self.variables.set(name, result)

                # Return None to suppress output (like Python REPL)
                return None

        except Exception as e:
            return f"Error in assignment: {e}"
    
    def _eval_expr(self, expr):
        """Evaluate expression with current context."""
        try:
            # FEATURE 1: Preprocess subscript notation
            expr_preprocessed = self._parse_subscripts(expr)
            
            # DEBUG: Print if expression changed
            if self.verbose and expr_preprocessed != expr:
                print(f"DEBUG: Subscript preprocessing:")
                print(f"  Before: {expr}")
                print(f"  After:  {expr_preprocessed}")
            
            expr = expr_preprocessed
            
            # Special case: Direct variable lookup for $ and _ prefixed names
            expr_stripped = expr.strip()
            if expr_stripped.startswith('$') or expr_stripped.startswith('_'):
                if expr_stripped.replace('_', '').replace('$', '').replace('[', '').replace(']', '').replace(',', '').replace(' ', '').isalnum():
                    result = self.variables.get(expr_stripped)
                    if result is not None:
                        return result
                    else:
                        return f"Variable '{expr_stripped}' not defined"

            # Parse generators from string BEFORE evaluation
            parsed_gens = self._parse_generators_from_string(expr_stripped)
            if parsed_gens:
                self.variables.set('_gens', parsed_gens)
                self.variables.set('_rels', set())

            # Check if $gens and _gens are linked
            is_linked = '_gens' in self.variables.links.get('$gens', set())

            # If linked, merge _gens into $gens (accumulation)
            if is_linked and parsed_gens:
                current_dollar_gens = self.variables.get('$gens')
                if current_dollar_gens is None:
                    current_dollar_gens = set()
                else:
                    current_dollar_gens = set(current_dollar_gens)  # Copy

                current_dollar_gens.update(parsed_gens)
                self.variables.set('$gens', current_dollar_gens)

            # *** NEW: Check for undefined operators if unlinked (strict mode) ***
            if not is_linked and parsed_gens:
                # Get list of all defined names
                defined_names = set()

                # Add algebra generators
                if self.algebra:
                    defined_names.update(self.algebra.generator_names)
                    defined_names.add('I')

                # Add $gens
                dollar_gens = self.variables.get('$gens')
                if dollar_gens:
                    defined_names.update(dollar_gens)

                # Add user variables
                for name in self.variables.list_vars():
                    if not name.startswith('$') and not name.startswith('_'):
                        defined_names.add(name)

                # Check which parsed generators are undefined
                undefined = parsed_gens - defined_names

                if undefined:
                    undefined_list = sorted(undefined)
                    if len(undefined_list) == 1:
                        return f"Error: Undefined operator '{undefined_list[0]}'"
                    else:
                        undefined_str = ", ".join(f"'{x}'" for x in undefined_list)
                        return f"Error: Undefined operators {undefined_str}"


            # Build namespace
            namespace = self._build_namespace()
        
            # Evaluate
            result = eval(expr, namespace, namespace)
            
            return result

        except Exception as e:
            return f"Error evaluating expression: {e}"

    def _parse_generators_from_string(self, expr_str):
        """Extract operator-like identifiers from expression string.

        Captures uppercase single letters and multi-char identifiers that
        look like operators (not functions like sqrt, char, etc.)
        
        IMPORTANT: Excludes identifiers that appear after a dot (attribute access),
        since those are being accessed from an object, not as bare operators.
        Also excludes identifiers inside string literals and local() function calls.

        Returns:
            Set of generator names found in the string
        """
        import re

        if self.verbose:
            print(f"DEBUG _parse_generators_from_string: {expr_str[:80]}")

        # First, find and remove string literals to avoid matching identifiers inside them
        # This handles both 'single' and "double" quoted strings
        string_pattern = r'''('[^']*'|"[^"]*")'''
        
        # Replace strings with placeholders to preserve positions
        expr_without_strings = re.sub(string_pattern, '""', expr_str)
        
        if self.verbose and expr_without_strings != expr_str:
            print(f"  After removing strings: {expr_without_strings[:80]}")
        
        # Second, find and remove local() function calls
        # We need to handle nested parentheses properly
        # Pattern: local(...) where ... can contain nested parens
        # Use a simpler approach: find 'local(' and match to its closing paren
        expr_cleaned = expr_without_strings
        
        # Find all local( positions
        local_pattern = r'local\s*\('
        matches = list(re.finditer(local_pattern, expr_cleaned))
        
        if self.verbose and matches:
            print(f"  Found {len(matches)} local() calls")
        
        # For each match, find its closing paren and mask the contents
        replacements = []
        for match in matches:
            start = match.end()  # Position after 'local('
            # Find matching closing paren
            depth = 1
            pos = start
            while pos < len(expr_cleaned) and depth > 0:
                if expr_cleaned[pos] == '(':
                    depth += 1
                elif expr_cleaned[pos] == ')':
                    depth -= 1
                pos += 1
            
            if depth == 0:
                # Found matching closing paren
                # Store the range to replace: from match.start() to pos (inclusive)
                replacements.append((match.start(), pos))
        
        # Replace local(...) calls with placeholders
        for start, end in reversed(replacements):  # Reverse to maintain positions
            expr_cleaned = expr_cleaned[:start] + 'local()' + expr_cleaned[end:]
        
        if self.verbose and replacements:
            print(f"  After masking local() calls: {expr_cleaned[:80]}")
        
        # Use expr_cleaned for pattern matching
        
        # First, find all attribute access patterns and mark those identifiers
        # Pattern: something.identifier (we want to exclude the identifier part)
        attr_access_pattern = r'\.([A-Za-z_][A-Za-z0-9_]*)'
        attr_accessed = set(re.findall(attr_access_pattern, expr_cleaned))
        
        # Also find identifiers that are followed by a dot (these are objects, not operators)
        # Pattern: identifier.something (we want to exclude the identifier part too)
        object_pattern = r'\b([A-Za-z_][A-Za-z0-9_]*)\.'
        objects_with_attrs = set(re.findall(object_pattern, expr_cleaned))
        
        # Find identifiers followed by [ (these are being indexed, not used as operators)
        # Pattern: identifier[ (e.g., A[(0,1)] or list[0])
        indexing_pattern = r'\b([A-Za-z_][A-Za-z0-9_]*)\['
        indexed_objects = set(re.findall(indexing_pattern, expr_cleaned))

        # Find all identifiers (sequences of letters, digits, underscores)
        identifiers = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', expr_cleaned)

        # Filter to keep only operator-like names
        # Exclude: built-in functions, commands, known variables
        exclude = {
            'sqrt', 'tensor', 'tensor_power', 'char', 'conj_op', 'conj_state',
            'TensorState', 'link', 'unlink', 'verbose', 'on', 'off',
            'vars', 'links', 'algebra', 'I', 'qudit', 'qubit',
            'OpChannel', 'opChannel', 'StChannel', 'stChannel',
            'opUpdate', 'stUpdate', 'TransformedState',
            'OpMeasurement', 'opMeasure', 'StMeasurement', 'stMeasure',
            'CollapsedState', 'OpBranches', 'opBranches', 'StBranches', 'stBranches',
            'compose_st_branches', 'compose_op_branches', 'QFT', 'qft', 'tensor_qft',
            'proj', 'ctrl', 'ctrl_single', 'type', 'list', 'dict', 'set',
            'Encoding', 'None', 'True', 'False', 'rep', 'comm', 'acomm',
            'gnsVec', 'gnsMat', 'spec', 'minimal_poly', 'MixedState', 'mixed',
            'GenFam', 'E', 'F', 'G',  # Generator family helpers
            'local', 'N', 'isclose', 'random'  # Helper functions and modules
        }

        gens = set()
        for name in identifiers:
            # Skip excluded names
            if name in exclude:
                continue

            # Skip $ and _ prefixed (those are context variables)
            if name.startswith('_') or name.startswith('$'):
                continue
            
            # Skip identifiers accessed as attributes (e.g., X in alg1.X)
            if name in attr_accessed:
                continue
            
            # Skip objects that have attribute access (e.g., alg1 in alg1.X)
            if name in objects_with_attrs:
                continue
            
            # Skip objects being indexed (e.g., A in A[(0,1)] or list in list[0])
            if name in indexed_objects:
                continue

            # Skip lowercase function-like names (heuristic: likely functions)
            # But keep single uppercase letters (X, Y, Z, A, B, etc.)
            if len(name) == 1 and name.isupper():
                gens.add(name)
            elif not name.islower():  # Mixed case or all caps
                gens.add(name)

        return gens
        
    def _clean_number(self, num, decimals=10):
        """Clean numerical noise from numbers.
        
        Args:
            num: Number to clean (int, float, or complex)
            decimals: Number of decimal places to round to (default 10)
        
        Returns:
            Cleaned number with trailing zeros removed
        """
        if isinstance(num, (int, bool)):
            return num
        
        if isinstance(num, float):
            # Round to remove noise
            rounded = round(num, decimals)
            # If very close to zero, make it exactly zero
            if abs(rounded) < 10**(-decimals):
                return 0.0
            # If very close to an integer, make it an integer
            if abs(rounded - round(rounded)) < 10**(-decimals):
                return int(round(rounded))
            return rounded
        
        if isinstance(num, complex):
            # Clean real and imaginary parts separately
            real = self._clean_number(num.real, decimals)
            imag = self._clean_number(num.imag, decimals)
            
            # If imaginary part is zero, return just the real part
            if imag == 0:
                return real
            # If real part is zero, return just imaginary
            if real == 0:
                return complex(0, imag)
            
            return complex(real, imag)
        
        return num
    
    def _display(self, result):
        """Pretty print result.

        Handles different types of yaw objects with appropriate formatting.
        """
        if result is None:
            return

        if isinstance(result, str):
            print(result)
            return

        # Handle sets (for $gens, _gens, etc.)
        if isinstance(result, set):
            if not result:
                print("∅")
            else:
                print("{" + ", ".join(str(x) for x in sorted(result)) + "}")
            return

        # *** NEW: Handle TensorSum ***
        from yaw_prototype import TensorSum
        if isinstance(result, TensorSum):
            print(str(result))
            return

        # Handle TensorProduct
        from yaw_prototype import TensorProduct
        if isinstance(result, TensorProduct):
            print(str(result))
            return

        # Track states in _state
        from yaw_prototype import State
        if isinstance(result, State):
            self.variables.set('_state', result)
            print(result)
            return

        from yaw_prototype import YawOperator
        if isinstance(result, YawOperator):
            # Show normalized form
            normalized = result.normalize(verbose=self.verbose)
            print(normalized)
            return

        if isinstance(result, (int, float, complex)):
            print(self._clean_number(result))
            return

        # Default: use Python's str
        print(result)
    
    def run(self):
        """Main REPL loop."""
        print()
        print("           ┓    ┓     •          ")
        print("┓┏┏┓┓┏┏  ┏┓┃┏┓┏┓┣┓┏┓┏┓┓┏  ┓┏┏┏┓┓┏")
        print("┗┫┗┻┗┻┛  ┗┻┗┗┫┗ ┗┛┛ ┗┻┗┗  ┗┻┛┗┻┗┫")
        print(" ┛           ┛                  ┛")
        print("Yaw: Algebraic Quantum Programming")
        print("Version 0.1.1")
        print("Torsor Labs, 2026")
        print("─" * 60)
        print("Type 'help' or 'credits' for more information.")
        
        # Show readline status
        if READLINE_AVAILABLE:
            print("[Command history saved to ~/.yaw_history]")
        
        try:
            while True:
                try:
                    # Get input
                    if self.in_statement:
                        prompt = "... "
                    else:
                        prompt = "*> "

                    line = input(prompt)
                        
                    # Check for exit
                    if line.strip() in ['exit', 'quit']:
                        print("Goodbye!")
                        break
                    
                    # Evaluate
                    result = self.eval_line(line)
                    
                    # Display
                    if result is not None:
                        self._display(result)
                        
                except KeyboardInterrupt:
                    print("\nUse 'exit' or 'quit' to exit")
                except EOFError:
                    print("\nGoodbye!")
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    if self.verbose:
                        traceback.print_exc()
        finally:
            # Always save history on exit
            self._save_history()

    def _show_credits(self):
        """Display credits."""
        credit_text = """
────────────────────────────────────────────────────────────
    Made by David Wakeham for Torsor Labs.
────────────────────────────────────────────────────────────
   """
        return credit_text.strip()
                    
    def _show_help(self):
        """Display help information."""
        help_text = """
────────────────────────────────────────────────────────────
    BASIC COMMANDS:
      algebra           Show current algebra
      exit, quit        Exit the REPL
      help              Show this help message
      vars              List all variables/operators
      links             Show context links
      verbose on/off    Toggle verbose normalization output
    
    READLINE FEATURES (if available):
      ↑ / ↓             Navigate command history
      ← / →             Move cursor in line
      Backspace         Delete character before cursor
      Tab               Auto-complete keywords and variables
      Ctrl+A / Ctrl+E   Jump to beginning/end of line
      Ctrl+K            Delete from cursor to end of line
      Ctrl+U            Delete entire line
      History saved to: ~/.yaw_history

    ALGEBRA DEFINITION:
      $alg = <X, Z | herm, unit, anti>
        Generators: X, Z, ...
        Relations: herm, unit, anti, pow(d), braid(ω)

      Examples:
        $alg = <X, Z | herm, unit, anti>                          # Qubit (Pauli)
        $alg = <X, Z | herm, unit, pow(3), braid(exp(2*pi*I/3))>  # Qutrit

    LOCAL CONTEXT (temporary algebra):
      expr ! <A, B | relations>   # Full algebra spec
      expr ! pow(3), herm         # Just relations

      Examples:
        A**3 ! pow(3)              # Evaluate AÂ³ with pow(3) relation
        X + Y ! X = 2, Y = 3        # Evaluate with temporary X=2, Y=3

    OPERATORS & STATES:
      Tensor product:    A @ B, psi @ phi
      Conjugation:       A >> B  (conjugate B by A: A† B A)
      State conjugation: A << psi (apply A† ... A to state)
      Measurement:       A | psi (expectation value)

    SUBSCRIPT NOTATION:
      P_{k}              Subscripted variables (k evaluated)
      [P_{k} = expr for k in range(3)]   # Create P_0, P_1, P_2
      
      GenFam('A')        Generator family helper
      A = GenFam('A')    # Create family, then use A[(i,j)] → 'A_{i,j}'
      [A[(i,j)] for i in range(2) for j in range(2)]  # Create A_{0,0}, A_{0,1}, etc.

    KEY FUNCTIONS:
      States:
        char(A, k)                 k-th eigenstate of A
        psi @ phi                  Tensor product of states

      Operators:
        proj(A, k)                 Projector onto k-th eigenspace
        qft(X, Z)                  Quantum Fourier Transform
        ctrl(Z, [I, X])            Controlled operation
        A @ B                      Tensor product of operators

      Channels:
        opChannel([K0, K1])        Operator channel (Heisenberg)
        stChannel([K0, K1])        State channel (Schrödinger)

      Measurement:
        opMeasure([K0, K1], psi)   Single-shot operator measurement
        stMeasure([K0, K1], psi)   Single-shot state measurement
        opBranches([K0, K1], psi)  All branches (operator)
        stBranches([K0, K1], psi)  All branches (state)
      
      Local Context (for functions):
        local(expr, **bindings)    Evaluate expr with variable bindings
        Example in function:
          def equiv(A):
              return local(A, Z=I)   # Equivalent to (A ! Z = I) at REPL
      
      Numerical Evaluation:
        N(expr)                    Numerically evaluate expression
        N(expr, precision=6)       Evaluate to 6 decimal places
        Example:
          mySum = local(sum(w**k for k in range(10)), w=exp(2*I*pi/10))
          N(mySum)                 # Returns 0.0 (very small number cleaned to 0)
      
      Tolerance-Based Comparison:
        isclose(a, b)              Check if a ≈ b (within tolerance)
        isclose(a, b, rtol=1e-5)   Custom relative tolerance
        Example:
          [isclose(f(k), f(k+p)) for k in range(d-p)]  # All True
      
      Classical Randomness:
        random.random()            Random float in [0.0, 1.0)
        random.randint(a, b)       Random integer in [a, b]
        random.choice(seq)         Random element from sequence
        random.shuffle(list)       Shuffle list in place
        Example:
          k = random.randint(0, d-1)
          prob = random.random()
          [random.choice([0,1]) for _ in range(10)]

    CONTEXT VARIABLES:
      _gens, _rels       Ephemeral (reset with each expression)
      $gens, $rels       Persistent (accumulate)
      _state             Last state used

      link($gens, _gens)   Enable accumulation
      unlink($gens, _gens) Disable accumulation

    PYTHON FEATURES:
      Multi-line statements (for, if, while, def, etc.)
      List comprehensions: [expr for x in iterable]
      Tuple unpacking: a, b = (1, 2)

    EXAMPLES:
      # Define algebra and create states
      $alg = <X, Z | herm, unit, anti>
      psi0 = char(Z, 0)
      psi1 = char(Z, 1)

      # Tensor products
      psi00 = psi0 @ psi0
      CNOT = ctrl(Z, [I, X])

      # Measurement
      Z @ Z | psi00                    # Expectation value
      measure = stMeasure([proj(Z, 0), proj(Z, 1)], psi0)
      collapsed, prob = measure()      # Single shot

      # Projectors and controlled ops
      projectors = [P_{k} = proj(Z, k) for k in range(2)]
      CX = ctrl(Z, [I, X])

      # QFT
      W = qft(X, Z)
      W >> Z                           # Should give X

      # Error correction
      TODO!
────────────────────────────────────────────────────────────
    """
        return help_text.strip()
    
def _extract_generators(self, result):
    """Extract generator names from an expression result.
    
    Returns:
        Set of generator names found in the expression
    """
    gens = set()
    
    def walk_expr(obj):
        """Recursively walk expression tree."""
        if isinstance(obj, YawOperator):
            # Check if this is a generator
            if hasattr(self.algebra, 'generator_names'):
                op_str = str(obj._expr)
                for gen_name in self.algebra.generator_names:
                    if gen_name in op_str:
                        gens.add(gen_name)
            walk_expr(obj._expr)
        elif isinstance(obj, TensorProduct):
            for factor in obj.factors:
                walk_expr(factor)
        elif hasattr(obj, 'args'):  # SymPy expression
            for arg in obj.args:
                walk_expr(arg)
    
    walk_expr(result)
    return gens

def main():
    """Entry point for yaw REPL."""
    repl = YawREPL()
    repl.run()

if __name__ == "__main__":
    main()
