import ast
import os

class ComplexityAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.loop_count = 0               # Total number of loops
        self.recursive_calls = {}         # Number of recursive calls in each function
        self.current_function = None
        self.loop_nesting = 0             # Current loop nesting level
        self.max_loop_nesting = 0         # Maximum loop nesting level

    def visit_FunctionDef(self, node):
        prev_function = self.current_function
        self.current_function = node.name
        # Initialize the recursive call count for this function
        self.recursive_calls[node.name] = 0
        self.generic_visit(node)
        self.current_function = prev_function

    def visit_For(self, node):
        self.loop_count += 1
        self.loop_nesting += 1
        self.max_loop_nesting = max(self.max_loop_nesting, self.loop_nesting)
        self.generic_visit(node)
        self.loop_nesting -= 1

    def visit_While(self, node):
        self.loop_count += 1
        self.loop_nesting += 1
        self.max_loop_nesting = max(self.max_loop_nesting, self.loop_nesting)
        self.generic_visit(node)
        self.loop_nesting -= 1

    def visit_Call(self, node):
        # Check if it's a recursive call (calling itself)
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name == self.current_function:
                self.recursive_calls[self.current_function] += 1
        self.generic_visit(node)

def estimate_complexity(analyzer):
    """
    Provide a heuristic estimate based on loop nesting levels and recursive calls.
    Note: This estimate is for reference only; actual time and space complexity are influenced by various factors.
    """
    # Estimate time complexity due to loops
    if analyzer.max_loop_nesting == 0:
        loop_complexity = "O(1)"
    else:
        loop_complexity = f"O(n^{analyzer.max_loop_nesting})"

    # Preliminary estimates for recursive calls in each function
    recursion_estimates = {}
    for func, count in analyzer.recursive_calls.items():
        if count == 0:
            recursion_estimates[func] = "No recursion"
        elif count == 1:
            recursion_estimates[func] = "Possibly linear recursion (O(n))"
        else:
            recursion_estimates[func] = "Possibly exponential recursion (O(2^n) or higher)"

    # Combine the effects of loops and recursion to provide an overall time complexity estimate (heuristic)
    time_complexity = loop_complexity
    if any(rec != "No recursion" for rec in recursion_estimates.values()):
        time_complexity += " + Some recursive calls"

    # Space complexity estimate: Mainly considering additional stack space and local variable storage during recursion
    if analyzer.max_loop_nesting == 0 and all(rec == "No recursion" for rec in recursion_estimates.values()):
        space_complexity = "O(1)"
    else:
        # This is a rough estimate; typically, recursive calls may lead to O(n) stack space usage
        space_complexity = "O(n) (due to recursive calls and local variables)"

    return {
        "estimated_time_complexity": time_complexity,
        "estimated_space_complexity": space_complexity,
        "recursion_estimates": recursion_estimates
    }

def analyze_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()
    tree = ast.parse(code, filename=filepath)
    analyzer = ComplexityAnalyzer()
    analyzer.visit(tree)
    estimates = estimate_complexity(analyzer)
    return analyzer, estimates

def generate_report(directory):
    report = []
    for filename in os.listdir(directory):
        if filename.endswith(".py"):
            filepath = os.path.join(directory, filename)
            analyzer, estimates = analyze_file(filepath)
            report.append({
                "filename": filename,
                "loop_count": analyzer.loop_count,
                "max_loop_nesting": analyzer.max_loop_nesting,
                "recursive_calls": analyzer.recursive_calls,
                "estimated_time_complexity": estimates["estimated_time_complexity"],
                "estimated_space_complexity": estimates["estimated_space_complexity"],
                "recursion_estimates": estimates["recursion_estimates"]
            })
    return report

if __name__ == "__main__":
    directory = "methods"  # Please replace this with the directory containing your algorithm files
    report = generate_report(directory)
    for item in report:
        print(f"File: {item['filename']}")
        print(f"  Total number of loops: {item['loop_count']}")
        print(f"  Maximum loop nesting level: {item['max_loop_nesting']}")
        print(f"  Recursive calls in each function: {item['recursive_calls']}")
        print(f"  Estimated time complexity: {item['estimated_time_complexity']}")
        print(f"  Estimated space complexity: {item['estimated_space_complexity']}")
        print(f"  Recursion estimate details: {item['recursion_estimates']}")
        print("-" * 40)
