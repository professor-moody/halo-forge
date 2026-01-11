"""
Verifiers Page

Test and manage verifiers for code/math validation.
"""

from nicegui import ui, app
from typing import Optional
from dataclasses import dataclass
import subprocess
import tempfile
import os

from ui.theme import COLORS


@dataclass
class VerifierInfo:
    """Information about a verifier."""
    name: str
    description: str
    domain: str
    icon: str
    languages: list[str]
    example_prompt: str
    example_solution: str


class Verifiers:
    """Verifier testing page component."""
    
    VERIFIERS = [
        VerifierInfo(
            name="HumanEval",
            description="Python function completion benchmark with test execution",
            domain="Code",
            icon="code",
            languages=["Python"],
            example_prompt='def add(a: int, b: int) -> int:\n    """Return the sum of a and b."""',
            example_solution='    return a + b',
        ),
        VerifierInfo(
            name="MBPP",
            description="Mostly Basic Python Problems for simple coding tasks",
            domain="Code",
            icon="code",
            languages=["Python"],
            example_prompt='Write a function to find the factorial of a number.',
            example_solution='def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)',
        ),
        VerifierInfo(
            name="LiveCodeBench",
            description="Multi-language code generation with execution",
            domain="Code",
            icon="terminal",
            languages=["Python", "JavaScript", "Go", "Rust"],
            example_prompt='Implement a function that reverses a string.',
            example_solution='def reverse_string(s: str) -> str:\n    return s[::-1]',
        ),
        VerifierInfo(
            name="Math",
            description="Mathematical expression verification with numerical tolerance",
            domain="Math",
            icon="calculate",
            languages=["Natural Language"],
            example_prompt='What is 15% of 80?',
            example_solution='12',
        ),
        VerifierInfo(
            name="GSM8K",
            description="Grade-school math word problems with chain-of-thought",
            domain="Math",
            icon="school",
            languages=["Natural Language"],
            example_prompt='James has 3 boxes of apples. Each box contains 12 apples. How many apples does James have in total?',
            example_solution='James has 3 boxes × 12 apples = 36 apples in total.\n#### 36',
        ),
    ]
    
    def __init__(self):
        # Restore selected verifier from storage
        stored_name = app.storage.user.get('selected_verifier_name')
        self.selected_verifier: Optional[VerifierInfo] = next(
            (v for v in self.VERIFIERS if v.name == stored_name), None
        ) if stored_name else None
        
        self.test_code: str = self.selected_verifier.example_solution if self.selected_verifier else ""
        self.test_result: Optional[dict] = None
        self.test_panel_container = None  # Reference for dynamic refresh
    
    def render(self):
        """Render the verifiers page."""
        with ui.column().classes('page-content w-full gap-6 p-6'):
            # Header
            ui.label('Verifiers').classes(
                f'text-2xl font-bold text-[{COLORS["text_primary"]}] animate-in'
            )
            
            # Verifier cards
            with ui.row().classes('w-full gap-4 flex-wrap animate-in stagger-1'):
                for i, verifier in enumerate(self.VERIFIERS):
                    self._render_verifier_card(verifier, i)
            
            # Test panel
            self.test_panel_container = ui.column().classes(
                f'w-full gap-4 p-6 rounded-xl bg-[{COLORS["bg_card"]}] '
                f'border border-[#2d343c] animate-in stagger-6'
            )
            with self.test_panel_container:
                self._render_test_panel()
    
    def _render_verifier_card(self, verifier: VerifierInfo, index: int):
        """Render a verifier card."""
        is_selected = self.selected_verifier == verifier
        
        with ui.card().classes(
            f'min-w-[200px] flex-1 p-4 rounded-xl cursor-pointer card-hover '
            f'animate-in stagger-{index + 1} '
            + (f'border-2 border-[{COLORS["primary"]}] bg-[{COLORS["primary"]}]/5' if is_selected
               else f'border border-[#2d343c] bg-[{COLORS["bg_card"]}]')
        ).on('click', lambda v=verifier: self._select_verifier(v)):
            with ui.column().classes('gap-3'):
                # Header
                with ui.row().classes('items-center gap-2'):
                    with ui.element('div').classes(
                        f'w-8 h-8 rounded-lg flex items-center justify-center '
                        f'bg-[{COLORS["accent"]}]/10'
                    ):
                        ui.icon(verifier.icon, size='18px').classes(
                            f'text-[{COLORS["accent"]}]'
                        )
                    ui.label(verifier.name).classes(
                        f'text-sm font-semibold text-[{COLORS["text_primary"]}]'
                    )
                
                # Description
                ui.label(verifier.description).classes(
                    f'text-xs text-[{COLORS["text_secondary"]}] line-clamp-2'
                )
                
                # Tags
                with ui.row().classes('gap-1 flex-wrap'):
                    ui.label(verifier.domain).classes(
                        f'px-2 py-0.5 rounded text-xs bg-[{COLORS["info"]}]/10 text-[{COLORS["info"]}]'
                    )
                    
                    for lang in verifier.languages[:2]:
                        ui.label(lang).classes(
                            f'px-2 py-0.5 rounded text-xs bg-[{COLORS["bg_secondary"]}] '
                            f'text-[{COLORS["text_muted"]}]'
                        )
    
    def _select_verifier(self, verifier: VerifierInfo):
        """Select a verifier for testing."""
        self.selected_verifier = verifier
        self.test_code = verifier.example_solution
        self.test_result = None
        app.storage.user['selected_verifier_name'] = verifier.name
        # Refresh only the test panel instead of full page
        if self.test_panel_container:
            self.test_panel_container.clear()
            with self.test_panel_container:
                self._render_test_panel()
        else:
            ui.navigate.to('/verifiers')  # Fallback
    
    def _render_test_panel(self):
        """Render the test panel."""
        with ui.row().classes('w-full items-center justify-between'):
            ui.label('Test Verifier').classes(
                f'text-base font-semibold text-[{COLORS["text_primary"]}]'
            )
            
            if self.selected_verifier:
                ui.label(f'Using: {self.selected_verifier.name}').classes(
                    f'text-sm text-[{COLORS["accent"]}]'
                )
        
        if not self.selected_verifier:
            with ui.column().classes('w-full items-center py-8 gap-2'):
                ui.icon('touch_app', size='48px').classes(
                    f'text-[{COLORS["text_muted"]}]'
                )
                ui.label('Select a verifier above to test').classes(
                    f'text-sm text-[{COLORS["text_muted"]}]'
                )
            return
        
        # Prompt display
        with ui.column().classes('w-full gap-2'):
            ui.label('Prompt').classes(
                f'text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
            )
            ui.html(f'''<pre class="w-full p-4 rounded-lg font-mono text-sm overflow-x-auto"
                         style="background: {COLORS["bg_primary"]}; color: {COLORS["text_secondary"]}; white-space: pre-wrap;">{self.selected_verifier.example_prompt}</pre>''')
        
        # Code input
        with ui.column().classes('w-full gap-2'):
            ui.label('Your Solution').classes(
                f'text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
            )
            ui.textarea(
                value=self.test_code,
                on_change=lambda e: setattr(self, 'test_code', e.value)
            ).classes('w-full font-mono').props(
                'outlined autogrow dark rows=6'
            )
        
        # Run button
        with ui.row().classes('w-full justify-end'):
            ui.button('Run Verification', icon='play_arrow', on_click=self._run_test).props(
                'unelevated'
            ).classes(f'bg-[{COLORS["primary"]}] text-white')
        
        # Results
        if self.test_result:
            self._render_result()
    
    def _render_result(self):
        """Render test results."""
        passed = self.test_result.get('passed', False)
        
        with ui.column().classes(
            f'w-full gap-3 p-4 rounded-lg '
            + (f'bg-[{COLORS["success"]}]/10 border border-[{COLORS["success"]}]/30' if passed
               else f'bg-[{COLORS["error"]}]/10 border border-[{COLORS["error"]}]/30')
        ):
            with ui.row().classes('items-center gap-2'):
                icon = 'check_circle' if passed else 'cancel'
                color = COLORS['success'] if passed else COLORS['error']
                
                ui.icon(icon, size='24px').classes(f'text-[{color}]')
                ui.label('Passed' if passed else 'Failed').classes(
                    f'text-base font-semibold text-[{color}]'
                )
            
            # Details
            if 'message' in self.test_result:
                ui.label(self.test_result['message']).classes(
                    f'text-sm text-[{COLORS["text_secondary"]}]'
                )
            
            if 'output' in self.test_result:
                ui.html(f'''<pre class="w-full p-3 rounded font-mono text-xs overflow-x-auto"
                             style="background: {COLORS["bg_primary"]}; color: {COLORS["text_secondary"]}; white-space: pre-wrap;">{self.test_result['output']}</pre>''')
    
    async def _run_test(self):
        """Run the verification test."""
        if not self.selected_verifier:
            ui.notify('Select a verifier first', type='warning')
            return
        
        ui.notify('Running verification...', type='info')
        
        try:
            # For Code verifiers, execute Python code
            if self.selected_verifier.domain == "Code":
                result = await self._run_python_verification()
            else:
                # For Math verifiers, do simple answer matching
                result = await self._run_math_verification()
            
            self.test_result = result
            
            if result['passed']:
                ui.notify('Verification passed!', type='positive')
            else:
                ui.notify('Verification failed', type='negative')
            
            # Refresh the test panel to show result
            if self.test_panel_container:
                self.test_panel_container.clear()
                with self.test_panel_container:
                    self._render_test_panel()
                    
        except Exception as e:
            self.test_result = {
                'passed': False,
                'message': f'Error: {str(e)}',
                'output': '',
            }
            ui.notify(f'Verification error: {e}', type='negative')
    
    async def _run_python_verification(self) -> dict:
        """Run Python code verification."""
        # Combine prompt (function signature) with solution
        full_code = self.selected_verifier.example_prompt + "\n" + self.test_code
        
        # Add simple test cases based on the verifier
        test_code = full_code + "\n\n"
        
        if self.selected_verifier.name == "HumanEval":
            # Test add function
            test_code += """
# Test cases
try:
    assert add(1, 2) == 3, "add(1, 2) should be 3"
    assert add(-1, 1) == 0, "add(-1, 1) should be 0"
    assert add(0, 0) == 0, "add(0, 0) should be 0"
    print("Test 1: ✓ add(1, 2) == 3")
    print("Test 2: ✓ add(-1, 1) == 0")
    print("Test 3: ✓ add(0, 0) == 0")
    print("\\nAll tests passed!")
except AssertionError as e:
    print(f"✗ {e}")
    exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)
"""
        elif self.selected_verifier.name == "MBPP":
            # Test factorial function
            test_code += """
# Test cases
try:
    assert factorial(0) == 1, "factorial(0) should be 1"
    assert factorial(1) == 1, "factorial(1) should be 1"
    assert factorial(5) == 120, "factorial(5) should be 120"
    print("Test 1: ✓ factorial(0) == 1")
    print("Test 2: ✓ factorial(1) == 1")
    print("Test 3: ✓ factorial(5) == 120")
    print("\\nAll tests passed!")
except AssertionError as e:
    print(f"✗ {e}")
    exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)
"""
        elif self.selected_verifier.name == "LiveCodeBench":
            # Test reverse_string function
            test_code += """
# Test cases
try:
    assert reverse_string("hello") == "olleh", "reverse_string('hello') should be 'olleh'"
    assert reverse_string("") == "", "reverse_string('') should be ''"
    assert reverse_string("a") == "a", "reverse_string('a') should be 'a'"
    print("Test 1: ✓ reverse_string('hello') == 'olleh'")
    print("Test 2: ✓ reverse_string('') == ''")
    print("Test 3: ✓ reverse_string('a') == 'a'")
    print("\\nAll tests passed!")
except AssertionError as e:
    print(f"✗ {e}")
    exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)
"""
        else:
            # Generic: just try to run the code
            test_code += "\nprint('Code executed successfully')\n"
        
        # Run in subprocess
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                ['python', temp_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            output = result.stdout + result.stderr
            passed = result.returncode == 0
            
            return {
                'passed': passed,
                'message': 'All tests passed' if passed else 'Some tests failed',
                'output': output.strip() or '(no output)',
            }
        finally:
            os.unlink(temp_path)
    
    async def _run_math_verification(self) -> dict:
        """Run math answer verification."""
        # Extract numeric answer from solution
        solution = self.test_code.strip()
        
        # For GSM8K format, extract answer after ####
        if '####' in solution:
            answer = solution.split('####')[-1].strip()
        else:
            # Try to find a number
            import re
            numbers = re.findall(r'-?\d+\.?\d*', solution)
            answer = numbers[-1] if numbers else solution
        
        # Check against expected (hardcoded for demo)
        expected = {
            "Math": "12",
            "GSM8K": "36",
        }.get(self.selected_verifier.name, "")
        
        passed = answer.strip() == expected.strip()
        
        return {
            'passed': passed,
            'message': f'Answer: {answer}' + (' ✓' if passed else f' (expected {expected})'),
            'output': f'Your answer: {answer}\nExpected: {expected}',
        }
