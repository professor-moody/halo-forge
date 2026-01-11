"""
Verifiers Page

Test and manage verifiers for code/math validation.
"""

from nicegui import ui
from typing import Optional
from dataclasses import dataclass

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
        self.selected_verifier: Optional[VerifierInfo] = None
        self.test_code: str = ""
        self.test_result: Optional[dict] = None
    
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
            with ui.column().classes(
                f'w-full gap-4 p-6 rounded-xl bg-[{COLORS["bg_card"]}] '
                f'border border-[#2d343c] animate-in stagger-6'
            ):
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
                    ui.element('span').classes(
                        f'px-2 py-0.5 rounded text-xs bg-[{COLORS["info"]}]/10 text-[{COLORS["info"]}]'
                    ).text(verifier.domain)
                    
                    for lang in verifier.languages[:2]:
                        ui.element('span').classes(
                            f'px-2 py-0.5 rounded text-xs bg-[{COLORS["bg_secondary"]}] '
                            f'text-[{COLORS["text_muted"]}]'
                        ).text(lang)
    
    def _select_verifier(self, verifier: VerifierInfo):
        """Select a verifier for testing."""
        self.selected_verifier = verifier
        self.test_code = verifier.example_solution
        self.test_result = None
        ui.navigate.to('/verifiers')  # Force refresh
    
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
            with ui.element('pre').classes(
                f'w-full p-4 rounded-lg bg-[{COLORS["bg_primary"]}] '
                f'font-mono text-sm text-[{COLORS["text_secondary"]}] overflow-x-auto'
            ):
                ui.text(self.selected_verifier.example_prompt)
        
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
                with ui.element('pre').classes(
                    f'w-full p-3 rounded bg-[{COLORS["bg_primary"]}] '
                    f'font-mono text-xs text-[{COLORS["text_secondary"]}] overflow-x-auto'
                ):
                    ui.text(self.test_result['output'])
    
    async def _run_test(self):
        """Run the verification test."""
        ui.notify('Running verification...', type='info')
        
        # Simulate verification
        import asyncio
        await asyncio.sleep(1)
        
        # Demo result
        self.test_result = {
            'passed': True,
            'message': 'All tests passed',
            'output': 'Test 1: ✓ add(1, 2) == 3\nTest 2: ✓ add(-1, 1) == 0\nTest 3: ✓ add(0, 0) == 0',
        }
        
        ui.notify('Verification complete', type='positive')
