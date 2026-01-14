"""
The Illustrator Agent - Diagram Generation.

The Illustrator generates visual diagrams for abstract mathematical concepts:
- Parses <<DIAGRAM_REQUEST: description>> tags from content
- Generates Python code (matplotlib/networkx) or Mermaid code
- Executes code in sandbox environment
- Returns image paths for embedding

Backend: Any LLM (code generation doesn't need specific model)
"""

import re
import uuid
from pathlib import Path
from typing import Optional

from shared.logging import get_logger
from llm.src.client import LLMClient

from .diagram_sandbox import DiagramSandbox, generate_fano_plane_code
from .repository import DocumentRepository

log = get_logger("documenter", "illustrator")


# Pattern to match diagram requests
DIAGRAM_REQUEST_PATTERN = re.compile(r'<<DIAGRAM_REQUEST:\s*(.+?)>>', re.IGNORECASE)


class Illustrator:
    """
    The Illustrator Agent - generates diagrams for mathematical concepts.

    Converts textual descriptions into visual diagrams using:
    - Python matplotlib for geometric figures
    - Python networkx for graphs and networks
    - Mermaid for flow diagrams and state machines
    """

    def __init__(
        self,
        repository: DocumentRepository,
        llm_client: LLMClient,
        sandbox_timeout: int = 30,
    ):
        """
        Initialize the Illustrator.

        Args:
            repository: Document repository for assets directory
            llm_client: LLM client for code generation
            sandbox_timeout: Timeout for diagram execution
        """
        self.repository = repository
        self.llm_client = llm_client
        self.sandbox = DiagramSandbox(
            assets_dir=repository.assets_dir,
            timeout_seconds=sandbox_timeout,
        )

    async def process_diagram_requests(
        self,
        content: str,
    ) -> tuple[str, list[str]]:
        """
        Process all diagram requests in content.

        Finds <<DIAGRAM_REQUEST: description>> tags, generates diagrams,
        and replaces tags with image links.

        Args:
            content: Content with diagram request tags

        Returns:
            (processed_content, list_of_generated_paths)
        """
        requests = DIAGRAM_REQUEST_PATTERN.findall(content)

        if not requests:
            return content, []

        log.info("illustrator.processing", request_count=len(requests))

        generated_paths = []
        processed_content = content

        for description in requests:
            diagram_path = await self.generate_diagram(description)

            if diagram_path:
                # Replace tag with image link
                tag = f"<<DIAGRAM_REQUEST: {description}>>"
                relative_path = f"assets/{diagram_path.name}"
                image_link = f"![{description}]({relative_path})"
                processed_content = processed_content.replace(tag, image_link, 1)
                generated_paths.append(str(diagram_path))

                log.info(
                    "illustrator.diagram_generated",
                    description=description[:50],
                    path=str(diagram_path),
                )
            else:
                # Remove failed tag with placeholder
                tag = f"<<DIAGRAM_REQUEST: {description}>>"
                placeholder = f"*[Diagram: {description}]*"
                processed_content = processed_content.replace(tag, placeholder, 1)

                log.warning(
                    "illustrator.diagram_failed",
                    description=description[:50],
                )

        return processed_content, generated_paths

    async def generate_diagram(
        self,
        description: str,
    ) -> Optional[Path]:
        """
        Generate a single diagram from description.

        Args:
            description: What the diagram should show

        Returns:
            Path to generated diagram or None on failure
        """
        # First, determine diagram type and generate code
        code = await self._generate_diagram_code(description)

        if not code:
            return None

        # Execute in sandbox
        output_path, error = await self.sandbox.execute(code)

        if error:
            log.warning(
                "illustrator.sandbox_error",
                error=error,
                description=description[:50],
            )
            # Try once more with simpler code
            simple_code = await self._generate_simple_fallback(description)
            if simple_code:
                output_path, error = await self.sandbox.execute(simple_code)

        return output_path

    async def _generate_diagram_code(self, description: str) -> Optional[str]:
        """Generate Python code for the diagram."""
        # Check for common patterns
        description_lower = description.lower()

        # Fano plane is common - use template
        if "fano" in description_lower and "plane" in description_lower:
            return generate_fano_plane_code()

        # Generate custom code via LLM
        prompt = f"""Generate Python matplotlib code to create a mathematical diagram.

DESCRIPTION: {description}

REQUIREMENTS:
1. Use matplotlib.pyplot as plt
2. Use numpy as np if needed
3. Use networkx as nx for graph structures if needed
4. The code will be executed in a sandbox - only these libraries are available
5. Do NOT include plt.show() - the figure will be saved automatically
6. Set appropriate figure size with figsize
7. Use ax.axis('off') for clean diagrams
8. Add a descriptive title

EXAMPLE TEMPLATE (for reference):
```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Your drawing code here
# ax.plot(...), ax.scatter(...), etc.

ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Title Here')
```

Reply with ONLY the Python code, no explanation or markdown formatting.
"""

        response = await self.llm_client.send(
            backend="claude",
            prompt=prompt,
            timeout_seconds=60,
        )

        if not response.success:
            log.warning("illustrator.code_generation_failed", error=response.error)
            return None

        # Clean up response - remove markdown code blocks if present
        code = response.text
        code = re.sub(r'^```python\n?', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```\n?', '', code, flags=re.MULTILINE)
        code = code.strip()

        return code

    async def _generate_simple_fallback(self, description: str) -> Optional[str]:
        """Generate a simple fallback diagram if main generation fails."""
        # Simple placeholder diagram
        return f'''
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.text(0.5, 0.5, "{description[:50]}...",
        ha='center', va='center', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Diagram Placeholder')
'''

    async def generate_mermaid(self, description: str) -> Optional[str]:
        """
        Generate Mermaid diagram code for flow/state diagrams.

        Mermaid is better for:
        - Flowcharts
        - State diagrams
        - Sequence diagrams

        Args:
            description: What the diagram should show

        Returns:
            Mermaid code or None on failure
        """
        prompt = f"""Generate Mermaid diagram code for the following:

DESCRIPTION: {description}

Requirements:
1. Use appropriate Mermaid syntax (flowchart, stateDiagram, etc.)
2. Keep it clean and readable
3. Use descriptive node labels

Reply with ONLY the Mermaid code, no explanation.
Example format:
```mermaid
flowchart TD
    A[Start] --> B[Process]
    B --> C[End]
```
"""

        response = await self.llm_client.send(
            backend="claude",
            prompt=prompt,
            timeout_seconds=30,
        )

        if not response.success:
            return None

        # Extract mermaid code
        code = response.text
        match = re.search(r'```mermaid\n(.*?)```', code, re.DOTALL)
        if match:
            return match.group(1).strip()

        return code.strip()

    def extract_requests(self, content: str) -> list[str]:
        """
        Extract diagram request descriptions from content.

        Args:
            content: Content to scan

        Returns:
            List of diagram descriptions
        """
        return DIAGRAM_REQUEST_PATTERN.findall(content)
