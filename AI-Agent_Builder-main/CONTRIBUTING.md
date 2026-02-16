# Contributing to AI Agent Builder

Thank you for your interest in contributing to the Multi-Modal Document Intelligence Platform! This document provides guidelines for contributing to the project to ensure consistency and quality.

## Development Workflow

### Branching Strategy
We follow a simplified Gitflow workflow:
- **`main`**: Production-ready code. all PRs should target this branch.
- **Feature Branches**: Create a new branch for each feature or fix.
  - Format: `feature/your-feature-name` or `fix/issue-description`
  - Example: `feature/add-voice-agent`

### Getting Started
1. **Fork the repository** (if you are an external contributor).
2. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/AI-Agent_Builder.git
   cd AI-Agent_Builder
   ```
3. **Set up the environment**:
   ```bash
   make install
   make setup
   ```
4. **Run the backend**:
   ```bash
   make run-backend
   ```

## Code Style & Standards

### Python (Backend)
- We use **Black** for code formatting and **isort** for import sorting.
- We use **Pylint** and **MyPy** for linting and type checking.
- All new code must be typed.
- Run formatters before committing:
  ```bash
  make format
  ```

### JavaScript/React (Frontend)
- Use functional components and hooks.
- Ensure components are modular and reusable.
- Follow ESLint configurations provided in the `frontend` directory.

## Adding a New Agent
To add a new agent to the LangGraph workflow:

1. **Create the Agent File**:
   - Create a new file in `backend/agents/`, e.g., `backend/agents/new_agent.py`.
   - Define a class inheriting from a base agent or simply a callable node function.
   - Implement the `process` method.

2. **Define the Node**:
   - Create a generic node wrapper function that handles state updates and error catching (see `vision_agent_node` in `vision_agent.py` for reference).

3. **Update the Workflow**:
   - Modify `backend/agents/workflow.py`:
     - Import your new agent node.
     - Add the node to the graph: `workflow.add_node("new_agent", new_agent_node)`.
     - Define edges and conditional logic for when this agent should be called.

## Pull Request Process
1. Ensure your code builds and runs locally.
2. Run tests: `make test`.
3. Update `README.md` if you changed any significant logic or requirements.
4. Submit a PR with a clear description of the changes.

## Reporting Bugs
Please open an issue on GitHub with:
- A clear title.
- Steps to reproduce.
- Expected vs. actual behavior.
- Logs or screenshots if applicable.
