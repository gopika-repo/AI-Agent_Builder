# ADR 001: Hybrid LangGraph + CrewAI Architecture for Agent Orchestration

## Status
Accepted

## Date
2026-02-14

## Context
The "AI Agents Builder System" challenge suggested using **CrewAI** as a potential framework for orchestrating the multi-agent system. However, the requirements for this document intelligence platform include:
1.  **Cyclic Workflows**: The ability to loop back (e.g., Conflict Resolution -> Vision -> Fusion) which is essential for self-correction.
2.  **Fine-Grained State Management**: Passing a complex `DocumentState` object (containing images, text, bounding boxes, and confidence scores) between agents with precise control.
3.  **Deterministic Control Flow**: While we want "agentic" properties, the high-level process (Vision -> OCR -> Layout -> ...) is largely a directed graph with conditional branching, rather than a purely open-ended conversation.
4.  **Production Reliability**: Need for a framework that supports persistence (checkpointing), streaming, and explicit error handling paths.

## Decision
We decided to use a **hybrid architecture** combining:
- **LangGraph** for the core document processing pipeline (deterministic, cyclic graph)
- **CrewAI** for high-level analytical tasks (document comparison, deep analysis with multi-agent collaboration)

This approach leverages the strengths of both frameworks:
- LangGraph excels at fine-grained, stateful control over the processing graph
- CrewAI excels at intuitive multi-agent task decomposition and delegation

## Consequences

### Positive
*   **Best of Both Worlds**: LangGraph provides deterministic control for the core pipeline, while CrewAI provides flexible multi-agent collaboration for analytical tasks.
*   **Superior Control**: LangGraph allows us to define the exact graph topology, including cycles and conditional edges, which maps perfectly to our pipeline architecture.
*   **Intuitive Analysis**: CrewAI's Agent/Task/Crew abstractions make it easy to compose high-level analytical workflows (Analyst → Fact Checker → Summarizer).
*   **State Persistence**: LangGraph checkpointing capabilities allow us to pause and resume workflows.
*   **Explicit Error Handling**: We can define specific "error handler" nodes and edges in LangGraph, ensuring the pipeline handles failures gracefully.
*   **Async Support**: Both frameworks integrate well with our FastAPI backend for high-throughput processing.

### Negative
*   **Higher Boilerplate**: LangGraph requires more explicit definition of nodes and edges compared to the higher-level abstractions of CrewAI.
*   **Learning Curve**: It requires a deeper understanding of graph theory concepts and LangChain internals.

## Mitigation
We have managed the boilerplate by modularizing each agent into its own file and using valid Pydantic models (`DocumentState`) to enforce type safety across the graph, turning the "boilerplate" into "type-safe infrastructure".
