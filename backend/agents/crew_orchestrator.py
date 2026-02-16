"""
CrewAI Orchestration Layer

Provides high-level task orchestration using CrewAI for
document analysis workflows. Works alongside the core
LangGraph pipeline by delegating complex reasoning tasks
to specialized CrewAI agents.

Architecture:
- CrewAI handles high-level task decomposition and delegation
- LangGraph handles the low-level stateful document processing pipeline
- This hybrid approach combines CrewAI's intuitive agent design
  with LangGraph's deterministic graph execution
"""

import logging
from typing import Dict, Any, Optional, List

from crewai import Agent, Task, Crew, Process
from langchain_community.llms import Ollama

logger = logging.getLogger(__name__)


class DocumentAnalysisCrew:
    """
    CrewAI-based crew for high-level document analysis tasks.
    
    This crew handles complex analytical tasks that benefit from
    multi-agent collaboration, such as:
    - Document summarization with different perspectives
    - Cross-document comparison
    - Quality assessment of extracted data
    """
    
    def __init__(self, llm=None):
        """
        Initialize the Document Analysis Crew.
        
        Args:
            llm: Language model to use. Defaults to Groq-backed model.
        """
        self.llm = llm
        self._setup_agents()
    
    def _setup_agents(self):
        """Create specialized CrewAI agents for document analysis."""
        
        self.analyst = Agent(
            role="Document Analyst",
            goal="Analyze document content and extract key insights with high accuracy",
            backstory=(
                "You are a senior document analyst with 20 years of experience "
                "in financial, legal, and technical document review. You excel at "
                "identifying key information, patterns, and anomalies in complex documents."
            ),
            verbose=True,
            allow_delegation=True,
            llm=self.llm
        )
        
        self.fact_checker = Agent(
            role="Fact Checker & Validator",
            goal="Verify the accuracy of extracted information and flag inconsistencies",
            backstory=(
                "You are a meticulous fact-checker who cross-references every claim "
                "against the source document. You have a keen eye for subtle errors, "
                "misinterpretations, and data formatting issues."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        self.summarizer = Agent(
            role="Executive Summarizer",
            goal="Create clear, concise summaries for both technical and non-technical audiences",
            backstory=(
                "You are an expert communicator who transforms complex technical "
                "documents into clear, actionable summaries. You can adapt your "
                "language for C-suite executives (ELI5) or domain experts."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def analyze_document(
        self,
        document_content: str,
        document_type: str = "general",
        analysis_depth: str = "standard"
    ) -> Dict[str, Any]:
        """
        Run a full document analysis using the CrewAI crew.
        
        Args:
            document_content: The extracted text content of the document
            document_type: Type of document (financial, legal, technical, general)
            analysis_depth: Depth of analysis (quick, standard, deep)
            
        Returns:
            Dictionary containing analysis results from all agents
        """
        logger.info(f"Starting CrewAI document analysis (type={document_type}, depth={analysis_depth})")
        
        # Define tasks for the crew
        analysis_task = Task(
            description=(
                f"Analyze the following {document_type} document content thoroughly. "
                f"Identify all key entities, dates, monetary values, and relationships. "
                f"Flag any sections that appear ambiguous or potentially incorrect.\n\n"
                f"Document Content:\n{document_content[:5000]}"
            ),
            expected_output=(
                "A structured analysis containing: "
                "1) Key entities and their roles, "
                "2) Important dates and deadlines, "
                "3) Monetary values and financial data, "
                "4) Potential issues or ambiguities flagged for review"
            ),
            agent=self.analyst
        )
        
        validation_task = Task(
            description=(
                "Review the analysis provided by the Document Analyst. "
                "Cross-reference all extracted facts against the original document. "
                "Verify numerical accuracy and flag any discrepancies."
            ),
            expected_output=(
                "A validation report containing: "
                "1) Confirmed facts (with confidence scores), "
                "2) Flagged discrepancies, "
                "3) Items requiring human review"
            ),
            agent=self.fact_checker
        )
        
        summary_task = Task(
            description=(
                "Based on the validated analysis, create two summaries:\n"
                "1) An ELI5 summary (simple language, key takeaways only)\n"
                "2) An Expert summary (technical details, precise terminology)\n"
                "Both summaries should be concise but comprehensive."
            ),
            expected_output=(
                "Two summaries: "
                "1) ELI5 version (3-5 sentences, simple language), "
                "2) Expert version (detailed technical summary with citations)"
            ),
            agent=self.summarizer
        )
        
        # Create and run the crew
        crew = Crew(
            agents=[self.analyst, self.fact_checker, self.summarizer],
            tasks=[analysis_task, validation_task, summary_task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            result = crew.kickoff()
            
            return {
                "status": "success",
                "analysis": str(result),
                "document_type": document_type,
                "analysis_depth": analysis_depth
            }
        except Exception as e:
            logger.error(f"CrewAI analysis failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "document_type": document_type
            }
    
    def compare_documents(
        self,
        doc1_content: str,
        doc2_content: str
    ) -> Dict[str, Any]:
        """
        Compare two documents using CrewAI agents.
        
        Args:
            doc1_content: Content of first document
            doc2_content: Content of second document
            
        Returns:
            Comparison results
        """
        comparison_task = Task(
            description=(
                f"Compare these two documents and identify:\n"
                f"1) Common themes and shared information\n"
                f"2) Contradictions or discrepancies\n"
                f"3) Unique information in each document\n\n"
                f"Document 1:\n{doc1_content[:3000]}\n\n"
                f"Document 2:\n{doc2_content[:3000]}"
            ),
            expected_output=(
                "A structured comparison with: "
                "1) Similarities, 2) Differences, 3) Contradictions, "
                "4) Recommendation on which document is more reliable"
            ),
            agent=self.analyst
        )
        
        crew = Crew(
            agents=[self.analyst],
            tasks=[comparison_task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            result = crew.kickoff()
            return {
                "status": "success",
                "comparison": str(result)
            }
        except Exception as e:
            logger.error(f"CrewAI comparison failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
