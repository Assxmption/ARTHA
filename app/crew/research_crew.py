"""
Research Crew
=============
Orchestrates all 4 agents and 4 tasks into a sequential CrewAI pipeline.
Handles job execution, progress tracking, output persistence, and error recovery.
"""

import os
import json
import uuid
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from crewai import Crew, Process
from dotenv import load_dotenv

from app.agents.planner import create_planner_agent
from app.agents.researcher import create_researcher_agent
from app.agents.verifier import create_verifier_agent
from app.agents.writer import create_writer_agent
from app.tasks.research_tasks import (
    create_planning_task,
    create_research_task,
    create_verification_task,
    create_writing_task,
)

load_dotenv()


class ResearchCrew:
    """
    Orchestrates the multi-agent research pipeline.

    Pipeline:
        1. PlannerAgent  → Creates research plan
        2. ResearchAgent → Gathers information from web
        3. VerifierAgent → Validates and quality-checks findings
        4. WriterAgent   → Produces final structured report
    """

    def __init__(self):
        self.reports_dir = Path(os.getenv("REPORTS_DIR", "data/Reports"))
        self.outputs_dir = Path(os.getenv("OUTPUTS_DIR", "data/Outputs"))
        self.verbose = os.getenv("CREW_VERBOSE", "true").lower() == "true"

        # Ensure directories exist
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        query: str,
        job_id: str,
        on_progress: Optional[Callable[[str, str], None]] = None,
    ) -> dict:
        """
        Execute the full research pipeline synchronously.

        Args:
            query: The user's research question
            job_id: Unique identifier for this research job
            on_progress: Optional callback(agent_name, message) for live updates

        Returns:
            dict with keys: report, job_id, query, timestamp, sources_count
        """
        started_at = datetime.utcnow()

        if on_progress:
            on_progress("system", f"🚀 Starting research for: {query}")

        try:
            # ── Build Agents ────────────────────────────────────────────────
            if on_progress:
                on_progress("system", "⚙️ Initialising agents...")

            planner = create_planner_agent()
            researcher = create_researcher_agent()
            verifier = create_verifier_agent()
            writer = create_writer_agent()

            # ── Build Tasks ─────────────────────────────────────────────────
            planning_task = create_planning_task(planner, query)
            research_task = create_research_task(researcher, query, planning_task)
            verification_task = create_verification_task(verifier, query, research_task)
            writing_task = create_writing_task(
                writer, query, planning_task, research_task, verification_task
            )

            # ── Assemble Crew ───────────────────────────────────────────────
            crew_max_rpm = int(os.getenv("CREW_MAX_RPM", "2"))
            crew = Crew(
                agents=[planner, researcher, verifier, writer],
                tasks=[planning_task, research_task, verification_task, writing_task],
                process=Process.sequential,
                verbose=self.verbose,
                max_rpm=crew_max_rpm,
            )

            # ── Execute ─────────────────────────────────────────────────────
            if on_progress:
                on_progress("planner", "📋 Planning research strategy...")

            result = crew.kickoff(inputs={"query": query})

            # Extract the final report text
            if hasattr(result, "raw"):
                report_text = result.raw
            else:
                report_text = str(result)

            # ── Save Outputs ────────────────────────────────────────────────
            completed_at = datetime.utcnow()
            duration_seconds = (completed_at - started_at).total_seconds()

            metadata = {
                "job_id": job_id,
                "query": query,
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "duration_seconds": duration_seconds,
                "status": "completed",
            }

            # Save full JSON output
            output_file = self.outputs_dir / f"{job_id}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {**metadata, "report": report_text},
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            # Save clean markdown report
            report_file = self.reports_dir / f"{job_id}.md"
            report_header = (
                f"# Research Report\n\n"
                f"**Query**: {query}  \n"
                f"**Generated**: {completed_at.strftime('%Y-%m-%d %H:%M UTC')}  \n"
                f"**Duration**: {duration_seconds:.1f}s  \n"
                f"**Job ID**: {job_id}\n\n---\n\n"
            )
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report_header + report_text)

            if on_progress:
                on_progress("system", f"✅ Research complete! Report saved.")

            return {
                "job_id": job_id,
                "query": query,
                "status": "completed",
                "report": report_text,
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "duration_seconds": duration_seconds,
                "report_file": str(report_file),
            }

        except Exception as e:
            error_msg = str(e)
            completed_at = datetime.utcnow()

            # Save error metadata
            error_data = {
                "job_id": job_id,
                "query": query,
                "started_at": started_at.isoformat(),
                "failed_at": completed_at.isoformat(),
                "status": "failed",
                "error": error_msg,
            }
            error_file = self.outputs_dir / f"{job_id}_error.json"
            with open(error_file, "w", encoding="utf-8") as f:
                json.dump(error_data, f, indent=2)

            if on_progress:
                on_progress("system", f"❌ Research failed: {error_msg}")

            return {
                "job_id": job_id,
                "query": query,
                "status": "failed",
                "error": error_msg,
                "started_at": started_at.isoformat(),
                "failed_at": completed_at.isoformat(),
            }

    def list_reports(self) -> list[dict]:
        """List all saved research reports with metadata."""
        reports = []
        for output_file in sorted(
            self.outputs_dir.glob("*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        ):
            if "_error" in output_file.name:
                continue
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                reports.append({
                    "job_id": data.get("job_id", output_file.stem),
                    "query": data.get("query", "Unknown"),
                    "status": data.get("status", "unknown"),
                    "completed_at": data.get("completed_at", ""),
                    "duration_seconds": data.get("duration_seconds", 0),
                })
            except Exception:
                continue
        return reports

    def get_report(self, job_id: str) -> dict | None:
        """Retrieve a specific report by job ID."""
        output_file = self.outputs_dir / f"{job_id}.json"
        if not output_file.exists():
            return None
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def delete_report(self, job_id: str) -> bool:
        """Delete a report and its associated files."""
        deleted = False
        for pattern in [f"{job_id}.json", f"{job_id}_error.json"]:
            f = self.outputs_dir / pattern
            if f.exists():
                f.unlink()
                deleted = True
        report_md = self.reports_dir / f"{job_id}.md"
        if report_md.exists():
            report_md.unlink()
            deleted = True
        return deleted
