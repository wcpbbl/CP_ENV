import json
import jsonlines
from rich import print
from openai import OpenAI
from dataclasses import dataclass, field
from threading import Lock

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.tool_manager import ToolManager
    from utils.agents import PatientAgent

_result_write_lock = Lock()


@dataclass
class RunContext:
    """
    Data container responsible for carrying all runtime data.
    Maintains the state context for the patient-doctor simulation loop.
    """
    # --- Source Data ---
    data_item: dict
    output_path: str
    state_memory_enabled: bool = False

    # --- Internal Agents & Tools ---
    patient_client: OpenAI = field(init=False)
    doctor_client: OpenAI = field(init=False)
    patient_agent: "PatientAgent" = field(init=False)
    tool_manager: "ToolManager" = field(init=False)

    # --- Agent Result Data (Simulation Outcomes) ---
    process_success: bool = field(init=False)
    patient_records: list = field(init=False)
    patient_records_simplified: list = field(init=False)
    diagnosis_reports: list = field(init=False)
    tool_usages: list = field(init=False)
    doctor_patient_diaglogues: list = field(init=False)
    retrieved_report_indices: set[int] = field(default_factory=set, init=False)

    def build_doctor_memory(
            self, encounter_turns: int, max_encounter_nums: int, include_report_text: bool = True
    ) -> str:
        """Render only observed records and successfully retrieved reports for the doctor."""
        completed_tests = []
        retrieved_tests = []
        retrieved_reports = []
        unread_tests = []
        for index, report in enumerate(self.diagnosis_reports):
            test = {"report_id": index + 1, "test_name": report["test_name"]}
            completed_tests.append(test)
            if index in self.retrieved_report_indices:
                retrieved_tests.append(test)
                if include_report_text:
                    retrieved_reports.append({**test, "test_result": report["test_result"]})
            else:
                unread_tests.append(test)

        memory = {
            "current_encounter": encounter_turns,
            "maximum_encounters": max_encounter_nums,
            "encounters_remaining_after_current": max(0, max_encounter_nums - encounter_turns),
            "completed_tests": completed_tests,
            "reports_retrieved_with_get_info": retrieved_tests,
            "reports_not_yet_retrieved_with_get_info": unread_tests,
        }
        if include_report_text:
            memory["retrieved_report_contents"] = retrieved_reports
            memory["previous_mdt_reports"] = [
                usage.get("content", usage.get("output", ""))
                for usage in self.tool_usages
                if usage.get("name") == "organize_mdt"
            ]
        return (
                "\n\n[Cross-encounter working memory]\n"
                "This state is maintained by the hospital environment. It supplements the medical record.\n"
                "Use get_info with the exact test_name to retrieve relevant completed reports that have "
                "not yet been retrieved. The report_id is a memory identifier, not a tool argument.\n"
                "Retrieved reports and MDT reports from earlier encounters are provided when a new "
                "consultation begins. Results already returned in this consultation remain in your history. "
                "The unread list tracks get_info retrieval; an MDT report may also discuss these tests.\n"
                "Review available findings before ordering more tests. If a repeat test is clinically "
                "necessary, explain why the existing result is insufficient in clinic_note.\n"
                "A new diagnostic_test stage and the following specialist consultation require two "
                "additional encounters. Allocate the remaining budget to resolving important uncertainty "
                "and producing the final report. State remaining uncertainty explicitly; do not invent "
                "findings or claim certainty merely because the budget is limited.\n"
                "Use the registered tool interface for tool calls. Your stage report must follow the "
                "clinic_note, suggestion, next_step format from the consultation instructions.\n"
                + json.dumps(memory, ensure_ascii=False, indent=2)
        )

    def output_agent_results(self):
        """
        Aggregates the simulation results and appends them to the output JSONL file.
        """
        results = {
            "original_data": self.data_item,
            "process_success": self.process_success,
            "patient_records": self.patient_records,
            "diagnosis_reports": self.diagnosis_reports,
            "tool_usages": self.tool_usages,
            "doctor_patient_diaglogues": self.doctor_patient_diaglogues
        }
        if self.state_memory_enabled:
            results["state_memory"] = {
                "enabled": True,
                "retrieved_report_indices": sorted(self.retrieved_report_indices),
            }
        # print(results)
        with _result_write_lock:
            with jsonlines.open(self.output_path, "a") as writer:
                writer.write(results)
