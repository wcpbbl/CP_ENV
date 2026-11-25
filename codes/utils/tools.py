"""
Medical Diagnostic Agent Tools Module.

This module implements the `Tools` class, which serves as the functional interface for the medical diagnostic LLM agent. It provides a collection of tools that allow the agent to interact with both internal patient data and external knowledge bases.

Key Functionalities:
1. Internal Data Retrieval (`get_info`): Fetches specific diagnostic test reports from the patient's medical records.
2. External Knowledge Search:
   - `search_wikipedia`: Queries general knowledge.
   - `search_pubmed`: Queries specialized medical literature and research papers.
3. Multi-Agent Simulation (`organize_mdt`): Orchestrates a Multi-Disciplinary Team (MDT) meeting by initializing specialist agents to discuss the case.
"""

import re
import json
import logging
import ast
import requests
from openai import OpenAIError

from utils.mdt import *


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from inference import Args
    from utils.run_context import RunContext


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def sear_wiki(query: str) -> str:
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper

    api_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=10000)
    tool = WikipediaQueryRun(api_wrapper=api_wrapper)
    result = tool.run(query)

    return result


class Tools:
    """
    Container for all concrete tool implementations provided to the LLM.
    """

    def __init__(self, args: "Args", context: "RunContext") -> None:
        # References the patient's medical record; list state is shared across the context.
        self.args = args
        self.context = context

    def _tool_error(self, result_name: str, message: str):
        result = "Tool error: " + message
        logger.warning("%s: %s", result_name, message)
        if self.args.doctor_responses_api:
            return (result_name, result)
        return result

    def get_info(self, test_names: list[str]) -> list[int]:
        """
        Fetches the results of a completed diagnostic test from the patient's medical record. This function should only be invoked for tests specified within the Completed Tests field of the Medical Record.

        Args:
            test_names (list[str]): A list of names for the diagnostic tests to retrieve.

        Returns:
            list[str]: A list of strings, each containing the name and detailed report of a corresponding test.
        """
        logger.info(f"\n--- function get_info ---:  asking for {test_names}\n")

        completed_test_names = [obj["test_name"] for obj in self.context.diagnosis_reports]
        if not completed_test_names:
            result = "\n".join(
                f"The patient has not undergone the test named {test_name}. Please order a test request."
                for test_name in test_names
            ) or "No completed diagnostic reports are available."
            if self.args.doctor_responses_api:
                return ("test_reports", result)
            return result

        prompt = \
"""
You are given a query list of requested report names {INSERT_Q_HERE} and a corpus list of available report names {INSERT_C_HERE}.

For each report name in the query list, find its corresponding index in the corpus list. If a report name is not found, return -1.

Output the results in a JSON array format:
```json
[...]
```
"""
        try:
            response = self.context.patient_client.chat.completions.create(
                model=self.args.patient_model_name,
                messages=[
                    {"role": "user", "content": prompt.replace("INSERT_Q_HERE", str(test_names)).replace("INSERT_C_HERE", str(completed_test_names))}
                ]
            )
        except OpenAIError as exc:
            logger.warning("get_info report matching request failed", exc_info=True)
            return self._tool_error(
                "test_reports",
                f"get_info could not match reports ({type(exc).__name__}). No reports were retrieved. "
                "Retry with the exact names from Completed Tests."
            )

        text = (response.choices[0].message.content or "").strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        payload = fenced.group(1).strip() if fenced else text
        try:
            indices = json.loads(payload)
        except json.JSONDecodeError:
            return self._tool_error(
                "test_reports",
                "get_info received an invalid report-matching response. No reports were retrieved. "
                "Retry with the exact names from Completed Tests."
            )

        if not isinstance(indices, list) or len(indices) != len(test_names) or not all(
            type(index) is int and -1 <= index < len(self.context.diagnosis_reports)
            for index in indices
        ):
            return self._tool_error(
                "test_reports",
                "get_info received invalid report indices. No reports were retrieved. "
                "Retry with the exact names from Completed Tests."
            )

        result = ""
        for i, j in enumerate(indices):
            if j != -1:
                try:
                    report = self.context.diagnosis_reports[j]
                    result += str(report) + "\n"
                    if getattr(self.context, "state_memory_enabled", False) and report.get("test_result"):
                        self.context.retrieved_report_indices.add(j)
                except:
                    continue
            elif j == -1:
                result += f"The patient has not undergone the test named {test_names[i]}. Please order a test request.\n"

        if not self.args.doctor_responses_api:
            return result
        else:
            return ("test_reports", result)

    def search_wikipedia(self, query: str) -> str:
        """
        Searches Wikipedia for a given query using the LangChain Wikipedia tool.

        Args:
            query (str): The term or question to search for on Wikipedia.

        Returns:
            str: The summarized search results from Wikipedia.
        """

        try:
            result = sear_wiki(query)
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Wikipedia retrieval failed for query %r", query, exc_info=True)
            return self._tool_error(
                "wiki_search_result",
                f"Wikipedia retrieval failed ({type(exc).__name__}). No external evidence was obtained. "
                "Continue using available case evidence or try another registered search tool."
            )

        prompt = \
"""
Here is the user's query: {INSERT_QUERY_HERE}, and here is the corresponding Wikipedia article found for this query: {INSERT_RESULT_HERE}.

Please provide a summary. If nothing is found in Wikipedia, respond with 'Nothing found.' Do not output anything else.
"""
        response = self.context.patient_client.chat.completions.create(
            model=self.args.patient_model_name,
            messages=[
                {"role": "user", "content": prompt.replace("INSERT_QUERY_HERE", query).replace("INSERT_RESULT_HERE", str(result))}
            ]
        )
        result = response.choices[0].message.content

        if not self.args.doctor_responses_api:
            return result
        else:
            return ("wiki_search_result", result)

    def search_pubmed(self, query: str) -> str:
        """
        Searches for relevant medical and scientific literature in the PubMed database
        using LangChain's PubMed tool.

        Args:
            query (str): The keyword or question to search for on PubMed.

        Returns:
            str: The summarized search results of papers.
        """
        logger.info(f"\n--- function search_pubmed ---:  asking for {query}\n")

        from langchain_community.tools.pubmed.tool import PubmedQueryRun
        from langchain_community.utilities import PubMedAPIWrapper

        api_wrapper = PubMedAPIWrapper(top_k_results=3, doc_content_chars_max=10000)
        pubmed_tool = PubmedQueryRun(api_wrapper=api_wrapper)
        result = pubmed_tool.run(query)
        logger.info(f"\n--- function search_pubmed ---:  searching result {result}\n")

        # Improved prompt for medical summarization
        prompt = \
"""
Here is the user's query: {INSERT_QUERY_HERE}, and here is the corresponding Pubmed article found for this query: {INSERT_RESULT_HERE}.

Please provide a summary. If nothing is found in PubMed, respond with 'Nothing found.' Do not output anything else.
"""
        response = self.context.patient_client.chat.completions.create(
            model=self.args.patient_model_name,
            messages=[
                {"role": "user", "content": prompt.replace("INSERT_QUERY_HERE", query).replace("INSERT_RESULT_HERE", str(result))}
            ]
        )
        result = response.choices[0].message.content
        logger.info(f"\n--- function search_pubmed ---:  summary result {result}\n")

        if not self.args.doctor_responses_api:
            return result
        else:
            return ("pubmed_search_result", result)

    def organize_mdt(self, recruited_agents: list[tuple[str, str]]) -> str:
        """
        Organize and simulate a Multidisciplinary Team (MDT) discussion for the patient's case.

        Convenes up to four specified medical experts to conduct an MDT meeting, analyze this complex case, and generate a collaborative conclusion.

        Args:
            recruited_agents (list[tuple[str, str]]): A list of (role, description) tuples for each medical expert (max 4).  Example: `[('Cardiologist', 'Expert in heart diseases')]`

        Returns:
            str: A comprehensive summary of the MDT discussion, analysis, and conclusions.
        """

        # print(recruited_agents)
        # print(type(recruited_agents))
        if type(recruited_agents) == str:
            recruited_agents = ast.literal_eval(recruited_agents)

        logger.info(f"\n--- function organize_mdt ---:  organize the group {recruited_agents}\n")

        initialize_settings(
            base_url=str(self.args.patient_base_url),
            api_key=self.args.patient_api_key,
            model_name=self.args.patient_model_name
        )

        patient_case = self.context.patient_records.copy()
        tests = []
        for obj in self.context.diagnosis_reports:
            tests.append({obj["test_name"]: obj["test_result"]})
            # tests.append({obj["test_name"]: obj["test_report"]})
        patient_case.append(
            {
                "Record Department": "Diagnosis Tests",
                "Reports": str(tests)
            }
        )
        result = mdt(str(patient_case), recruited_agents)

        logger.info(f"\n--- function organize_mdt ---:  MDT Report:\n{result}\n")
        if not self.args.doctor_responses_api:
            return result
        else:
            return ("mdt_final_report", result)


if __name__ == "__main__":
    tools_instance = Tools()

    # print(tools_instance.search_wikipedia("Artificial Intelligence"))

    # print(tools_instance.search_pubmed("systemic vasculitis with scrotal eschar and hemorrhagic manifestations"))

    tools_instance.organize_mdt([('Hematologist', 'Expert in hematologic malignancies and immune-mediated hemolytic anemia'), ('Infectious Disease Specialist', 'Specialist in febrile neutropenia and bloodstream infections'), ('Dermatologist', 'Expert in cutaneous complications of systemic diseases and drug reactions'), ('Immunologist', 'Specialist in complement-mediated disorders and autoimmune conditions')])
