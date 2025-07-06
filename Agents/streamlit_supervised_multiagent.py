import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver  # <-- Add memory checkpoint import
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from pprint import pformat
from datetime import datetime
import os
from dotenv import load_dotenv
from opik.integrations.langchain import OpikTracer  # <-- Add Opik import

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.environ.get("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "Research_agent"

# --- Set up OPIK environment variables ---
# os.environ["OPIK_API_KEY"] = os.getenv("OPIK_API_KEY")
# os.environ["OPIK_WORKSPACE"] = "abhishek-seth"
# opik_tracer = OpikTracer()


# --- Supervised Multi AI Agent Architecture (from notebook) ---
def build_supervised_multiagent_workflow():
    class SupervisorState(MessagesState):
        """State for the multi-agent system"""

        next_agent: str = ""
        research_data: str = ""
        analysis: str = ""
        final_report: str = ""
        task_complete: bool = False
        current_task: str = ""

    # Initialize the chat model (Groq)
    llm = init_chat_model("groq:llama-3.1-8b-instant")

    def create_supervisor_chain():
        supervisor_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a supervisor managing a team of agents:\n\n1. Researcher - Gathers information and data\n2. Analyst - Analyzes data and provides insights\n3. Writer - Creates reports and summaries\n\nBased on the current state and conversation, decide which agent should work next.\nIf the task is complete, respond with 'DONE'.\n\nCurrent state:\n- Has research data: {has_research}\n- Has analysis: {has_analysis}\n- Has report: {has_report}\n\nRespond with ONLY the agent name (researcher/analyst/writer) or 'DONE'.\n""",
                ),
                ("human", "{task}"),
            ]
        )
        return supervisor_prompt | llm

    def supervisor_agent(state: SupervisorState):
        messages = state["messages"]
        task = messages[-1].content if messages else "No task"
        has_research = bool(state.get("research_data", ""))
        has_analysis = bool(state.get("analysis", ""))
        has_report = bool(state.get("final_report", ""))
        chain = create_supervisor_chain()
        decision = chain.invoke(
            {
                "task": task,
                "has_research": has_research,
                "has_analysis": has_analysis,
                "has_report": has_report,
            },
            # config={"callbacks": [opik_tracer]},
        )
        decision_text = decision.content.strip().lower()
        if "done" in decision_text or has_report:
            next_agent = "end"
            supervisor_msg = "✅ Supervisor: All tasks complete! Great work team."
        elif "researcher" in decision_text or not has_research:
            next_agent = "researcher"
            supervisor_msg = (
                "📋 Supervisor: Let's start with research. Assigning to Researcher..."
            )
        elif "analyst" in decision_text or (has_research and not has_analysis):
            next_agent = "analyst"
            supervisor_msg = "📋 Supervisor: Research done. Time for analysis. Assigning to Analyst..."
        elif "writer" in decision_text or (has_analysis and not has_report):
            next_agent = "writer"
            supervisor_msg = "📋 Supervisor: Analysis complete. Let's create the report. Assigning to Writer..."
        else:
            next_agent = "end"
            supervisor_msg = "✅ Supervisor: Task seems complete."
        return {
            "messages": [AIMessage(content=supervisor_msg)],
            "next_agent": next_agent,
            "current_task": task,
        }

    def researcher_agent(state: SupervisorState):
        task = state.get("current_task", "research topic")
        research_prompt = f"""As a research specialist, provide comprehensive information about: {task}\n\nInclude:\n1. Key facts and background\n2. Current trends or developments\n3. Important statistics or data points\n4. Notable examples or case studies\n\nBe concise but thorough."""
        research_response = llm.invoke(
            [HumanMessage(content=research_prompt)]
            # , config={"callbacks": [opik_tracer]}
        )
        research_data = research_response.content
        agent_message = f"🔍 Researcher: I've completed the research on '{task}'.\n\nKey findings:\n{research_data[:500]}..."
        return {
            "messages": [AIMessage(content=agent_message)],
            "research_data": research_data,
            "next_agent": "supervisor",
        }

    def analyst_agent(state: SupervisorState):
        research_data = state.get("research_data", "")
        task = state.get("current_task", "")
        analysis_prompt = f"""As a data analyst, analyze this research data and provide insights:\n\nResearch Data:\n{research_data}\n\nProvide:\n1. Key insights and patterns\n2. Strategic implications\n3. Risks and opportunities\n4. Recommendations\n\nFocus on actionable insights related to: {task}"""
        analysis_response = llm.invoke(
            [HumanMessage(content=analysis_prompt)]
            # ,config={"callbacks": [opik_tracer]}
        )
        analysis = analysis_response.content
        agent_message = f"📊 Analyst: I've completed the analysis.\n\nTop insights:\n{analysis[:400]}..."
        return {
            "messages": [AIMessage(content=agent_message)],
            "analysis": analysis,
            "next_agent": "supervisor",
        }

    def writer_agent(state: SupervisorState):
        research_data = state.get("research_data", "")
        analysis = state.get("analysis", "")
        task = state.get("current_task", "")
        writing_prompt = f"""As a professional writer, create an executive report based on:\n\nTask: {task}\n\nResearch Findings:\n{research_data[:1000]}\n\nAnalysis:\n{analysis[:1000]}\n\nCreate a well-structured report with:\n1. Executive Summary\n2. Key Findings\n3. Analysis & Insights\n4. Recommendations\n5. Conclusion\n\nKeep it professional and concise."""
        report_response = llm.invoke(
            [HumanMessage(content=writing_prompt)]
            #  ,config={"callbacks": [opik_tracer]}
        )
        report = report_response.content
        final_report = f"""
📄 FINAL REPORT
{'='*50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Topic: {task}
{'='*50}

{report}

{'='*50}
Report compiled by Multi-Agent AI System powered by Groq
"""
        return {
            "messages": [
                AIMessage(
                    content=f"✍️ Writer: Report complete! See below for the full document."
                )
            ],
            "final_report": final_report,
            "next_agent": "supervisor",
            "task_complete": True,
        }

    def router(state: SupervisorState):
        next_agent = state.get("next_agent", "supervisor")
        if next_agent == "end" or state.get("task_complete", False):
            return END
        if next_agent in ["supervisor", "researcher", "analyst", "writer"]:
            return next_agent
        return "supervisor"

    # Add memory checkpointing
    # memory = MemorySaver()
    workflow = StateGraph(SupervisorState)
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("analyst", analyst_agent)
    workflow.add_node("writer", writer_agent)
    workflow.set_entry_point("supervisor")
    for node in ["supervisor", "researcher", "analyst", "writer"]:
        workflow.add_conditional_edges(
            node,
            router,
            {
                "supervisor": "supervisor",
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                END: END,
            },
        )
    # graph = workflow.compile(checkpointer=memory)
    graph = workflow.compile()
    return graph


# --- Streamlit UI ---
st.title("Research and Analysis Multi-Agent Workflow")

st.write(
    """
Enter a task/question below. When you click Submit, the Supervised Multi AI Agent Architecture will process your request and display the final report.
"""
)

if "task_input" not in st.session_state:
    st.session_state["task_input"] = ""

col1, col2 = st.columns([1, 1])
clear_clicked = col2.button("Clear")
submit_clicked = col1.button("Submit")

if clear_clicked:
    st.session_state["task_input"] = ""
    st.rerun()

task = st.text_area(
    "Enter your task/question:",
    value=st.session_state["task_input"],
    placeholder="Type your task/question here...",
    key="task_input",
)

if submit_clicked and st.session_state["task_input"].strip():
    with st.spinner("Running multi-agent workflow ..."):
        try:
            workflow = build_supervised_multiagent_workflow()
            response = workflow.invoke(
                {"messages": [HumanMessage(content=st.session_state["task_input"])]}
            )
            final_report = response.get("final_report", None)
            if final_report:
                st.subheader("Final Report")
                st.markdown(final_report)
            else:
                st.warning("No final report generated. Full response:")
                st.text(pformat(response))
        except Exception as e:
            st.error(f"Error running workflow: {e}")
