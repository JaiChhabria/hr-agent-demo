import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
from langchain_community.callbacks import StreamlitCallbackHandler

# IMPORT load_data TO FIX THE DROPDOWN ISSUE
from backend_logic import get_hr_agent, calculate_hike_impact, simulate_employee_updates_logic, reset_demo_data, load_data

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & INITIALIZATION
# ---------------------------------------------------------
st.set_page_config(page_title="Agentic HR Demo", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .stChatMessage {background-color: #f0f2f6; border-radius: 10px; padding: 10px;}
    .stButton>button {width: 100%; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

# CRITICAL FIX: Ensure Data Exists Before UI Loads
# This prevents the "Missing Dropdown" bug on first load or after reset.
if not os.path.exists("employees.csv"):
    with st.spinner("Initializing HR Database..."):
        load_data()

# ---------------------------------------------------------
# 2. SIDEBAR - "COMMAND CENTER"
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=100)
    st.title("HR Command Center")
    
    # LOAD DROPDOWN DATA (Now guaranteed to exist)
    try:
        df_ui = pd.read_csv("employees.csv")
        emp_names = sorted(df_ui['Name'].astype(str).tolist())
        depts = sorted(df_ui['Department'].unique().tolist())
    except Exception as e:
        st.error(f"Data Error: {e}")
        emp_names = []
        depts = []

    # WORKFLOW
    st.subheader("🧠 Agent Workflow")
    workflow_dot = """
    digraph G {
        rankdir=TB;
        node [fontname="Arial"];
        User [shape=oval, style=filled, fillcolor=lightblue];
        Router [shape=box, style=filled, fillcolor=gold];
        Analytics [shape=ellipse, fillcolor=lightgrey, style=filled];
        Policy [shape=ellipse]; Drafter [shape=ellipse];
        User -> Router;
        Router -> Analytics [label="Trends?"];
        Router -> Policy; Router -> Drafter;
    }
    """
    st.graphviz_chart(workflow_dot)
    st.markdown("---")
    
    # 1. STRATEGIC INSIGHTS (TERMS & TRENDS)
    st.subheader("📈 Terms & Trends")
    
    # Global View
    if st.button("📊 Global Headcount"):
        st.session_state.prompt_trigger = "Calculate the total Headcount for EACH Department. **Mandatory: Output the result as a Markdown Table with columns: Department | Headcount.**"

    st.caption("Attrition Drill-Down:")
    # Dropdowns are populated now!
    sel_dept = st.selectbox("Department:", depts, key="dept_select")
    analysis_type = st.selectbox("View Trend By:", ["Year over Year (2024-2025)", "Monthly Breakdown (All Time)"])
    
    if st.button(f"📉 Analyze Terms ({sel_dept})"):
        if "Year" in analysis_type:
            prompt = f"Analyze 'attrition.csv' for Department '{sel_dept}'. Group the exits by YEAR of Exit_Date. Show the count for each year. **Output as a Markdown Table.**"
        else:
            prompt = f"Analyze 'attrition.csv' for Department '{sel_dept}'. Convert Exit_Date to 'YYYY-MM' format. Group by this Year-Month and count the exits. **Output as a Markdown Table sorted by date.**"
        
        st.session_state.prompt_trigger = prompt

    st.markdown("---")

    # 2. DATA QUALITY
    st.subheader("⚡ Ops & Data Quality")
    if st.button("🚀 Run Daily Data Audit"):
        st.session_state.prompt_trigger = "Run a data quality audit. If missing data is found, send correction emails."
    
    if st.button("📩 Simulate: Employees Reply"):
        report = simulate_employee_updates_logic()
        st.success("✅ Simulation Complete.")
        st.session_state.messages.append({"role": "assistant", "content": report})
        st.session_state.prompt_trigger = "Verify if the data remediation was successful."
        st.rerun()

    st.markdown("---")

    # 3. POLICY COPILOT
    st.subheader("✍️ Policy Email Copilot")
    copilot_emp_name = st.selectbox("Draft email for:", emp_names, key="copilot_select")
    copilot_scenario = st.selectbox("Select Scenario:", ["Relocation Request (International)", "Expense Reimbursement (Client Dinner)"])
    
    if st.button("📝 Draft Policy Response"):
        if copilot_scenario == "Relocation Request (International)":
            st.session_state.prompt_trigger = f"Draft a reply to {copilot_emp_name} regarding relocation to London. Explain Visa/Contract rules. **IMPORTANT: Display full email text in your response.**"
        else:
            st.session_state.prompt_trigger = f"Draft a reply to {copilot_emp_name} regarding client dinner reimbursement. Mention the 'No Alcohol' policy. **IMPORTANT: Display full email text in your response.**"
            
    st.markdown("---")

    # 4. COMP MODELER (Rich Format)
    st.subheader("💰 Compensation Modeler")
    comp_name = st.selectbox("Select Employee:", emp_names, key="comp_select") 
    hike_slider = st.slider("Hike %", 0, 40, 15)
    
    if st.button("Run Compensation Model"):
        # Safe lookup in case data changed
        try:
            selected_id = df_ui[df_ui['Name'] == comp_name]['Employee_ID'].values[0]
            result = calculate_hike_impact(selected_id, hike_slider)
            st.session_state.messages.append({"role": "assistant", "content": result})
            st.rerun()
        except:
            st.error("Employee not found. Try resetting data.")
        
    st.markdown("---")
    if st.button("⚠️ Reset Data"):
        reset_demo_data()
        st.warning("Data Reset. Refreshing...")
        st.rerun()

# ---------------------------------------------------------
# 3. MAIN CHAT INTERFACE
# ---------------------------------------------------------
st.title("🤖 Unified Agentic HR Assistant")
st.markdown("Ask about *Attrition*, *Engagement*, *Policy*, or *Data Quality*.")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I am ready to analyze workforce data."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"], avatar="🤖" if msg["role"]=="assistant" else "👤").write(msg["content"])

user_input = st.chat_input("Type your request here...")
if "prompt_trigger" in st.session_state and st.session_state.prompt_trigger:
    user_input = st.session_state.prompt_trigger
    del st.session_state.prompt_trigger

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user", avatar="👤").write(user_input)
    with st.chat_message("assistant", avatar="🤖"):
        st_callback = StreamlitCallbackHandler(st.container())
        agent = get_hr_agent()
        try:
            response = agent.invoke({"input": user_input}, {"callbacks": [st_callback]})
            st.write(response["output"])
            st.session_state.messages.append({"role": "assistant", "content": response["output"]})
        except Exception as e:
            st.error(f"Error: {e}")