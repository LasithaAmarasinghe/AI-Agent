from datetime import datetime
from agno.agent import Agent
from agno.playground import Playground, serve_playground_app
from agno.storage.sqlite import SqliteStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.models.groq import Groq
from agno.tools.exa import ExaTools
from textwrap import dedent
from dotenv import load_dotenv

load_dotenv()

agent_storage: str = "tmp/agents.db"
today = datetime.now().strftime("%Y-%m-%d")

web_agent = Agent(
    name="Web Agent",
    model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct"),
    tools=[DuckDuckGoTools()],
    instructions=["Always include sources"],
    storage=SqliteStorage(table_name="web_agent", db_file=agent_storage),
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True,
)

news_agent = Agent(
    name="News Agent",
    model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct"),
    tools=[DuckDuckGoTools()],
    instructions=["Always include sources"],
    storage=SqliteStorage(table_name="news_agent", db_file=agent_storage),
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True)],
    instructions=["Always use tables to display data"],
    storage=SqliteStorage(table_name="finance_agent", db_file=agent_storage),
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True,
)

research_agent = Agent(
    name="Research Agent",
    model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"),
    tools=[ExaTools(start_published_date=today, type="keyword")],
    description=dedent("""\
        You are a distinguished AI research scientist with expertise
        in analyzing and synthesizing complex information. Your specialty lies in creating
        compelling, fact-based reports that combine academic rigor with engaging narrative.

        Your writing style is:
        - Clear and authoritative
        - Engaging but professional
        - Fact-focused with proper citations
        - Accessible to educated non-specialists\
    """),
    expected_output=dedent("""\
    A professional research report in markdown format:

    # {Compelling Title That Captures the Topic's Essence}

    ## Executive Summary
    {Brief overview of key findings and significance}

    ## Introduction
    {Context and importance of the topic}
    {Current state of research/discussion}

    ## Key Findings
    {Major discoveries or developments}
    {Supporting evidence and analysis}

    ## Implications
    {Impact on field/society}
    {Future directions}

    ## Key Takeaways
    - {Bullet point 1}
    - {Bullet point 2}
    - {Bullet point 3}

    ## References
    - [Source 1](link) - Key finding/quote
    - [Source 2](link) - Key finding/quote
    - [Source 3](link) - Key finding/quote

    ---\
    """),
    show_tool_calls=True,
    add_datetime_to_instructions=True,
    storage=SqliteStorage(table_name="research_agent", db_file=agent_storage),
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True,
    )


app = Playground(agents=[web_agent, news_agent, finance_agent, research_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)