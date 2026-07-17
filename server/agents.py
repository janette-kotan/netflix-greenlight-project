import os
import json


# Import LLM directly from crewai to satisfy internal Pydantic validation
from crewai import Agent, Task, Crew, Process, LLM

# Define the local Ollama connection using CrewAI's native class
local_llm = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434"
)

# ---------------------------------------------------------
# Agent Persona 1: The Theme Extractor Agent
# ---------------------------------------------------------
theme_extractor = Agent(
    role="Expert Creative Content Analyst",
    goal="Parse raw, unstructured creative movie and TV pitches to extract core themes, emotional hooks, and tropes.",
    backstory="You are a veteran Hollywood development executive. You excel at reading raw, messy brainstorming text or plot descriptions and instantly identifying the underlying creative hooks, genre blends, and narrative elements.",
    llm=local_llm,
    verbose=True,
    allow_delegation=False
)

# ---------------------------------------------------------
# Agent Persona 2: The Feature Mapper Agent
# ---------------------------------------------------------
feature_mapper = Agent(
    role="Structured Data Alignment Specialist",
    goal="Map raw narrative themes and keywords to a rigid, predefined list of 730 machine learning features.",
    backstory="You are an expert analytics translation engine. Your job is to take qualitative creative themes (like 'anti-hero', 'dystopian', 'spaceship') and systematically map them to a strict index of target features, outputting clean structured data for predictive pipelines.",
    llm=local_llm,
    verbose=True,
    allow_delegation=False
)

def test_agent_pipeline():
    test_pitch = "A story about a girl from the Joseon era in Korea reincarnating as an actress in the modern day and falling in love with a wealthy CEO."
    
    # 1. Define the parsing task
    extract_task = Task(
        description=f"Analyze this raw creative pitch: '{test_pitch}'. Identify the primary genres, tone, and active tropes/keywords.",
        expected_output="A bulleted list of extracted genres, tone tags, and narrative keywords.",
        agent=theme_extractor
    )
    
    # 2. Assemble the single-task test crew
    test_crew = Crew(
        agents=[theme_extractor],
        tasks=[extract_task],
        process=Process.sequential
    )
    
    print("Initializing test crew connection to local Ollama instance...")
    result = test_crew.kickoff()
    print("\n--- TEST OUTCOME FROM LOCAL LLM ---")
    print(result.raw)


def run_production_agent_pipeline(user_pitch: str):
    # 1. Task for the Theme Extractor
    analysis_task = Task(
        description=f"Analyze this raw creative pitch: '{user_pitch}'. Identify the primary genres, tone, and active tropes/keywords.",
        expected_output="A clean breakdown of extracted genres, tone tags, and narrative keywords.",
        agent=theme_extractor
    )
    
    # 2. Task for the Feature Mapper (Enforcing rigid JSON structure)
    mapping_task = Task(
        description=(
            "Review the creative breakdown provided by the Content Analyst. Translate these qualitative findings "
            "into a clean, structured JSON object containing exactly two keys: 'genres' and 'keywords'. "
            "The values must be plain text arrays matching structural dataset targets. Do not output markdown, "
            "do not include backticks, do not include conversational prose. Return only raw, valid JSON."
        ),
        expected_output="A raw JSON string with keys 'genres' and 'keywords'. Example: {'genres': ['Sci-Fi'], 'keywords': ['robot']}",
        agent=feature_mapper
    )
    
    # 3. Assemble the full production Crew executing sequentially
    production_crew = Crew(
        agents=[theme_extractor, feature_mapper],
        tasks=[analysis_task, mapping_task],
        process=Process.sequential,
        verbose=True
    )
    
    print(f"\nProcessing creative concept pitch via local intelligence layer...")
    crew_output = production_crew.kickoff()
    
    print("\n--- RAW AGENT LAYER OUTPUT ---")
    print(crew_output.raw)
    return crew_output.raw

if __name__ == "__main__":
    sample_pitch = "A story about a girl from the Joseon era in Korea reincarnating as an actress in the modern day and falling in love with a wealthy CEO."
    run_production_agent_pipeline(sample_pitch)
    test_agent_pipeline()