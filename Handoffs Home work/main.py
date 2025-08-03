from agents import Agent, Runner, trace
from connection import config
import asyncio


poet_agent = Agent(
    name='Poet Agent',
    instructions="""
        You are a poet agent.
        Your role is to generate a two-stanza poem or process an input poem.
        Poems can be lyric (emotional), narrative (storytelling), or dramatic (performance).
        If you are asked without a poem, generate a short two-stanza poem on emotions.
    """,
)

lyric_analyst_agent = Agent(
    name='Lyric Analyst Agent',
    instructions="""
        You analyze lyric poetry focusing on emotions, feelings, and musicality.
        Provide insights about the poem's mood, use of rhythm, and personal voice.
    """,
)

narrative_analyst_agent = Agent(
    name='Narrative Analyst Agent',
    instructions="""
        You analyze narrative poetry focusing on story telling elements: plot, characters, and imagery.
    """,
)

dramatic_analyst_agent = Agent(
    name='Dramatic Analyst Agent',
    instructions="""
        You analyze dramatic poetry emphasizing voice, dialogue, and performance aspects.
    """,
)

class CustomParentAgent(Agent):
    async def run(self, input, config):
        poet_output = await poet_agent.run(input, config)
        poem_text = poet_output.output.lower()

        if "dialogue" in poem_text or "voice" in poem_text or "stage" in poem_text:
            next_agent = dramatic_analyst_agent
        elif "story" in poem_text or "character" in poem_text or "event" in poem_text:
            next_agent = narrative_analyst_agent
        else:
            next_agent = lyric_analyst_agent

        final_output = await next_agent.run(poet_output.output, config)
        return final_output             

parent_agent = CustomParentAgent(
    name='Parent Poet Orchestrator',
    instructions="""
        You are the orchestrator agent for poetry tasks.
        When given a request or poem, first delegate to the poet agent to generate or process poem.
        After receiving the poem, detect whether it's lyric, narrative, or dramatic poetry.
        Delegate the poem to the corresponding analyst agent for deeper analysis.
        If the type is unclear or multiple types apply, delegate to all analysts.
        If the query is unrelated to poetry, respond politely and do not delegate.
    """,
    handoffs=[poet_agent, lyric_analyst_agent, narrative_analyst_agent, dramatic_analyst_agent]
)   

async def main():
    with trace("Handoffs Homework"):
        poem_or_query = """
        The sun sets in crimson fire,
        My heart aches with lost desire.

        In silent shadows, I remain,
        Whispering dreams in falling rain.
        """

        result = await Runner.run(
            parent_agent,
            poem_or_query,
            run_config=config
        )
        print("--- final output -----")
        print(result.final_output)
        print("--- Last agent -----")
        print(result.last_agent.name)

if __name__ == "__main__":
    asyncio.run(main())
