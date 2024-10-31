import streamlit as st
from crewai import Agent, Task, Crew, Process
from textwrap import dedent
from langchain_openai import ChatOpenAI
from openai import OpenAI
import os
from datetime import datetime
from dotenv import load_dotenv
import requests
import json

## Changes needed:
## Create reel using photos & text content

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY in your .env file!")

# Initialize OpenAI
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

class EventPostGenerator:
    def __init__(self, event_details):
        self.event_details = event_details
        self.setup_agents()
        self.setup_tasks()
        self.setup_crew()

    ## system_prompt: This is the agent's prompt
    ## user_prompt: This is the prompt from the previous agent or from user
    ## agent_model: You can give different models. Checkout more models for each agent here. You can try different ones for different agents: https://openrouter.ai/docs/models
    ## temperature: Give a higher number for more variety and randomness in the output. Good for IF content & pic
    def get_openrouter_response(self, system_prompt, user_prompt, agent_model, temperature):
        ## You can give multiple models in this dictionary, and use this function to call with diffferent models each time
        or_models = {"meta-llama3":"meta-llama/llama-3-70b-instruct",
                      "meta-llama3.2":"meta-llama/llama-3.2-11b-vision-instruct:free"}
        
        ## This call fetches the model as per the paramter from the dict. If nothing matches, defaults to "llama 3"
        model = or_models.get(agent_model, "llama 3")
        response = ""
        # total_tokens = 0

        try:
            OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
            system_message = [{"role": "system", "content": system_prompt}]
            user_message = [{"role": "user", "content": user_prompt}]
            
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                data=json.dumps({
                    "model": model,
                    "messages": system_message+user_message,
                    # "top_k": voice_settings['top_k'],
                    # "top_p": voice_settings['top_p'],
                    # "max_tokens": voice_settings['max_tokens'],
                    # "min_tokens": voice_settings['min_tokens'],
                    "temperature": temperature,
                    # "length_penalty": voice_settings['length_penalty'],
                    # "presence_penalty": voice_settings['presence_penalty'],
                    # "frequency_penalty": voice_settings['frequency_penalty'],
                    # "repetition_penalty": voice_settings['repetition_penalty']
                })
            )
            response_str = response.content.decode('utf-8')
            data = json.loads(response_str)
            response = data['choices'][0]['message']['content']
            # total_tokens = data['usage']['total_tokens']
        except Exception as e:
            error = "Error: {}".format(str(e))
        return response

    def generate_event_image(self, prompt):
        try:
            ## Function call to generate image, based on model passed as parameter 
            response = openai_client.images.generate(
                model="dall-e-3",
                prompt=f"Create a professional, eye-catching social media banner for {prompt}. The style should be modern and engaging, perfect for Instagram. Include visual elements related to {self.event_details['theme']}.",
                size="1024x1024",
                quality="standard",
                n=1,
            )
            return response.data[0].url
        except Exception as e:
            st.error(f"Error generating image: {str(e)}")
            return None

    def setup_agents(self):
        self.headline_creator = Agent(
            role='Headline Creator',
            goal='Create an attention-grabbing headline for the event',
            backstory=dedent("""
                You are a master of creating viral headlines that capture attention
                instantly. You know how to use emojis effectively and create excitement
                in just one line.
            """),
            verbose=True,
            llm=llm
        )

        self.quote_creator = Agent(
            role='Inspirational Quote Creator',
            goal='Create an exciting and relevant quote about the event ',
            backstory="You create powerful, memorable quotes that inspire action and create FOMO in one line.",
            verbose=True,
            llm=llm
        )

        self.details_formatter = Agent(
            role='Event Details Formatter',
            goal='Format event details in an exciting and readable way',
            backstory="You transform boring event details into exciting announcements.",
            verbose=True,
            llm=llm
        )

        self.cta_specialist = Agent(
            role='Call-to-Action Specialist',
            goal='Create compelling registration and joining statements',
            backstory="You specialize in creating urgency and excitement for event registration.",
            verbose=True,
            llm=llm
        )

        self.hashtag_specialist = Agent(
            role='Hashtag Specialist',
            goal='Generate trending and relevant hashtags',
            backstory="You create perfect hashtag combinations that increase post visibility.",
            verbose=True,
            llm=llm
        )

    def setup_tasks(self):
        self.headline_task = Task(
            description=dedent(f"""
                Create a powerful one-line headline for {self.event_details['event_name']}.
                Use relevant emojis and make it attention-grabbing.
                Theme: {self.event_details['theme']}
                Return only the headline, nothing else.
            """),
            expected_output="One-line headline with emojis",
            agent=self.headline_creator
        )

        self.quote_task = Task(
            description=dedent(f"""
                Create an exciting quote about {self.event_details['event_name']} and its theme: {self.event_details['theme']}.
                Make it inspiring and memorable.
                Return only the quote, nothing else.
            """),
            expected_output="Inspiring quote",
            agent=self.quote_creator
        )

        self.details_task = Task(
            description=dedent(f"""
                Format these event details in an exciting way:
                Event: {self.event_details['event_name']}
                Date: {self.event_details['date']}
                Venue: {self.event_details['venue']}
                Theme: {self.event_details['theme']}
                Use appropriate emojis and formatting.
                Return only the formatted details, nothing else.
            """),
            expected_output="Formatted event details",
            agent=self.details_formatter
        )

        self.cta_task = Task(
            description=dedent(f"""
                Create an exciting registration call-to-action using this link: {self.event_details['registration_link']}
                Also create a compelling "Join us!" closing statement.
                Make them urgent and exciting.
                Return only the CTA and closing statement, nothing else.
            """),
            expected_output="Registration CTA and closing statement",
            agent=self.cta_specialist
        )

        self.hashtag_task = Task(
            description=dedent(f"""
                Create 8-10 relevant hashtags for this event:
                Event: {self.event_details['event_name']}
                Theme: {self.event_details['theme']}
                Combine trending, local, and theme-specific tags.
                Return only the hashtags, nothing else.
            """),
            expected_output="List of effective hashtags",
            agent=self.hashtag_specialist
        )

    def setup_crew(self):
        self.crew = Crew(
            agents=[
                self.headline_creator,
                self.quote_creator,
                self.details_formatter,
                self.cta_specialist,
                self.hashtag_specialist
            ],
            tasks=[
                self.headline_task,
                self.quote_task,
                self.details_task,
                self.cta_task,
                self.hashtag_task
            ],
            verbose=True,
            process=Process.sequential
        )

    def generate_post(self):
        try:
            # Get results from crew
            results = self.crew.kickoff()
            
            # Extract content from tasks_output
            if hasattr(results, 'tasks_output'):
                outputs = []
                for task_output in results.tasks_output:
                    # Get the raw content from each task output
                    if hasattr(task_output, 'raw'):
                        outputs.append(task_output.raw)
                    else:
                        outputs.append(str(task_output))
            else:
                st.error("Unexpected results structure")
                st.json(str(results))
                return "Error generating content"

            # Format the post with all outputs
            post = f"""
{outputs[0]}  

{outputs[1]}  

{outputs[2]}

{outputs[3]}

.
.
.

{outputs[4]}
"""
            return post.strip()
            
        except Exception as e:
            st.error(f"Error formatting post: {str(e)}")
            st.error("Results structure:")
            st.json(str(results))
            return "Error generating post content. Please try again."

def main():
    st.set_page_config(page_title="College Event Post Generator", layout="wide")
    
    # Custom CSS for Instagram-like styling
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .post-container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .event-image {
            width: 100%;
            border-radius: 8px;
        }
        .post-content {
            white-space: pre-line;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            font-size: 16px;
            line-height: 1.5;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🎯 College Event Post Generator")
    
    # Form for event details
    with st.form("event_details_form"):
        col1, col2 = st.columns(2)
        with col1:
            event_name = st.text_input("Event Name", "")
            date = st.date_input("Event Date")
        with col2:
            theme = st.text_input("Event Theme", "")
            venue = st.text_input("Venue", "")
        
        registration_link = st.text_input("Registration Link", "")
        submit_button = st.form_submit_button("Generate Post")

    if submit_button and event_name and theme:
        with st.spinner("Creating your event post..."):
            event_details = {
                "event_name": event_name,
                "date": date.strftime("%B %d, %Y"),
                "venue": venue,
                "theme": theme,
                "registration_link": registration_link
            }
            
            generator = EventPostGenerator(event_details)
            
            # Generate image
            image_url = generator.generate_event_image(f"{event_name} - {theme}")
            
            # Generate content
            content = generator.generate_post()
            
            # Display results in Instagram-like format
            st.markdown("## 📱 Preview")
            col1, col2, col3 = st.columns([1,3,1])
            with col2:
                with st.container():
                    st.markdown('<div class="post-container">', unsafe_allow_html=True)
                    if image_url:
                        st.image(image_url, use_column_width=True)
                    st.markdown('<div class="post-content">', unsafe_allow_html=True)
                    st.markdown(content)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Add copy button
                    if st.button("📋 Copy Post Content"):
                        st.code(content, language="text")
                        st.success("Content copied to clipboard!")

if __name__ == "__main__":
    main()