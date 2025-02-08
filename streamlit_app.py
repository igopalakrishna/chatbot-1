import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot that uses OpenAI's GPT-4o-mini model to generate responses. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    import gradio as gr
    import os
    import json
    import whisper
    
    from dotenv import load_dotenv
    
    from io import BytesIO
    from pydub import AudioSegment
    from pydub.playback import play
    
    # Initialization
    
    load_dotenv()
    
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if openai_api_key:
        print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
    else:
        print("OpenAI API Key not set")
        
    MODEL = "gpt-4o-mini"
    openai = OpenAI()
    
    system_message = "You are a professional AI interviewer conducting mock interviews."
    system_message += " Ask one thoughtful and relevant question at a time based on the candidate's profile."
    system_message += " Wait for a response before proceeding to the next question."
    system_message += " Conclude the interview naturally after all key areas have been addressed or when it feels appropriate based on the flow of conversation."
    system_message += " Indicate the end of the interview politely and thank the candidate for their time."
    
    ask_question_function = {
        "name": "ask_question",
        "description": "Ask a question during an interview based on the candidate's profile and the current topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic or domain of the question (e.g., programming, behavioral, problem-solving)."
                },
                "difficulty_level": {
                    "type": "string",
                    "enum": ["easy", "medium", "hard"],
                    "description": "The difficulty level of the question."
                }
            },
            "required": ["topic", "difficulty_level"],
            "additionalProperties": False
        }
    }
    
    evaluate_answer_function = {
        "name": "evaluate_answer",
        "description": "Evaluate the candidate's response to a question and provide constructive feedback.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The candidate's response to the question."
                },
                "expected_qualities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of qualities or key points expected in the answer (e.g., clarity, technical accuracy, examples)."
                }
            },
            "required": ["answer", "expected_qualities"],
            "additionalProperties": False
        }
    }
    
    conclude_interview_function = {
        "name": "conclude_interview",
        "description": "Determine whether to conclude the interview based on the flow and key topics covered.",
        "parameters": {
            "type": "object",
            "properties": {
                "key_topics_covered": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of key topics or areas that have already been addressed."
                },
                "candidate_engagement": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "The candidate's level of engagement during the interview."
                }
            },
            "required": ["key_topics_covered", "candidate_engagement"],
            "additionalProperties": False
        }
    }
    
    follow_up_function = {
        "name": "ask_follow_up",
        "description": "Ask a follow-up question to prompt the candidate to elaborate on their response.",
        "parameters": {
            "type": "object",
            "properties": {
                "previous_answer": {
                    "type": "string",
                    "description": "The candidate's previous response that needs further elaboration."
                },
                "specific_aspect": {
                    "type": "string",
                    "description": "The specific aspect of the answer that needs clarification or more detail."
                }
            },
            "required": ["previous_answer", "specific_aspect"],
            "additionalProperties": False
        }
    }
    
    provide_feedback_function = {
        "name": "provide_feedback",
        "description": "Summarize the candidate's performance and provide constructive feedback.",
        "parameters": {
            "type": "object",
            "properties": {
                "strengths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key strengths demonstrated by the candidate during the interview."
                },
                "areas_for_improvement": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Suggestions for areas where the candidate can improve."
                }
            },
            "required": ["strengths", "areas_for_improvement"],
            "additionalProperties": False
        }
    }
    
    tools = [
        {"type": "function", "function": ask_question_function},
        {"type": "function", "function": evaluate_answer_function},
        {"type": "function", "function": conclude_interview_function},
        {"type": "function", "function": follow_up_function},
        {"type": "function", "function": provide_feedback_function}
    ]
    
    def handle_tool_call(message):
        """Handle tool calls based on the AI response."""
        tool_call = message.tool_calls[0]  # Assuming one tool call at a time
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
    
        response_content = {}
    
        if function_name == "ask_question":
            topic = arguments.get("topic")
            difficulty = arguments.get("difficulty_level")
            response_content = {"question": f"Here is a {difficulty} question on {topic}."}
    
        elif function_name == "evaluate_answer":
            answer = arguments.get("answer")
            expected_qualities = arguments.get("expected_qualities", [])
            response_content = {
                "evaluation": f"The answer '{answer}' is evaluated based on {expected_qualities}."
            }
    
        elif function_name == "conclude_interview":
            key_topics = arguments.get("key_topics_covered", [])
            candidate_engagement = arguments.get("candidate_engagement")
            response_content = {
                "conclusion": f"Interview is concluded after covering {key_topics}. Engagement level: {candidate_engagement}."
            }
    
        elif function_name == "ask_follow_up":
            previous_answer = arguments.get("previous_answer")
            specific_aspect = arguments.get("specific_aspect")
            response_content = {
                "follow_up": f"Can you elaborate on '{specific_aspect}' in your answer: '{previous_answer}'?"
            }
    
        elif function_name == "provide_feedback":
            strengths = arguments.get("strengths", [])
            areas_for_improvement = arguments.get("areas_for_improvement", [])
            response_content = {
                "feedback": f"Strengths: {strengths}. Areas for improvement: {areas_for_improvement}."
            }
    
        response = {
            "role": "tool",
            "content": json.dumps(response_content),
            "tool_call_id": message.tool_calls[0].id
        }
        
        return response
    
    from io import BytesIO
    from pydub import AudioSegment
    from pydub.playback import play

    def talker(message):
        response = openai.audio.speech.create(
          model="tts-1",
          voice="onyx",    # Also, try replacing onyx with alloy
          input=message
        )
        
        audio_stream = BytesIO(response.content)
        audio = AudioSegment.from_file(audio_stream, format="mp3")
        play(audio)
    
    import whisper 
    model  = whisper.load_model("tiny.en")
    
    def transcribe(audio_file):
        speech_to_text = model.transcribe(audio_file)["text"]
    
        return speech_to_text
    
    def handle_audio(audio_file, history):
        """Handle user voice input, transcribe it, and provide an audio response."""
        if audio_file is not None:
            try:
                # Transcribe the audio
                text = transcribe(audio_file)
                
                # Update history with user input
                history.append({"role": "user", "content": text})
                
                # Generate AI response
                response = openai.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": system_message}] + history,
                    tools = tools
                )
    
                if response.choices[0].finish_reason=="tool_calls":
                    message = response.choices[0].message
                    response = handle_tool_call(message)
                    messages.append(message)
                    messages.append(response)
                    response = openai.chat.completions.create(model=MODEL, messages=messages)
                
                # Access the AI response message content
                reply = response.choices[0].message.content
                
                # Update history with AI response
                history.append({"role": "assistant", "content": reply})
                
                # Respond using text-to-speech
                talker(reply)
                
                return history  # Return updated chatbot display
            finally:
                if os.path.exists(audio_file):
                    os.remove(audio_file)  # Clean up temporary file
    
            return history
    
    with gr.Blocks() as ui:
        with gr.Row():
            chatbot = gr.Chatbot(height=500, type="messages", label="AI Assistant")  # Chatbot to display conversation
    
        with gr.Row():
            audio_input = gr.Audio(type="filepath", label="Speak to AI")  # Audio input for user voice
    
        with gr.Row():
            clear = gr.Button("Clear")
    
        # Audio input handling
        audio_input.stop_recording(
            handle_audio, 
            inputs=[audio_input, chatbot], 
            outputs=chatbot
        ).then(
            lambda history: history, 
            inputs=[chatbot], 
            outputs=chatbot
        )
    
        # Clear button to reset history
        clear.click(lambda: ([{"role": "system", "content": system_message}], [{"role": "system", "content": system_message}]), 
                    inputs=None, outputs=chatbot, queue=False)
    
    ui.launch(inbrowser=True)
            



    




    
    


    
    
    
