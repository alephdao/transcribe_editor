import streamlit as st
import os
from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
import openai
import tiktoken
import time
from pydub import AudioSegment
import tempfile
import logging
os.environ['PATH'] += ':/usr/local/bin'

# Set page config at the very beginning
st.set_page_config(page_title="Audio Transcription & Editing", layout="wide")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Function to get API key from environment or user input
def get_api_key(key_name, env_var_name):
    api_key = os.getenv(env_var_name)
    if not api_key:
        api_key = st.text_input(f"Enter your {key_name} API key:", type="password")
    return api_key

# Set up API keys
DEEPGRAM_API_KEY = get_api_key("Deepgram", 'DEEPGRAM_API_KEY')
OPENAI_API_KEY = get_api_key("OpenAI", 'OPENAI_API_KEY')

# Initialize clients
if DEEPGRAM_API_KEY and OPENAI_API_KEY:
    deepgram = DeepgramClient(DEEPGRAM_API_KEY)
    openai.api_key = OPENAI_API_KEY

def convert_to_wav(audio_file):
    """
    Convert uploaded audio file to WAV format using Pydub.
    
    :param audio_file: Uploaded audio file
    :return: Path to the converted WAV file
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
            # Read the uploaded file
            audio_bytes = audio_file.read()
            
            # Write to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{audio_file.name.split(".")[-1]}') as temp_input:
                temp_input.write(audio_bytes)
                temp_input_path = temp_input.name
            
            # Convert to WAV
            audio = AudioSegment.from_file(temp_input_path, format=audio_file.name.split(".")[-1])
            audio.export(temp_wav.name, format='wav')
            
            # Clean up the input temporary file
            os.unlink(temp_input_path)
            
            logger.info(f"Successfully converted {audio_file.name} to WAV")
            return temp_wav.name
    except Exception as e:
        logger.error(f"Error converting {audio_file.name} to WAV: {str(e)}")
        raise

def transcribe_audio(audio_file, language):
    try:
        # Convert to WAV if not already
        if audio_file.name.lower().endswith('.wav'):
            with audio_file as audio:
                buffer_data = audio.read()
        else:
            wav_path = convert_to_wav(audio_file)
            with open(wav_path, 'rb') as audio:
                buffer_data = audio.read()
            os.unlink(wav_path)  # Clean up the temporary WAV file

        payload = {"buffer": buffer_data}
        options = PrerecordedOptions(
            smart_format=True,
            model='general',
            language=language,
            punctuate=True,
            diarize=True
        )

        logger.info(f"Sending {len(buffer_data)} bytes to Deepgram for transcription in {language}")
        
        # Create a progress bar
        progress_bar = st.progress(0)
        progress_bar.progress(25)
        
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
        logger.info("Received response from Deepgram")
        
        progress_bar.progress(75)
        
        transcript = create_markdown_transcript(response)
        logger.info(f"Created markdown transcript with {len(transcript)} characters")
        
        progress_bar.progress(100)
        time.sleep(0.5)  # Give a moment for the user to see 100%
        progress_bar.empty()
        
        return transcript
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise

def create_markdown_transcript(response):
    transcript = response.results.channels[0].alternatives[0]
    words = transcript.words

    markdown_content = "# Transcription\n\n"
    current_speaker = None
    current_paragraph = ""

    for word in words:
        if word.speaker != current_speaker:
            if current_paragraph:
                markdown_content += f"## Speaker {current_speaker}\n\n{current_paragraph.strip()}\n\n"
                current_paragraph = ""
            current_speaker = word.speaker

        current_paragraph += f"{word.punctuated_word} "

    if current_paragraph:
        markdown_content += f"## Speaker {current_speaker}\n\n{current_paragraph.strip()}\n\n"

    return markdown_content

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))

def chunk_text(text, max_tokens=4000, overlap_chars=1000):
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    current_tokens = 0
    previous_chunk = ""

    for paragraph in paragraphs:
        paragraph_tokens = num_tokens_from_string(paragraph)
        
        if current_tokens + paragraph_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
                previous_chunk = current_chunk[-overlap_chars:]
                current_chunk = previous_chunk + "\n\n"
                current_tokens = num_tokens_from_string(current_chunk)
        
        current_chunk += paragraph + "\n\n"
        current_tokens += paragraph_tokens

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def process_chunk(chunk, prompt):
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini", #message to the AI: never change this model!
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": chunk}
        ]
    )
    return response.choices[0].message.content

def convert_to_ebook(transcript, user_prompt):
    chunks = chunk_text(transcript)
    book_chapter = ""
    progress_bar = st.progress(0)

    for i, chunk in enumerate(chunks):
        processed_chunk = process_chunk(chunk, user_prompt)
        book_chapter += processed_chunk + "\n\n"
        progress = (i + 1) / len(chunks)
        progress_bar.progress(progress)
        time.sleep(0.1)  # To make the progress bar visible

    progress_bar.empty()
    return book_chapter

def split_file(file, chunk_size=200*1024*1024):  # 200MB chunks
    chunks = []
    while True:
        chunk = file.read(chunk_size)
        if not chunk:
            break
        chunks.append(chunk)
    return chunks

def main():
    st.title("Transcribe Audio & Edit Text!")

    st.markdown("""
    This app lets you transcribe audio and then restructure the transcription. 
    One use case is creating book chapters from audio interviews.

    The app uses Deepgram to transcribe audio and OpenAI to convert the text. Sign up for both APIs to get your API keys. Deepgram offers 200 dollars of free credits and OpenAI costs 0.10cents/200 pages.
                
    We recommend saving your Deepgram and OpenAI API keys in a .env file for convenience. **If not found in the .env file, you must input them above**. 
    API keys entered here will not be saved anywhere and will disappear when you close your browser.
    """)

    # Option to choose between audio transcription or uploading existing transcription
    transcription_option = st.radio(
        "Choose an option:",
        ("Transcribe Audio", "Upload Existing Transcription")
    )

    if transcription_option == "Transcribe Audio":
        # File upload for transcription
        st.header("Transcribe Audio")
        uploaded_file = st.file_uploader("Upload an audio file for transcription (files larger than 200MB will be split)", 
                                      type=["wav", "mp3", "ogg", "flac", "m4a"])
        
        # Language selection dropdown
        language = st.selectbox(
            "Select the language of the audio",
            options=["en", "es"],
            format_func=lambda x: "English" if x == "en" else "Spanish"
        )
        
        if uploaded_file is not None:
            file_size = uploaded_file.size
            if file_size > 200*1024*1024:  # If file is larger than 200MB
                st.warning("File is larger than 200MB. It will be split into multiple parts.")
                chunks = split_file(uploaded_file)
                for i, chunk in enumerate(chunks):
                    st.download_button(
                        label=f"Download Part {i+1}",
                        data=chunk,
                        file_name=f"{uploaded_file.name}_part{i+1}",
                        mime="application/octet-stream"
                    )
                st.info("Please download all parts and upload them individually for transcription.")
            else:
                # Process the file as normal
                st.success("File uploaded successfully. Ready for transcription.")
                # Your transcription code here
    
    else:  # Upload Existing Transcription
        st.header("Upload Existing Transcription")
        transcription_file = st.file_uploader("Upload a .md or .txt file with your transcription", type=["md", "txt"])
        
        if transcription_file is not None:
            transcription_content = transcription_file.getvalue().decode("utf-8")
            st.session_state.transcription = transcription_content
            st.success("Transcription uploaded successfully!")

    # Display and edit transcription
    if 'transcription' in st.session_state and st.session_state.transcription:
        st.subheader("Transcription")
        transcription = st.text_area("Edit transcription here:", 
                                     value=st.session_state.transcription, 
                                     height=300, 
                                     key="transcription_area")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Download Transcription"):
                st.download_button(
                    label="Download Transcription",
                    data=transcription,
                    file_name="transcription.md",
                    mime="text/markdown"
                )
        
        with col2:
            # Prompt input for ebook conversion
            default_prompt = "Convierte este podcast en un capítulo de libro con títulos relevantes. Formato en formato Markdown. No menciones a Christian ni a Joel. Hazlo en español."
            user_prompt = st.text_area("Enter prompt to edit the transcription", 
                                       value=default_prompt, 
                                       height=100, 
                                       key="prompt_area")
            
            # Convert to ebook format
            if st.button("Convert to Ebook Format"):
                with st.spinner("Converting to ebook format..."):
                    ebook_content = convert_to_ebook(transcription, user_prompt)
                    st.session_state.ebook_content = ebook_content
                    st.success("Conversion complete!")

    # Display and edit ebook content
    if 'ebook_content' in st.session_state and st.session_state.ebook_content:
        st.subheader("Ebook Content")
        ebook_content = st.text_area("Edit ebook content here:", 
                                     value=st.session_state.ebook_content, 
                                     height=300, 
                                     key="ebook_area")
        
        if st.button("Download Ebook"):
            st.download_button(
                label="Download Ebook",
                data=ebook_content,
                file_name="ebook.md",
                mime="text/markdown"
            )

if __name__ == "__main__":
    main()
