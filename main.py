import gc
import time
import torch
import pyaudio
import multiprocessing
import sounddevice as sd
from utils_voice_assistant.preprocessor import Preprocessor
from utils_voice_assistant.streaming_buffer import StreamBuffer
from models_voice_assistant.stt_llm_tts_model import STT_LLM_TTS


def record(audio_buffer, start_recording):
    """Record an audio stream from the microphone in a separate process  
        Args:
            audio_buffer: multiprocessing queue to store the recorded audio data
            start_recording: multiprocessing value to start and stop the recording
    """
    RATE = 16000
    CHUNK = 1024

    # Open audio input stream
    audio = pyaudio.PyAudio()
    streamIn = audio.open(format=pyaudio.paFloat32, channels=1,
                            rate=RATE, input=True, input_device_index=0,
                            frames_per_buffer=CHUNK)
    
    while(True):
        try:
            # start_recording is set to 1 in the main loop to start the recording
            if start_recording == 0:
                time.sleep(0.1)
                continue
            # read a chunk of fixed size from the input stream and add it to the input buffer 
            data = streamIn.read(CHUNK, exception_on_overflow=False)
            audio_buffer.put(data)

        except KeyboardInterrupt:
            return
        
        except Exception as e:
            raise e 
        
def play_audio(audio_output_buffer):
    """Play synthesized audio data in a separate process  
        Args:
            audio_output_buffer: multiprocessing-queue to receive audio data
    """
    fs = 24000
    while(True):
        # get next audio data 
        wav = audio_output_buffer.get()
        # play the audio and wait until it is finished (only this sub process is blocked, not the main loop)
        sd.play(wav, fs, blocking=True) 

def flush():
  """Flush Cuda cache to prevent side effect and slowdowns   
  """
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()

def main_loop(streaming_buffer, model, audio_input_buffer, audio_output_buffer,  start_recording):
    """Wait for audio input, call voice assistant model and play synthesized speech  
        Args:
            streaming_buffer: streaming buffer instance to store preprocessed audio chunks
            model: instance of STT_LLM_TTS model
            audio_input_buffer: multiprocessing queue for audio input
            audio_output_buffer: multiprocessing queue for audio output
            start_recording: multiprocessing value to start recording of audio chunks
    """
    # init preprocessor and streaming iterator
    preprocessor = Preprocessor()
    streaming_buffer_iter = iter(streaming_buffer)
    
    # send signal to recording process to start the recording
    start_recording.value = 1

    # control buffer stream id for first chunk 
    first_chunk = True
    first_response = True

    # start main loop
    while True:

        # get as many audio chunks from the buffer as possible. If the buffer is empty, an exception is thrown 
        # and the inner loop breaks
        while True:
            # select stream id (-1) for first chunk (0) else
            if first_chunk:
                stream_id = -1
                first_chunk = False
            else:
                stream_id = 0

            # try to get the next audio chunk, if buffer is empty an exception is thrown 
            try:
                # get audio data from buffer
                data = audio_input_buffer.get(block=False)
                
                # preprocess audio data
                t = torch.frombuffer(data, dtype=torch.float32)
                t = torch.unsqueeze(t,0)
                length = torch.tensor([t.shape[1]], dtype=torch.float32)
                processed_signal, _ = preprocessor(t, length)

                # add processed audio chunks to the streaming buffer 
                streaming_buffer.append_processed_signal(processed_signal, stream_id=stream_id)
            except Exception as e:
                # leave inner loop and process received data
                break
                
        # check if enough audio chunks were recorded for a forward path
        if streaming_buffer.buffer is not None and streaming_buffer.buffer.size(-1) > streaming_buffer.buffer_idx + streaming_buffer.shift_size:
            # --> enough chunks are available 

            # get preprocessed audio chunks from buffer
            data = next(streaming_buffer_iter, None)
            if data is None: 
                break
            chunk_audio, chunk_lengths = data
            
            # call model and pass preprocessed audio data
            chunk_audio = chunk_audio.to("cuda")
            chunk_lengths = chunk_lengths.to("cuda")
            text, wav, interrupt = model(chunk_audio, chunk_lengths) 
        else:
            # --> not enough chunks. Call model with empty input to generate text
            text, wav, interrupt = model(None, None)

        # clear buffer when receiving the first response from the model to delete every audio that was 
        # recorded before the model was fully initialized
        if first_response:
            first_response = False
            streaming_buffer.reset_buffer()
            first_chunk = True

        # TODO: Implement interrup behavior to stop audio process when user starts speaking

        # model return is None except when a new sentence is generated and synthesized 
        if text is not None:
            # --> A new sentence is finished
            print(text.replace("\n", ""))

            # Put synthesized audio to output buffer which will be played by the play-audio process
            audio_output_buffer.put(wav)
        
        time.sleep(0.001) # TODO Is this really needed?
                
def main():
    """ Start processes for recording and audio output, initialize voice assist model and start main loop 
    """
    # !! Make sure to start multiprocessing before using any pytorch tensors to prevent GPU memory problems !! 

    # start multiprocesses for sound input
    audio_buffer = multiprocessing.Queue() 
    start_recording = multiprocessing.Value('i', 0)
    record_process = multiprocessing.Process(target=record, args=(audio_buffer,start_recording))
    record_process.start()

    # start multiprocesses for sound output
    audio_output_buffer = multiprocessing.Queue()
    play_audio_process = multiprocessing.Process(target=play_audio, args=(audio_output_buffer,))
    play_audio_process.start()

    # initialize buffer for processed audio input
    streaming_buffer = StreamBuffer(chunk_size=16, shift_size=16)

    # get device
    if torch.cuda.is_available():
        device = 'cuda'
        # flush GPU memory
        flush()
    else:
        device = 'cpu'
        
    # init STT-LLM-TTS pipeline
    model = STT_LLM_TTS(device=device)

    # start inference
    main_loop(streaming_buffer, model, audio_buffer, audio_output_buffer,  start_recording)
   
if __name__ == "__main__":
    main()
