import gc
import time
import torch
import pyaudio
import librosa
import argparse
import numpy as np
import multiprocessing
import sys
from utils_voice_assistant.preprocessor import Preprocessor
from utils_voice_assistant.streaming_buffer import StreamBuffer
from models_voice_assistant.stt_llm_tts_model import STT_LLM_TTS

TARGET_SAMPLE_RATE = 16000

def find_supported_audio_format(audio, device_index, verbose):
    # Assuming the device supports a commonly used sample rate if not found explicitly.
    supported_rates = [16000, 32000, 44100, 48000]
    supported_channels = [1, 2]  # Mono and Stereo
    found_rate = None
    found_channels = None

    if verbose:
        print(f"Checking for supported rates: {supported_rates}")
    for rate in supported_rates:
        try:
            if audio.is_format_supported(rate,
                                         input_device=device_index,
                                         input_channels=1,
                                         input_format=pyaudio.paFloat32):
                found_rate = rate
                break
        except ValueError:
            pass
    if verbose:
        print(f"Rate selected: {found_rate}")
        print('')
        print(f"Checking for supported channel counts: {supported_channels}")
    for channels in supported_channels:
        try:
            if audio.is_format_supported(found_rate,
                                         input_device=device_index,
                                         input_channels=channels,
                                         input_format=pyaudio.paFloat32):
                found_channels = channels
                break
        except ValueError:
            pass
    if verbose:
        print(f"Channel count selected: {found_channels}")
        print('')

    if found_rate is None or found_channels is None:
        print(f'Error: Audio device index [{device_index}]:')
        print(f'  We were unable to find an accepted sample rate or channels.')
        print(f'  rate found "{found_rate}". (Need {supported_rates})')
        print(f'  channels found "{found_channels}". (Need {supported_channels})')
        sys.exit(1)
    return found_rate, found_channels

def list_pyaudio_devices(audio):
    print("  Available pyaudio devices:")
    for i in range(audio.get_device_count()):
        dev = audio.get_device_info_by_index(i)
        print((i,dev['name'],dev['maxInputChannels']))

def record(audio, rate, channels, audio_buffer, start_recording, input_device_index, verbose) :
    """Record an audio stream from the microphone in a separate process  
        Args:
            audio_buffer: multiprocessing queue to store the recorded audio data
            start_recording: multiprocessing value to start and stop the recording
    """
    CHUNK = 2048

    # Open audio input stream
    if verbose:
        print(f"Attempting to use audio index {input_device_index}")
        print(f'  rate: {rate}')
        print(f'  ch: {channels}')
        print(f'  fmt: {pyaudio.paFloat32}')
        print(f'  idx: {input_device_index}')
        print(f'  frames per buffer: {CHUNK}')
    streamIn = audio.open(format=pyaudio.paFloat32, channels=channels,
                            rate=rate, input=True, input_device_index=input_device_index,
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
    import sounddevice as sd
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

def main_loop(device, audio_input_buffer, audio_output_buffer,  start_recording, sample_rate):
    """Wait for audio input, call voice assistant model and play synthesized speech  
        Args:
            streaming_buffer: streaming buffer instance to store preprocessed audio chunks
            model: instance of STT_LLM_TTS model
            audio_input_buffer: multiprocessing queue for audio input
            audio_output_buffer: multiprocessing queue for audio output
            start_recording: multiprocessing value to start recording of audio chunks
    """
    # init preprocessor
    preprocessor = Preprocessor()
    
    # Initialize buffer for processed audio input
    streaming_buffer = StreamBuffer(chunk_size=16, shift_size=16)
    streaming_buffer_iter = iter(streaming_buffer)

    # Initialize speech-to-text, language model, text-to-speech (STT-LLM-TTS) pipeline
    model = STT_LLM_TTS(device=device)
    
    # send signal to recording process to start the recording
    start_recording.value = 1

    # control buffer stream id for first chunk 
    first = True

    # start main loop
    while True:

        # get as many audio chunks from the buffer as possible. If the buffer is empty, an exception is thrown 
        # and the inner loop breaks
        while True:
            # select stream id (-1) for first chunk (0) else
            if first:
                stream_id = -1
                first = False
            else:
                stream_id = 0

            # try to get the next audio chunk, if buffer is empty an exception is thrown 
            try:
                # get audio data from buffer

                data = audio_input_buffer.get(block=False)
                
                # resample audio data to target sample rate of STT model
                t = np.frombuffer(data, dtype=np.float32)
                if sample_rate != TARGET_SAMPLE_RATE:
                    t = librosa.core.resample(t, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
                    t = t.transpose()
                t = torch.from_numpy(t)
                t = torch.unsqueeze(t,0)
                # preprocess audio data
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

        # TODO: Implement interrup behavior to stop audio process when user starts speaking

        # model return is None except when a new sentence is generated and synthesized 
        if text is not None:
            # --> A new sentence is finished
            print(text.replace("\n", ""))

            # Put synthesized audio to output buffer which will be played by the play-audio process
            audio_output_buffer.put(wav)
        
        time.sleep(0.001) # TODO Is this really needed?
                
def main():
    """Start the recording process in the main thread and all other processes in a separate process."""
    multiprocessing.set_start_method('spawn', force=True)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio-device-idx', type=int, default=0, help='Index of the audio device for recording')
    parser.add_argument('--audio-details', action='store_true', help='Display audio device info verbosely')
    args = parser.parse_args()

    list_pyaudio_devices(pyaudio.PyAudio())
    print(f"\nCurrently input device with id {args.audio_device_idx} is used for recording. To change the audio device, please use the --audio-device-idx parameter.\n")

    # Start multiprocessing queues and values
    audio_input_buffer = multiprocessing.Queue()
    audio_output_buffer = multiprocessing.Queue()
    start_recording = multiprocessing.Value('i', 0)

    # Initialize PyAudio in the main thread for recording
    audio = pyaudio.PyAudio()
    # get supported sample rate and number of channels for the given device
    sample_rate, audio_channels = find_supported_audio_format(audio, args.audio_device_idx, args.audio_details)

    # Determine processing device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        flush()  # Flush GPU memory if necessary

    # Start other processes in a separate process
        
    play_audio_process = multiprocessing.Process(target=play_audio, args=(audio_output_buffer,))
    play_audio_process.start()
        
    main_loop_process = multiprocessing.Process(target=main_loop, args=(device, audio_input_buffer, audio_output_buffer,  start_recording, sample_rate))
    main_loop_process.start()

    record(audio, sample_rate, audio_channels, audio_input_buffer, start_recording, args.audio_device_idx, args.audio_details)

    play_audio_process.join() 
    main_loop_process.join()

if __name__ == "__main__":
    main()