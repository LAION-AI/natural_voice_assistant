import time
import torch
import numpy as np
from copy import deepcopy
from utils_voice_assistant.nemo_loader import load_rnnt_model
from models_voice_assistant.TTS.style_tts2_model import StyleTTS2Model
from transformers import AutoTokenizer, AutoModelForCausalLM

import json

VERBOSE = False

class STT(torch.nn.Module):
    # Constants or FastConformer RNNT model
    LAST_CHANNEL_CACHE_SIZE = 70
    MAX_SYMBOLS = 10
    SOS = 1024
    BLANK_INDEX = 1024
    PRED_RNN_LAYERS = 1

    # Options for loading model checkpoint from NGC cloud
    NEMO_URL = "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_hybrid_large_streaming_80ms/versions/1.20.0/files/stt_en_fastconformer_hybrid_large_streaming_80ms.nemo"
    NEMO_DESCRIPTION = "For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_hybrid_large_streaming_80ms"

    def __init__(self, vocabulary_path):
        """Initialize the encoder, decoder and joint model of the Streaming STT FastConformer.
        Args:
            vocabulary_path: Valid file path to a .npy file containing the vocabulary of the Joint model
        Returns:
            Instantce of the Streaming STT model
        """
        super(STT, self).__init__()

        # load vocabulary
        self.vocabulary = list(np.load(vocabulary_path))

        # load encoder, decoder and joint of FastConformer RNNT model (checkpint is downloaded and cahched)
        enc_dec_rnnt_model = load_rnnt_model(self.NEMO_URL, self.NEMO_DESCRIPTION)
        self.encoder = enc_dec_rnnt_model.encoder.eval()
        self.decoder = enc_dec_rnnt_model.decoder.eval()
        self.joint = enc_dec_rnnt_model.joint.eval()

        # Init encoder state
        self.cache_last_channel, self.cache_last_time, self.cache_last_channel_len = self.encoder.get_initial_cache_state(batch_size=1)
        
        ## Init decoder state
        self.y_sequence = []
        self.dec_state = None
        self.last_token = None

    def forward(self,processed_signal, processed_signal_length):
        """Perform an encoding, decoding and joint step on the given audio signal.
        Args:
            processed_signal: preprocessed audio chunk 
            processed_signal_length: number of encodedings in the chunk 
        Returns:
            Sequence of transcribed tokens
        """
        with torch.no_grad():
            ## call encoder
            with torch.jit.optimized_execution(False): 
                encoder_output = self.encoder(
                audio_signal=processed_signal,
                length=processed_signal_length,
                cache_last_channel=self.cache_last_channel,
                cache_last_time=self.cache_last_time,
                cache_last_channel_len=self.cache_last_channel_len,
            )
            encoded,encoded_len,self.cache_last_channel,self.cache_last_time,self.cache_last_channel_len = encoder_output
            self.cache_last_channel = self.cache_last_channel[:, :, -self.LAST_CHANNEL_CACHE_SIZE :, :]

            ## call decoder and joint
            with torch.inference_mode():
                self.y_sequence, self.dec_state, self.last_token  = self.greedy_RNNT_decode(
                    encoder_output=encoded, 
                    encoded_lengths=encoded_len, 
                    y_sequence=self.y_sequence, 
                    dec_state= self.dec_state, 
                    last_token=self.last_token)

            return self.y_sequence

    def states_to_device(self, dec_state, device='cpu'):
        """Maps the decoding state to the given device
        Args:
            dec_state: hidden state of the decoder model
            device: target device 
        Returns:
            Hidden state of the decoder mapped to the target device
        """
        if torch.is_tensor(dec_state):
            dec_state = dec_state.to(device)
        elif isinstance(dec_state, (list, tuple)):
            dec_state = tuple(self.states_to_device(dec_i, device) for dec_i in dec_state)
        return dec_state

    def label_collate(self, labels, device=None):
        if isinstance(labels, torch.Tensor):
            return labels.type(torch.int64)

        batch_size = len(labels)
        max_len = max(len(label) for label in labels)

        cat_labels = np.full((batch_size, max_len), fill_value=0.0, dtype=np.int32)
        for e, l in enumerate(labels):
            cat_labels[e, : len(l)] = l
        labels = torch.tensor(cat_labels, dtype=torch.int64, device=device)

        return labels

    def batch_select_state(self, batch_states, idx):
        if batch_states is not None:
            state_list = []
            for state_id in range(len(batch_states)):
                states = [batch_states[state_id][layer][idx] for layer in range(self.PRED_RNN_LAYERS)]
                state_list.append(states)
            return state_list
        else:
            return None

    def batch_concat_states(self, batch_states):
        state_list = []
        for state_id in range(len(batch_states[0])):
            batch_list = []
            for sample_id in range(len(batch_states)):
                tensor = torch.stack(batch_states[sample_id][state_id])  # [L, H]
                tensor = tensor.unsqueeze(0)  # [1, L, H]
                batch_list.append(tensor)

            state_tensor = torch.cat(batch_list, 0)  # [B, L, H]
            state_tensor = state_tensor.transpose(1, 0)  # [L, B, H]
            state_list.append(state_tensor)

        return state_list

    def tokens_to_text(self,prediction):
        """Translate a sequence of predicted tokens to text
        Args:
            prediction: sequence of tokens
        Returns:
            A string containing the translated text
        """
        prediction = [p for p in prediction if p != self.BLANK_INDEX]
        # De-tokenize the integer tokens
        text = ""
        for token in prediction:
            text += self.vocabulary[token]
        return text.replace("▁"," ")

    def decoder_step(self, label,hidden):
        """Perform a single decoding step
        Args:
            label: previous predicted token 
            hidden: hidden state after the last decoding step
        Returns:
            decoder outut and new hidden state
        """     
        if isinstance(label, torch.Tensor):
            if label.dtype != torch.long:
                label = label.long()
        else:
            if label == self.SOS:
                # Last token was start or stopp token, call decoder with empty target
                return self.decoder.predict(None, hidden, add_sos=False, batch_size=None)
            label = self.label_collate([[label]])
        # call decoder conditioned on the previous predicted label
        return self.decoder.predict(label, hidden, add_sos=False, batch_size=None)

    def joint_step(self, enc, pred):
        """Perform a single joint step
        Args:
            enc: encoded audio signal
            pred: decoder output
        Returns:
            probabilities over the tokens of the vocabulary
        """  
        with torch.no_grad():
            logits = self.joint.joint(enc, pred)
        return logits
    
    def greedy_RNNT_decode(self, encoder_output, encoded_lengths, y_sequence = [], dec_state=None, last_token = None):
        """Perform decoder and joint step for every encoded signal in the given audio chunk
        Args:
            encoder_output: encoded audio chunk containing multiple encodings
            encoded_lengths: number of encodings in the chunk 
            y_sequence: current sequence of transcribed tokens
            dec_state: previous decoder hidden state 
            last_token: previous predicted token
        Returns:
            updated sequence of transcribed tokens
            updated decoder hidden state
            updated last predicted token
        """ 
        encoder_output = encoder_output.transpose(1, 2) 
        encoder_output = encoder_output[0, :, :].unsqueeze(1)
        encoded_lengths = encoded_lengths[0]
        y_sequence = (y_sequence.cpu().tolist() if isinstance(y_sequence, torch.Tensor) else y_sequence)
        if dec_state is not None:
            dec_state = self.batch_concat_states([dec_state])
            dec_state = self.states_to_device(dec_state, encoder_output.device)

        # For timestep t in X_t
        for time_idx in range(encoded_lengths):
            # Extract encoder embedding at timestep t
            encoder_output_t = encoder_output.narrow(dim=0, start=time_idx, length=1)

            # Setup exit flags and counter
            not_blank = True
            symbols_added = 0

            # While blank is not predicted, or we dont run out of max symbols per timestep
            while not_blank and (self.MAX_SYMBOLS is None or symbols_added < self.MAX_SYMBOLS):
                # In the first timestep, we initialize the network with RNNT Blank
                # In later timesteps, we provide previous predicted label as input.
                if last_token is None and dec_state is None:
                    last_label = self.SOS
                else:
                    last_label = self.label_collate([[last_token]])
            
                # Decoder + Joint Step
                dec_output, hidden_prime = self.decoder_step(last_label, dec_state)
                logp = self.joint_step(encoder_output_t, dec_output)[0, 0, 0, :]
                del dec_output

                # torch.max(0) op doesnt exist for FP 16.
                if logp.dtype != torch.float32:
                    logp = logp.float()

                # get index k, of max prob
                v, k = logp.max(0)
                k = k.item()  # K is the label at timestep t_s in inner loop, s >= 0.
                del logp

                # If blank token is predicted, exit inner loop, move onto next timestep t
                if k == self.BLANK_INDEX:
                    not_blank = False
                else:
                    y_sequence.append(k)
                    dec_state = hidden_prime
                    last_token = k

                # Increment token counter.
                symbols_added += 1

        # prpare outputs and decoder state for next step
        y_sequence = ( y_sequence.to(torch.long) if isinstance(y_sequence, torch.Tensor) else torch.tensor(y_sequence, dtype=torch.long))
        dec_state = self.batch_select_state(dec_state, 0)
        dec_state = self.states_to_device(dec_state)

        return y_sequence, dec_state, last_token
    
class LLM(torch.nn.Module):
    MIN_LENGTH = 10 # minimum number of generated tokens before EOS token can be predicted 

    def __init__(self, model_name, device="cuda"):
        """Initialized tokenizer and LLM model from huggingface
        Args:
            model_name: huggingface model descriptor
        Returns:
            Instnatce of the LLM model 
        """ 
        super(LLM, self).__init__()

        self.device = device

        # Initialize given model. Weights are downloaded from HF hub and cached
        self.model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, trust_remote_code=True).to(self.device)
        self.model.eval()

        # Initialize tokenizer for given model. Weights are downloaded from HF hub and cached
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Specify special tokens for inference
        self.pad_token_id =  self.model.generation_config.eos_token_id
        self.eos_token_id =  [self.model.generation_config.eos_token_id]
        self.eos_token_id_tensor = torch.tensor(self.eos_token_id).to(self.device)
        self.sentence_stop_token_id_tensor = torch.tensor([13]).to(self.device)

    def tokenize(self, text):
        """Resolve the given text into a sequence of tokens
        Args:
            text: string of arbitrary length
        Returns:
            Sequence of tokens
        """ 
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
    
    def detokenize(self, tokens):
        """Resolve the given sequence of tokens to a text string
        Args:
            tokens: sequence of tokens
        Returns:
            string containing the resolved text
        """ 
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def forward(self, input_token, past_key_values, cur_len):
        """Perform a single LLM inference step  
        Args:
            input_token: either input token from user or previous generated token from LLM
            past_key_values: key-value cache from LLM
            cur_len: length of the current token sequence that was generated by the LLM 
        Returns:
            next token and updated key-value cache
        """ 
        unfinished_sequences = torch.ones([1], dtype=torch.long, device=self.device)
        # prepare model inputs and set key-value chache
        model_inputs = self.model.prepare_inputs_for_generation(input_token, past_key_values=past_key_values, use_cache=True)
        
        # forward pass to get logits and updated key value cache
        outputs =self.model(**model_inputs, return_dict=True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

        # Prevent prediction of EOS token if min lenght of sequence is not reached
        if cur_len < self.MIN_LENGTH: 
            for i in self.eos_token_id:
                next_token_logits[:, i] = -float("inf")
        
        # Select token with highes probability
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        next_tokens = next_tokens * unfinished_sequences + self.pad_token_id * (1 - unfinished_sequences)
        return next_tokens, past_key_values

class TTS(torch.nn.Module):
     def __init__(self, device="cuda"):
        """Initialize STT model and perform warm up steps
        Args: 
        Returns:
            Instance of TTS model
        """
        super(TTS, self).__init__()

        # init Style TTS model 
        self.tts_model = StyleTTS2Model(device=device)
        
        # perform multiple inference steps for warmup
        for _ in range(3):
            start = time.time()
            self.forward("warming up!")
            if VERBOSE:
                print("tts warm up: ", round(time.time() - start,3))

     def forward(self, text):
         """Syntehsize the entire text
        Args: 
            text: String of text that should be synthesized
        Returns:
            synthesized audio data in wave format
        """
         wav = self.tts_model(text)
         return wav 
 
class STT_LLM_TTS(torch.nn.Module):
    
    THRESHOLD_VOICE_DETECTION  = -30000
    
    def __init__(self, device):
        """Initialize STT, LLM and TTS model and the state of the voice assistant 
        Args: 
        Returns:
            Instance of voice assistant model
        """
        super(STT_LLM_TTS, self).__init__()

        # Init STT model 
        self.stt = STT(vocabulary_path="vocab.npy")
        
        # Init LLM model 
        self.llm = LLM(model_name="microsoft/phi-2", device=device)

        # Init TTS model
        self.tts = TTS(device=device)

        # Init voice assistant state
        self.transcribed_tokens = []
        self.transcribed_words = []
        self.current_word = ""
        self.first = True
        self.last_token_timestep = None
        self.start_generation_timestep = None
        self.past_key_values = None
        self.past_key_values_backup = None
        self.last_token = None
        self.generating = False
        self.response_sequence = []
        self.response_sentence = []
        self.transcribing = True 
        self.last_return = None
        self.no_speech_count = 0

        # Tracking latencies
        self.INIT_TIME = time.time()
        self.times_stt = []
        self.times_llm = []
        self.times_start = []
        self.times_interrupt = []
        self.times_tts = []

    def call_LLM(self, input, reason, tokenize=True, ignore_output=False):
        """
        Pass the given input to the LLM to either generate a new output or just update the key value cache
        If the input is a string, the tokenizer could output multiple tokens which will lead to multiple
        forward paths of the LLM in one single function call. 
        Args:
            input: A string or a single token
            reason: A description of why the LLM was called (Input, Generating, Format, Warm up). Just for tracking latencies
            tokenize: If true, the input is treated as a string and first tokenized to a sequence of tokens
            ignore output: If yes, the key-value cache is not update (used for warm up)
        """ 
        if tokenize:
            # input is a string and have to be tokenized first
            tokens = self.llm.tokenize(input)
            # if string is tokenized into several tokens, all of them are passed to the LLM sequentially
            for t in tokens[0]:
                t_start = time.time()
                t = torch.unsqueeze(torch.unsqueeze(t, 0),0)
                # LLM forward path
                last_token, past_key_values = self.llm(t, self.past_key_values, len(self.response_sequence))
                t_end = time.time()
                if VERBOSE:
                    print(" -LLM ("+reason+")", round(t_end-t_start,3))
                self.times_llm.append({"start":t_start-self.INIT_TIME, "end":t_end-self.INIT_TIME, "reason":reason})

                if ignore_output == False:
                    self.last_token = last_token
                    self.past_key_values = past_key_values
        else:
            # input is a token and is directly passed to the LLM.
            t_start = time.time()
            # LLM forward path
            last_token, past_key_values = self.llm(input, self.past_key_values, len(self.response_sequence))
            t_end = time.time()
            if VERBOSE:
                print(" -LLM ("+reason+")", round(t_end-t_start,3))
            self.times_llm.append({"start":t_start-self.INIT_TIME, "end":t_end-self.INIT_TIME, "reason":reason})

            if ignore_output == False:
                self.last_token = last_token
                self.past_key_values = past_key_values
        
    def call_STT(self, processed_signal, processed_signal_length):
        """Pass the given audio signal to the STT to get a transcription
        Args:
            processed_signal: preprocessed audio chunk
            processed_signal_length: number of timesteps in this chunk
        Return:
            The transcribed text (Could be an empty string if no speech was detected)
        """ 
        t_start = time.time()

        # Call STT model to transcribe the given audio chunk
        y_sequence = self.stt(processed_signal, processed_signal_length)
        y_sequence = list(y_sequence.cpu().numpy())

        # Select new tokens and add them to the current sequence. new_tokens can alls be empty
        # if no word or subword could be detected in the chunk 
        new_tokens = y_sequence[len(self.transcribed_tokens):]
        self.transcribed_tokens += new_tokens

        # reslve new token to text
        transcribed_text = self.stt.tokens_to_text(new_tokens)
        t_end = time.time()
        if VERBOSE:
            print(" -STT: ",transcribed_text, round(t_end-t_start,3))
        speech_detected = not transcribed_text==""
        self.times_stt.append({"start":t_start-self.INIT_TIME, "end":t_end-self.INIT_TIME, "transcribed":speech_detected})

        return transcribed_text

    def handle_transcription(self, transcribed_text):
        """Handle new transcription and decide if it should be passed to the LLM
        Args:
            transcribed_text: Transcription output from the STT. Could be a sequence of words, a single word,
                                a sub-word or an empty string
        """ 
        if len(transcribed_text)>0:
            # reset timer for last recognized word / subword
            self.last_token_timestep = time.time()
            print(transcribed_text)

        # First STT run is very slow, show Start prompt afterwards and add format tokens to LLM
        if self.first:
            self.first = False
            print("\n\n ## Listening... ")

            # Add format token
            self.call_LLM(input="\nInstruct:", reason="format", tokenize=True)

        if (len(transcribed_text) == 0 or transcribed_text.startswith(" ")) and len(self.current_word)>0:
            # --> new word
            new_word = self.current_word
            self.transcribed_words.append(new_word)
            self.current_word = transcribed_text
            self.counter = 0

            # call LLM to process new input token
            self.call_LLM(input=new_word, reason="input", tokenize=True)
        
        # if new token is just a subword, add it to the current word
        else:
            self.current_word += transcribed_text

    def reset_sentence(self):
        """Resets token buffer after end of sentence
        """ 
        # only reset current sentence not the entire response sequence
        self.response_sentence = []
        # reset timer 
        self.start_generation_timestep = time.time()

    def reset_sequence(self):
        """Resets current sentence and current sequence after end of sequence.
        Helper variables are set to start a new sequence when the next word is transcribed 
        """ 
        self.first = True
        self.generating = False
        self.current_word = ""
        self.transcribing = True
        self.last_token_timestep = None
        # reset sentence and sequenc 
        self.response_sequence = []
        self.response_sentence = []
        self.start_generation_timestep = None

    def handle_stop_conditions(self):
        """Check stopping conditions and decide if the current sentence or sequence should be returned
        Returns:
            end: Boolean value if sentence or sequence ended
            response: Finished generated text by the LLM or None if end is False
            wav: Synthesized speech from the TTS or None if end is False
        """ 
        # Check if eos token or a fullstop was predicted
        end_of_sentence = self.last_token.eq(self.llm.sentence_stop_token_id_tensor)
        end_of_sequence = self.last_token.eq(self.llm.eos_token_id_tensor)
        end = False
        wav = None
        response = None

        if end_of_sentence or len(self.response_sentence)>=50:
            # --> End of Sentence
            if time.time() - self.last_token_timestep > 0.3:
                end = True
                response = self.llm.detokenize(self.response_sentence)
                response = "".join(response)

                # synthesize generated sequence
                t_start_tts = time.time()
                wav = self.tts(response)
                self.times_tts.append({"start":t_start_tts-self.INIT_TIME, "end":time.time()-self.INIT_TIME})

                # calculate and print latency
                latency = time.time()-self.start_generation_timestep
                print("## Total Latency: ", round(latency,3))

                # Save recorded latencies for files for speed evaluation
                with open('latencies/times_stt', 'w') as fout:
                    json.dump(self.times_stt, fout)
                with open('latencies/times_llm', 'w') as fout:
                    json.dump(self.times_llm, fout)
                with open('latencies/times_start', 'w') as fout:
                    json.dump(self.times_start, fout)
                with open('latencies/times_interrupt', 'w') as fout:
                    json.dump(self.times_interrupt, fout)
                with open('latencies/times_tts', 'w') as fout:
                    json.dump(self.times_tts, fout)

                self.reset_sentence()
                
        if end_of_sequence:
            # --> End of Sequence  

            # This if statement prevents the assistant to return a sequence to early when the user is
            # still speaking TODO find a better solution to deal with early eos tokens
            if time.time() - self.last_token_timestep > 0.3:
                # reset state of the voice assistnat (but keep key-value cache for context)
                
                # detokenize current sentence 
                response = self.llm.detokenize(self.response_sentence)
                response = "".join(response)

                # Sometimes the previous sentence was already the end of the sequence but the EOS
                # token is generated after the end of sentence token. If the sequence ends but the current
                # sentence is too short, nothing is returned and no speech is synthesized
                if len(response) > 3:
                    end = True
                    # synthesize generated sequence
                    wav = self.tts(response)

                    # calculate and print latency
                    latency = time.time()-self.start_generation_timestep
                    print("## Total Latency: ", round(latency,3))
                    
                self.reset_sequence()

        return end, response, wav
  
    def forward(self, processed_signal, processed_signal_length):
        """Perform a single voice assistant forward path
           1) If a processed signal is passed and speech is detected, the audio chunk is transcribed
           and the transcribed tokens are passed to the LLM to update its key-value chache
           2) If processed signal is None (buffer did not collected enough bytes of audio signal) or no speech is detected
           Only LLM is called to generate the next token 
        Args: 
            processed_signal: preprossed audio chunk 
            processed_signal_length: Number of timesteps in the chunk 
        Returns:
            If end of sequence of end of sentence token was generated, return generated text, syntehsized
            audio and flag if an interrupt was detected
            else: return None
        """
        with torch.inference_mode():
            with torch.no_grad(): 
                transcribed_text = ""
                if processed_signal is not None:
                    ## Detect voice to activate transcription
                    signal_strength = torch.sum(processed_signal).item()
                    if signal_strength > self.THRESHOLD_VOICE_DETECTION:
                        self.transcribing = True
                        self.no_speech_count = 0
                    else:
                        self.no_speech_count += 1

                    ## Speech to text
                    if  self.transcribing:
                        transcribed_text = self.call_STT(processed_signal, processed_signal_length)

                # Handle Interrupt when generating but user continues speaking
                if self.generating and len(transcribed_text)>0:
                    # User speaks while sentence is generated 
                    # Decide if its part of the current query or the user wants to interrupt and ask something else
                    if time.time() - self.last_token_timestep <= 0.8:
                        # --> Same query 
                        self.generating = False
                        self.last_token_timestep = time.time()
                        # reset key value chache to the beginning of generation
                        self.past_key_values = deepcopy(self.past_key_values_backup)
                        # reset generated sentence and whole generated sequence
                        self.response_sequence = [] 
                        self.response_sentence = []
                        self.times_interrupt.append({"start":time.time()-self.INIT_TIME})
                    else:
                        # --> Interrupt
                        # reset state of voice assistant entirely TODO: find better solution
                        self.first = True
                        self.past_key_values = None 
                        self.generating = False
                        self.response_sentence = []
                        self.response_sequence = []
                        self.current_word = ""
                        self.transcribing = True
                        self.last_token_timestep = None
                        self.start_generation_timestep = None
                        self.times_interrupt.append({"start":time.time()-self.INIT_TIME})
                        return None, None, True
                
                # Handle token generation when no new word was detected
                if self.last_token_timestep is not None and not self.first and len(transcribed_text)==0:
                    # stop transcription after certain time until LLM generation is finished
                    # TODO disable STT right at the beginning and handle interrupts by amplitude in audio input
                    if self.transcribing and self.no_speech_count>1:
                        if VERBOSE:
                            print("[Stop transcribing!]")
                        self.transcribing = False

                    # Start token generation
                    if not self.generating:
                        self.generating = True
                        # backup key-value chache to restore it in case on an interrupt
                        self.past_key_values_backup = deepcopy(self.past_key_values)
                        self.start_generation_timestep = time.time()
                        self.times_start.append({"start":self.start_generation_timestep-self.INIT_TIME})
                        if VERBOSE:
                            print("\n\n[START] ", round(time.time()-self.last_token_timestep,3))

                        # pass the remaining words / subwords of the input sequence to the LLM
                        if self.current_word != "":
                            self.call_LLM(input=self.current_word, reason="input", tokenize=True)
                        
                        # Add format tokens 
                        # TODO calculate key value cache earlier and use it here to save inference time
                        self.call_LLM(input="\nOutput:", reason="format", tokenize=True)
                        self.response_sequence.append(self.last_token)
                        self.response_sentence.append(self.last_token)

                # LLM generates a new token and adds it to the response sequence
                if self.generating:
                    self.last_token = torch.unsqueeze(self.last_token, 0)
                    self.call_LLM(input=self.last_token, reason="generating", tokenize=False)
                    self.response_sequence.append(self.last_token)
                    self.response_sentence.append(self.last_token)

                    # check for stopping conditions
                    end, response, wav = self.handle_stop_conditions()
                    if end:
                        return response, wav, False
                else:
                    ## keep LLM "warm". This avoids slow inference after LLM was idle for a longer time
                    self.call_LLM(input=".", reason="warming_up", tokenize=True, ignore_output=True)

                    # --> not in generation mode handle new transcribed token 
                    if processed_signal is not None:
                        self.handle_transcription(transcribed_text)
                
                # If generated sentence/sequence is not finished --> Return none
                self.last_return = time.time()
                return None, None, False
            