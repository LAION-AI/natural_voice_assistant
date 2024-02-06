# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

"""
Streaming Buffer to store preprocessed audio chunks and prepare them as input to the Streaming STT model 
"""

class StreamBuffer():
    PRE_ENCODED_CACHE_SIZE = 9
    INPUT_FEATURES = 80
    SAMPLING_FRAMES = [1, 8] #sampling frames from model?

    def __init__(self, chunk_size=16,shift_size=16, device="cuda"):
        self.buffer = None
        self.buffer_idx = 0
        self.streams_length = None
        self.chunk_size = chunk_size
        self.shift_size = shift_size
        self.device = device

    def __iter__(self):
        while True:
            if self.buffer_idx >= self.buffer.size(-1):
                return
            audio_chunk = self.buffer[:, :, self.buffer_idx : self.buffer_idx + self.chunk_size]

            # checking to make sure the audio chunk has enough frames to produce at least one output after downsampling
            if self.buffer_idx == 0:
                cur_sampling_frames = self.SAMPLING_FRAMES[0]
            else:
                cur_sampling_frames = self.SAMPLING_FRAMES[1] 
            if audio_chunk.size(-1) < cur_sampling_frames:
                return

            # Adding the cache needed for the pre-encoder part of the model to the chunk
            # if there is not enough frames to be used as the pre-encoding cache, zeros would be added
            zeros_pads = None
            if self.buffer_idx == 0:
                cache_pre_encode_num_frames = self.PRE_ENCODED_CACHE_SIZE
                cache_pre_encode = torch.zeros((audio_chunk.size(0), self.INPUT_FEATURES, cache_pre_encode_num_frames),device=audio_chunk.device,dtype=audio_chunk.dtype,)
            else:
                start_pre_encode_cache = self.buffer_idx - self.PRE_ENCODED_CACHE_SIZE
                if start_pre_encode_cache < 0:
                    start_pre_encode_cache = 0
                cache_pre_encode = self.buffer[:, :, start_pre_encode_cache : self.buffer_idx]
                if cache_pre_encode.size(-1) < self.PRE_ENCODED_CACHE_SIZE:
                    zeros_pads = torch.zeros((audio_chunk.size(0),audio_chunk.size(-2),self.PRE_ENCODED_CACHE_SIZE - cache_pre_encode.size(-1),),device=audio_chunk.device,dtype=audio_chunk.dtype,)
            added_len = cache_pre_encode.size(-1)
            audio_chunk = torch.cat((cache_pre_encode, audio_chunk), dim=-1)

            if zeros_pads is not None:
                # TODO: check here when zero_pads is not None and added_len is already non-zero
                audio_chunk = torch.cat((zeros_pads, audio_chunk), dim=-1)
                added_len += zeros_pads.size(-1)

            max_chunk_lengths = self.streams_length - self.buffer_idx
            max_chunk_lengths = max_chunk_lengths + added_len
            chunk_lengths = torch.clamp(max_chunk_lengths, min=0, max=audio_chunk.size(-1))

            self.buffer_idx += self.shift_size
            yield audio_chunk, chunk_lengths

    def __len__(self):
        return len(self.buffer)

    def is_buffer_empty(self):
        if self.buffer_idx >= self.buffer.size(-1):
            return True
        else:
            return False

    def append_processed_signal(self, processed_signal, stream_id=-1):
        processed_signal_length = torch.tensor(processed_signal.size(-1), device=self.device)
        if self.buffer is None:
            self.buffer = processed_signal
            self.streams_length = torch.tensor([processed_signal_length], device=self.device)
        else:
            if stream_id < 0:
                self.buffer = torch.nn.functional.pad(self.buffer, pad=(0, 0, 0, 0, 0, 1))
                self.streams_length = torch.cat(
                    (self.streams_length, torch.tensor([0], device=self.device)), dim=-1
                )
                stream_id = len(self.streams_length) - 1
            needed_len = self.streams_length[stream_id] + processed_signal_length
            if needed_len > self.buffer.size(-1):
                self.buffer = torch.nn.functional.pad(self.buffer, pad=(0, needed_len - self.buffer.size(-1)))

            self.buffer[
                stream_id, :, self.streams_length[stream_id] : self.streams_length[stream_id] + processed_signal_length
            ] = processed_signal
            self.streams_length[stream_id] = self.streams_length[stream_id] + processed_signal.size(-1)
    
    def reset_buffer(self):
        self.buffer = None
        self.buffer_idx = 0
        self.streams_length = None