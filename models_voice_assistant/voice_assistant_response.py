"""
A simple wrapper class to handle responses from the Voice Assistant model
"""

class VoiceAssistantResponse():
    def __init__(self, id=0, input="", response="", speech=None, finished=False) -> None:
        self.id = id
        self.input = input
        self.response = response
        self.speech = speech
        self.finished = finished
        self.interrupt = False
    