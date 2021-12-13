import uuid

from nlubridge.datasets import NLUdataset


class ToyDataset(NLUdataset):
    def __init__(self, **kw_args):
        texts = [
            "Book me a flight from Cairo to Redmond next Thursday",
            "What's the weather like in Seattle?",
        ]
        intents = ["BookFlight", "GetWeather"]
        entities = [
            [
                {"entity": "LocationFrom", "start": 22, "end": 26},
                {"entity": "LocationTo", "start": 31, "end": 37},
            ],
            [{"entity": "Location", "start": 27, "end": 33}],
        ]
        super().__init__(texts, intents, entities, **kw_args)


class SyntheticDataset(NLUdataset):
    def __init__(self, n_samples, intents, **kw_args):
        texts = [str(uuid.uuid1()) for _ in range(n_samples)]
        intents = intents * (round(n_samples / len(intents)) + 1)
        intents = intents[:n_samples]
        super().__init__(texts, intents, None, **kw_args)


class TrainingDataset(NLUdataset):
    def __init__(self):
        help = [
            "I need help",
            "help me",
            "please help me",
            "what can i do?",
            "how does this work?",
            "help, i need somebody",
            "hello can i get help",
            "i really need help",
            "i dont know how this works",
            "how is this supposed to be used",
            "can someone help me",
            "how do i do this",
            "show me how this works please",
            "i need a manual",
            "can i read the manual",
            "instructions please",
            "i need better instructions",
            "i have trouble using this",
            "i cannot use this",
            "I wish i knew how this works" "",
        ]
        affirm = [
            "say your name please",
            "what's your name",
            "how can i call you",
            "how may i dub you",
            "can you tell me your name",
            "who are you",
            "please tell me your name",
            "i wish i knew your name",
            "how do you call yourself",
            "how do people call you",
        ]
        intents = ["help"] * 20 + ["affirm"] * 10
        texts = help + affirm
        super().__init__(texts, intents, None)
