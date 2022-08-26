import uuid

from nlubridge.nlu_dataset import NluDataset


class ToyDataset(NluDataset):
    def __init__(self, **kw_args):
        """Create a NLUdataset with two samples and intent and entity annotations."""
        texts = [
            "Book me a flight from Cairo to Redmond next Thursday",
            "What's the weather like in Seattle?",
        ]
        intents = ["BookFlight", "GetWeather"]
        entities = [
            [
                {"entity": "LocationFrom", "start": 22, "end": 27},
                {"entity": "LocationTo", "start": 31, "end": 38},
            ],
            [{"entity": "Location", "start": 27, "end": 34}],
        ]
        super().__init__(texts, intents, entities, **kw_args)


class SyntheticDataset(NluDataset):
    def __init__(self, n_samples, intents, **kw_args):
        """
        Create an NLUdataset with n_samples samples.

        Utterances are random character sequences and intents from the passed intents
        are assigned to them evenly distributed.
        """
        texts = [str(uuid.uuid1()) for _ in range(n_samples)]
        intents = intents * (round(n_samples / len(intents)) + 1)
        intents = intents[:n_samples]
        super().__init__(texts, intents, None, **kw_args)


class TrainingDataset(NluDataset):
    def __init__(self):
        """Create a NLUdataset for testing with two intents and no entities."""
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
