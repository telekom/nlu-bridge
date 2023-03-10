<h1 align="center">
    Telekom NLU Bridge
</h1>

<p align="center">
    <a href="https://github.com/telekom/nlu-bridge/commits" title="Last Commit"><img src="https://img.shields.io/github/last-commit/telekom/nlu-bridge?style=flat"></a>
    <a href="https://github.com/telekom/nlu-bridge/issues" title="Open Issues"><img src="https://img.shields.io/github/issues/telekom/nlu-bridge?style=flat"></a>
    <a href="https://github.com/telekom/nlu-bridge/blob/main/LICENSE" title="License"><img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat"></a>
</p>

<p align="center">
  <a href="#development">Development</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#support-and-feedback">Support</a> •
  <a href="#how-to-contribute">Contribute</a> •
  <a href="#licensing">Licensing</a>
</p>

The goal of this project is to provide a unified API to several popular intent recognition
applications.

## About this component

### Installation

The core package including NLUdataset and Baseline vendors can be installed for Python\<=3.8
using pip

```
pip install nlubridge
```

To include optional dependencies for the vendors, e.g. Watson Assistant, type

```
pip install nlubridge[watson]
```

Following install options are available:

- `watson`
- `fasttext`
- `luis`
- `rasa2`
- `rasa3`
- `spacy`
- `huggingface`

Development tools can be installed with option `develop`.

Some vendors require access credentials like API tokens, URLs etc. These can be passed
on construction of the objects. Alternatively, such arguments can be passed as
environment variables, where the vendor will look for variables named variable
VENDORNAME_PARAM_NAME.

Some vendors require additional dependencies. E.g., Spacy requires a model that
can be downloaded (for the  model de_core_news_sm) with

```
python -m spacy download de_core_news_sm
```

### Migration from v0

With realease 1.0.0 we introduce a couple of changes to the names of files and vendor
classes(see also https://github.com/telekom/nlu-bridge/issues/18).

Most notably:

- datasets.NLUdataset -> nlu_dataset.NluDataset
- vendors.vendors.Vendor -> - vendors.vendor.Vendor
- new supackage `dataloaders` that holds all functions for loading data into an NluDataset
- new function `nlu_dataset.concat` to concatenate NluDatasets passed in a list
- can load dataloaders, NluDataset, Vendor, OUT_OF_SCOPE_TOKEN, EntityKeys, concat,
  directly from nlubridge like `from nlubridge import Vendor`
- Load vendors like `from nlubridge.vendors import Rasa3`
- former `TelekomModel` now called `CharNgramIntentClassifier`
- Some vendor names changed for clarity and consistency (see "List of supported vendors"
  for the new names)

### Usage

Here is an example for the TfidfIntentClassifier:

```python
import os

import pandas as pd

from nlubridge.vendors import TfidfIntentClassifier
from nlubridge import NluDataset

dataset = NluDataset(texts, intents)
dataset = dataset.shuffle()

classifier = TfidfIntentClassifier()

train, test = dataset.train_test_split(test_size=0.25, random_state=0)
classifier = classifier.train_intent(train)
predicted = classifier.test_intent(test)
res = pd.DataFrame(list(zip(test.intents, predicted)), columns=['true', 'predicted'])
```

If you need to configure **stratification**, use the `stratification` parameter (defaults to `"intents"` and uses the intents in the dataset as stratification basis; whatever _else_ you pass along has to conform to `sklearn.model_selection.train_test_split(stratify=)`:

```python
train, test = dataset.train_test_split(test_size=0.25, random_state=0, stratification=None)    # deactivate stratification (sklearn default for train_test_split)
```

To compare your own vendor or algorithm to existing vendors in this package, you can
write a Vendor Subclass for your vendor, and possibly a dataloader function. Feel free
to share your implementation using this repo. Similarly, fixes and extensions for the
existing vendors are always welcome.

### Logging

Most of the code uses python logging to report its progress. To get logs printed out
to console or Jupyter notebook, a logger needs to be configured, before the nlutests
code. Usually, log messages are on INFO level. This can be configured like this:

```python
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
```

### Concepts / Architecture

- **Vendors**\
  The [`vendors`](/nlubridge/vendors/) subpackage implements standardized interfaces to
  the specific vendors. A specific `Vendor` instance is in charge of dealing with
  converting the data to the required format, uploading data to the cloud if applicable,
  training models and making predictions.

- **Datasets**\
  The [`nlu_dataset`](/nlubridge/nlu_dataset/) module provides a standard interface to
  NLU data. Data stored in different vendor's custom format can be loaded as a dataset
  and provided to any different vendor.

- **Data Loaders**\
  The [`dataloaders`](/nlubridge/dataloaders/) subpackage provides functions to load
  data that are in a vendor-specific format as NluDataset.

### List of supported vendors

| Vendor Class | Status | Intents | Entities | Algorithm |
| ------ | ------ | ------- | -------- | --------- |
| [TfidfIntentClassifier](/nlubridge/vendors/tfidf_intent_classifier.py) |  ✓  | ✓ | ✗ |  TFIDF on words + SVM |
| [FastText](https://fasttext.cc) |  ✓  | ✓ | ✗ |  fasttext |
| [Spacy](https://spacy.io/usage/training#section-textcat) | ✓ | ✓ | ✗ | BoW linear + CNN |
| [WatsonAssistant](https://www.ibm.com/watson/services/conversation/) | ✓  | ✓ | ✗ | Propietary (probably LR) |
| [Luis](https://www.luis.ai/home) | needs testing | ✓ | ✗ | Propietary (probably LR) |
| [CharNgramIntentClassifier](/nlubridge/vendors/char_ngram_intent_classifier.py)  | ✓ | ✓ | ✗ | tf-idf on char n-grams + SGD |
| [Rasa2](https://github.com/RasaHQ/rasa) | ✓ | ✓ | ✓ |  configurable |
| [Rasa3](https://github.com/RasaHQ/rasa) | ✓ | ✓ | ✓ |  configurable |

### Features

- Abstract class for Vendors with convenience methods (ex: scoring and scikit-learn compatibility)
- Abstract class for datasets with convenience methods (ex: train_test_split, indexing, iteration)
- Rate limiting to comply with cloud providers requirements

## Development

_TBD_

### Build

_TBD_

## Code of Conduct

This project has adopted the [Contributor Covenant](https://www.contributor-covenant.org/) in version 2.0 as our code of conduct. Please see the details in our [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). All contributors must abide by the code of conduct.

## Working Language

We decided to apply _English_ as the primary project language.

Consequently, all content will be made available primarily in English. We also ask all interested people to use English as language to create issues, in their code (comments, documentation etc.) and when you send requests to us. The application itself and all end-user facing content will be made available in other languages as needed.

## Documentation

The full documentation for the telekom nlu-bridge can be found in _TBD_

## Support and Feedback

The following channels are available for discussions, feedback, and support requests:

| Type                     | Channel                                                |
| ------------------------ | ------------------------------------------------------ |
| **Issues**   | <a href="/../../issues/new/choose" title="General Discussion"><img src="https://img.shields.io/github/issues/telekom/nlu-bridge?style=flat-square"></a> </a>   |
| **Other Requests**    | <a href="mailto:opensource@telekom.de" title="Email Open Source Team"><img src="https://img.shields.io/badge/email-Open%20Source%20Team-green?logo=mail.ru&style=flat-square&logoColor=white"></a>   |

## How to Contribute

Contribution and feedback is encouraged and always welcome. For more information about how to contribute, the project structure, as well as additional contribution information, see our [Contribution Guidelines](./CONTRIBUTING.md). By participating in this project, you agree to abide by its [Code of Conduct](./CODE_OF_CONDUCT.md) at all times.

## Licensing

Copyright (c) 2021 Deutsche Telekom AG.

Licensed under the **MIT License** (the "License"); you may not use this file except in compliance with the License.

You may obtain a copy of the License by reviewing the file [LICENSE](./LICENSE) in the repository.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the [LICENSE](./LICENSE) for the specific language governing permissions and limitations under the License.
