import json

import pytest

from neuralese.audit import certify
from neuralese.cli import main
from neuralese.contracts import Observation, Receipt, UncertifiedPackError
from neuralese.translator import translate_stream

from packutil import make_pack
