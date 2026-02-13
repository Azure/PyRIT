# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid

from pyrit.executor.attack.single_turn.beam_search import Beam, TopKBeamReviewer


class TestTopKBeamReviewer:
    def test_review_k2d0(self):
        beam1 = Beam(id=str(uuid.uuid4()), text="beam1", score=0.9)
        beam2 = Beam(id=str(uuid.uuid4()), text="beam2", score=0.8)
        beam3 = Beam(id=str(uuid.uuid4()), text="beam3", score=0.7)
        beams = [beam1, beam2, beam3]
        reviewer = TopKBeamReviewer(k=2, drop_chars=0)
        top_k_beams = reviewer.review(beams)
        assert len(top_k_beams) == 3
        assert top_k_beams[0].text == "beam1"
        assert top_k_beams[1].text == "beam2"
        assert top_k_beams[2].text == "beam1"

    def test_review_k2d1(self):
        beam1 = Beam(id=str(uuid.uuid4()), text="beam11", score=0.9)
        beam2 = Beam(id=str(uuid.uuid4()), text="beam22", score=0.8)
        beam3 = Beam(id=str(uuid.uuid4()), text="beam33", score=0.7)
        beam4 = Beam(id=str(uuid.uuid4()), text="beam44", score=0.6)
        beam5 = Beam(id=str(uuid.uuid4()), text="beam55", score=0.5)
        beams = [beam1, beam2, beam3, beam4, beam5]
        reviewer = TopKBeamReviewer(k=2, drop_chars=1)
        top_k_beams = reviewer.review(beams)
        assert len(top_k_beams) == 5
        assert top_k_beams[0].text == "beam11"
        assert top_k_beams[1].text == "beam22"
        assert top_k_beams[2].text == "beam1"
        assert top_k_beams[3].text == "beam2"
        assert top_k_beams[4].text == "beam1"
