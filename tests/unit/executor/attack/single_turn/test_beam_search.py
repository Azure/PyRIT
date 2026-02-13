# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid

import pytest

from pyrit.executor.attack.single_turn.beam_search import Beam, TopKBeamReviewer


class TestBeam:
    @pytest.mark.parametrize("n_extend", [1, 2, 4])
    def test_grammar_smoke(self, n_extend):
        beam = Beam(id=str(uuid.uuid4()), text="beam1", score=0.9)

        expected_grammar = f"""
start: PREFIX CONTINUATION
PREFIX: "beam1"
CONTINUATION: /.{{0,{n_extend}}}/
"""
        assert beam.get_grammar(n_chars=n_extend) == expected_grammar

    def test_grammar_with_newline(self):
        beam = Beam(id=str(uuid.uuid4()), text="beam1\nbeam2", score=0.9)

        expected_grammar = """
start: PREFIX CONTINUATION
PREFIX: "beam1\\nbeam2"
CONTINUATION: /.{0,1}/
"""
        assert beam.get_grammar(n_chars=1) == expected_grammar


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

    def test_review_k1d2(self):
        beam1 = Beam(id=str(uuid.uuid4()), text="beam111", score=0.9)
        beam2 = Beam(id=str(uuid.uuid4()), text="beam222", score=0.8)
        beam3 = Beam(id=str(uuid.uuid4()), text="beam333", score=0.7)
        beams = [beam1, beam2, beam3]
        reviewer = TopKBeamReviewer(k=1, drop_chars=2)
        top_k_beams = reviewer.review(beams)
        assert len(top_k_beams) == 3
        assert top_k_beams[0].text == "beam111"
        assert top_k_beams[1].text == "beam1"
        assert top_k_beams[2].text == "beam1"
