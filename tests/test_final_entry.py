import pytest
from scripts.reanalyze_final_entry import final_entry


def records(points,battery=30,outcome="returned"):
    rows=[dict(step=i+1,before=a,position=b,battery=battery-i-1,outcome=None)
          for i,(a,b) in enumerate(zip(points,points[1:]))]
    rows[-1]["outcome"]=outcome
    return rows


def test_reverse_inside_then_progress():
    r=records([(9,5),(9,6),(8,6),(8,7),(8,8),(8,9),(9,9)])
    event=final_entry(r)
    assert event["onset"]==1 and event["progress_met_step"]==5


def test_exit_discards_previously_progressing_entry():
    r=records([(5,6),(6,6),(7,6),(8,6),(7,6),(6,6),(5,6),
               (6,6),(7,6),(8,6),(9,6),(9,7),(9,8),(9,9)])
    assert final_entry(r)["onset"]==7


def test_exhaustion_before_and_after_progress():
    r=records([(9,5),(9,6)],battery=1,outcome="exhausted")
    assert final_entry(r)["onset"] is None
    r=records([(9,5),(9,6),(9,7),(9,8)],battery=3,outcome="exhausted")
    assert final_entry(r)["onset"] is None
    r=records([(9,5),(9,6),(9,7),(9,8),(9,9)],battery=4,outcome="exhausted")
    assert final_entry(r)["onset"]==1


def test_no_accumulation_and_final_outside():
    r=records([(6,5),(6,6),(6,7),(6,6),(6,7),(6,6)],outcome="time_limit")
    assert final_entry(r)["onset"] is None
    r=records([(5,6),(6,6),(7,6),(8,6),(7,6),(6,6),(5,6)],outcome="exhausted")
    assert final_entry(r)["onset"] is None


def test_incomplete_rejected():
    with pytest.raises(ValueError): final_entry([])
    with pytest.raises(ValueError): final_entry(records([(9,5),(9,6)],outcome=None))
