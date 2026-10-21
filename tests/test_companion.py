"""Session-only conversation, cancellation, privacy boundaries and reminders."""
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest

from argus.companion import SessionMemory, companion_route
from argus.tasks import TaskRunner
from argus.reminders import Reminders, parse_reminder
from argus.voice_policy import barge_in_command
from argus.agent import GemmaAgent
from argus.config import AgentConfig
from argus.telemetry import Telemetry


def test_observations_expire_and_never_imply_current_position():
    clock = [0]
    memory = SessionMemory(limit=2, clock=lambda: clock[0])
    memory.observe('keys', True, 'left')
    assert 'last view' in memory.report()
    clock[0] = 20
    assert '20 seconds ago' in memory.report()
    assert companion_route('is it still there?',memory)[:2] == ('find','keys')
    clock[0] = 301
    assert memory.last() is None
    memory.observe('phone',True,'right')
    memory.observe('door',False)
    assert len(memory._items) == 2
    assert "doesn't mean" in memory.report()
    memory.forget()
    assert not memory.target and memory.last() is None


@pytest.mark.parametrize('text,skill,name',[
    ('watch my keys','watch','keys'),('remember this place as kitchen','mark_place','kitchen'),
    ('guide me to kitchen','navigate','kitchen'),('where am i','location',''),
    ('what can you do','answer',''),('can i cross','safety_status',''),
    ('forget everything','forget',''),('be brief','preference','brief')])
def test_companion_rules(text,skill,name):
    result = companion_route(text,SessionMemory())
    assert result[:2] == (skill,name)


def test_unclear_pronoun_asks_instead_of_inventing_object():
    assert 'Which object' in companion_route('where is it',SessionMemory())[2]


def test_tasks_are_bounded_and_cancel_token_survives_new_commands():
    runner = TaskRunner()
    started, release, finished = threading.Event(),threading.Event(),threading.Event()
    seen=[]
    def task():
        started.set()
        release.wait(2)
        seen.append(runner.cancelled())
        finished.set()
    assert runner.submit('find keys',task)
    assert started.wait(1)
    assert not runner.submit('second',lambda:None)
    runner.cancel()
    release.set()
    assert finished.wait(1)
    runner.close()
    assert seen == [True]
    assert not runner.submit('after shutdown',lambda:None)


@pytest.mark.parametrize('text,result',[
    ('set a timer for five minutes',(300,'Your timer is ready')),
    ('remind me in 10 seconds to check my tea',(10,'check my tea')),
    ('set a timer for 0 seconds',None),('set a timer for 25 hours',None)])
def test_reminder_parse(text,result):
    assert parse_reminder(text) == result


def test_reminders_fire_once_and_clear():
    clock=[0]
    reminders=Reminders(Mock(),clock=lambda:clock[0],start=False)
    assert reminders.add(2,'Tea')
    clock[0]=2
    assert reminders.due() == ['Tea']
    assert reminders.due() == []
    reminders.add(3,'Other')
    reminders.clear()
    clock[0]=20
    assert reminders.due() == []
    reminders.stop()
    assert not reminders.add(2,'After shutdown')


@pytest.mark.parametrize('text,result',[
    ('Argus, stop','stop'),('Argus, pause','pause'),('stop',None),
    ('Argus, where are my keys',None),('the television says stop',None)])
def test_barge_in_requires_wake_and_control(text,result):
    assert barge_in_command(text) == result


def test_agent_masks_text_keeps_image_bytes_and_does_not_mutate_config(monkeypatch):
    captured=[]
    def post(*args,**kwargs):
        captured.append(kwargs['json'])
        return SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'choices':[{'message':{'content':'Done'}}]})
    monkeypatch.setattr('argus.agent.requests.post',post)
    cfg=AgentConfig(max_tokens=48,temperature=.3)
    agent=GemmaAgent(cfg)
    url='data:image/jpeg;base64,01712345678'
    agent._post([{'role':'user','content':[{'type':'text','text':'Call 01712345678'},
                   {'type':'image_url','image_url':{'url':url}}]}],None,max_tokens=12,temperature=0)
    payload=captured[0]
    assert '01712345678' not in payload['messages'][0]['content'][0]['text']
    assert payload['messages'][0]['content'][1]['image_url']['url'] == url
    assert payload['max_tokens']==12 and payload['temperature']==0 and payload['cache_prompt'] is False
    assert cfg.max_tokens==48 and cfg.temperature==.3


def test_all_telemetry_text_boundaries_mask_private_numbers(capsys):
    bus=Telemetry()
    recorded=[]
    bus.sinks.append(recorded.append)
    bus.log('user','Call 01712345678')
    bus.set(grounding='01712345678')
    bus.record('intent',name='01712345678',nested={'text':'01712345678'})
    assert '01712345678' not in capsys.readouterr().out
    assert '01712345678' not in str(bus.snapshot())
    assert '01712345678' not in str(recorded)


def test_unknown_reminder_duration_never_claims_it_was_set():
    result=companion_route('set a timer for a million years',SessionMemory())
    assert result[0]=='answer' and 'whole number' in result[2]


def test_session_name_and_forget_are_private_in_memory():
    memory=SessionMemory()
    assert companion_route('call me Fardeen',memory)[:2] == ('preference_name','fardeen')
    memory.preferences['name']='Fardeen'
    assert 'Fardeen' in companion_route('what is my name',memory)[2]
    memory.forget()
    assert 'What would you like' in companion_route('what is my name',memory)[2]


def test_fresh_wide_camera_never_returns_stalled_frame():
    from argus.cameras import CameraRig
    rig=CameraRig.__new__(CameraRig)
    frame=np.ones((2,2,3))
    rig.wide=SimpleNamespace(latest=lambda:(frame,time.perf_counter()-2,1))
    assert rig.get_wide_frame() is None
    rig.wide=SimpleNamespace(latest=lambda:(frame,time.perf_counter(),2))
    assert rig.get_wide_frame() is not None


def test_inventory_decodes_seats_without_changing_mover_filter():
    from argus.movers import decode
    from argus.object_inventory import ESSENTIAL_CLASSES,describe
    raw=np.zeros((84,2),np.float32)
    raw[:4,0]=(100,100,40,40)
    raw[60,0]=.9  # chair
    raw[:4,1]=(500,200,50,100)
    raw[4,1]=.9
    assert [d.cls for d in decode(raw,1,0,0,(416,640))]==['person']
    found=decode(raw,1,0,0,(416,640),classes=ESSENTIAL_CLASSES)
    assert {d.cls for d in found}=={'chair','person'}
    result=describe(found,640)
    assert 'chair' in result and 'not verified' in result


def test_cancelled_dialogue_suppresses_model_reply_and_history():
    from argus.dialogue import DialogueManager
    runner=TaskRunner()
    speaker=SimpleNamespace(speak=Mock())
    o=SimpleNamespace(tasks=runner,speaker=speaker,_reader=None)
    def model(*args,**kwargs):
        runner.local.cancel.set()
        return '{"skill":"chat","say":"This must not be spoken"}'
    o.agent=SimpleNamespace(_post_text=model)
    d=DialogueManager(o)
    runner.local.cancel=threading.Event()
    d.handle('how are you today')
    speaker.speak.assert_not_called()
    assert not d.history
    runner.close()


def test_session_conversation_remembers_name_and_clears_history():
    from argus.dialogue import DialogueManager
    said=[]
    o=SimpleNamespace(speaker=SimpleNamespace(speak=lambda t,*_:said.append(t)),_reader=None)
    d=DialogueManager(o)
    d.handle('call me Fardeen')
    d.handle('what is my name')
    assert 'fardeen' in said[-1]
    assert d.expect_reply
    d.handle('forget everything')
    assert not d.history and not d.memory.preferences
    assert 'logs are unchanged' in said[-1]


def test_sensitive_text_is_redacted_after_face_gate():
    from argus.orchestrator import Orchestrator
    from argus.ocr import Reading,Word,order_lines
    from argus.sensitive import find_sensitive
    o=Orchestrator.__new__(Orchestrator)
    frame=np.random.default_rng(0).integers(0,255,(80,160,3),dtype=np.uint8)
    o.privacy=SimpleNamespace(apply=Mock(return_value=(frame,0)))
    box=np.array([[10,10],[120,10],[120,40],[10,40]],np.float32)
    reading=Reading(order_lines([Word('01712345678',box,.99)]),.1,True)
    o._privacy_ocr=SimpleNamespace(read=Mock(return_value=reading))
    safe=o._apply_privacy(frame)
    assert not np.array_equal(safe[10:40,10:120],frame[10:40,10:120])
    assert o.privacy.apply.call_count==1 and o._privacy_ocr.read.call_count==1


def test_find_answer_comes_from_tool_and_not_a_language_model():
    from argus.orchestrator import Orchestrator
    from argus.grounding import Detection
    o=Orchestrator.__new__(Orchestrator)
    frame=np.zeros((80,160,3),np.uint8)
    o.rig=SimpleNamespace(get_wide_frame=lambda:frame)
    o.cfg=SimpleNamespace(privacy=SimpleNamespace(require_gate=True),agent=SimpleNamespace(fast_locate=True))
    o.privacy=SimpleNamespace(ready=True)
    o._apply_privacy=lambda f:f
    o.grounder=SimpleNamespace(find_object=lambda n,f:Detection(n,.9,(5,5,20,20),(12,12)))
    o.memory=SessionMemory()
    o.tasks=TaskRunner()
    o.agent=SimpleNamespace(with_tool_result=Mock(side_effect=AssertionError('Must not ask model')))
    said=[]
    o.speaker=SimpleNamespace(speak=said.append)
    answer=o._handle_query('find the keys')
    assert answer=='I found the keys to your left.'
    assert o.memory.last().found and o.memory.last().direction=='left'
    assert len(said)==1
    o.tasks.close()


def test_long_document_excerpt_is_bounded_relevant_and_masked():
    from argus.document_context import excerpt
    text='\n'.join(['Unrelated sentence.']*800+['The warranty ends in December. Call 01712345678.']+['Other content.']*800)
    result=excerpt(text,'When does the warranty end?',max_chars=600)
    assert len(result)<=600 and 'warranty ends' in result
    assert '01712345678' not in result and 'Excerpt only' in result


@pytest.mark.parametrize('asynchronous',[False,True])
def test_cancel_all_timers_reaches_timer_tool_before_generic_cancel(asynchronous):
    from argus.orchestrator import Orchestrator
    from argus.dialogue import DialogueManager
    o=Orchestrator.__new__(Orchestrator)
    o.memory=SessionMemory()
    o.reminders=Reminders(Mock(),start=False)
    o.reminders.add(60,'Tea')
    o.speaker=SimpleNamespace(speak=Mock())
    o.dialogue=DialogueManager(o)
    if asynchronous:
        o.submit_query('cancel all timers')
    else:
        o.handle_query('cancel all timers')
    assert o.reminders._items==[]
    o.reminders.stop()
