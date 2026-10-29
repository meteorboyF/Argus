"""Recording-first native layout. Pure rendering; no cameras or inference.

Input images must already have the display privacy treatment applied by
DemoView. This layer never changes assistant, privacy, or calibration policy.
"""
from functools import lru_cache
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .sensitive import mask_text

SIZE = (1920,1080)
BG = '#101718'
SURFACE = '#172123'
LINE = '#2C3A3D'
WHITE = '#F1F0E8'
MUTED = '#95A8AA'
ACCENT = '#6AE3D0'
AMBER = '#F4C787'
FEATURES = {
    'seat':'Seat detection',
    'objects':'Object detection',
    'hazards':'Approach detection',
    'mapping':'Local mapping',
    'reading':'Text recognition',
    'conversation':'Conversation',
    'privacy':'Privacy',
    'auto':'Assistant',
}


@lru_cache(maxsize=20)
def font(size,bold=False,mono=False):
    filename = 'UbuntuMono-R.ttf' if mono else ('Ubuntu-B.ttf' if bold else 'Ubuntu-R.ttf')
    candidates = [Path('/usr/share/fonts/truetype/ubuntu')/filename,
                  Path('/usr/share/fonts/truetype/dejavu')/('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path),size)
    return ImageFont.load_default()


def wrap(text, face, width, max_lines):
    """Pixel-based wrap with bounded long tokens and visible truncation."""
    lines=[]
    for paragraph in str(text).splitlines() or ['']:
        line=''
        for word in paragraph.split():
            # Very long OCR tokens must not escape their column.
            pieces=[]
            while face.getlength(word)>width and len(word)>1:
                end=max(1,int(len(word)*width/max(face.getlength(word),1)))
                while end>1 and face.getlength(word[:end])>width:
                    end-=1
                pieces.append(word[:end]); word=word[end:]
            pieces.append(word)
            for piece in pieces:
                candidate=(line+' '+piece).strip()
                if line and face.getlength(candidate)>width:
                    lines.append(line);line=piece
                else:
                    line=candidate
        if line:
            lines.append(line)
    if len(lines)>max_lines:
        lines=lines[:max_lines]
        while lines[-1] and face.getlength(lines[-1]+'…')>width:
            lines[-1]=lines[-1][:-1]
        lines[-1]+='…'
    return lines


@lru_cache(maxsize=16)
def text_tile(text,size,color,width,max_lines=1,bold=False,mono=False):
    face=font(size,bold,mono)
    lines=wrap(mask_text(str(text)),face,width,max_lines)
    height=max(1,len(lines)*(size+10))
    tile=Image.new('RGBA',(width,height),(0,0,0,0))
    draw=ImageDraw.Draw(tile)
    for i,line in enumerate(lines):
        draw.text((0,i*(size+10)),line,font=face,fill=color)
    return tile


def put(image,text,xy,size=24,color=WHITE,width=1000,lines=1,bold=False,mono=False):
    tile=text_tile(str(text),size,color,width,lines,bold,mono)
    image.paste(tile,xy,tile)


def infer_feature(values,requested):
    if requested!='auto':
        return requested
    intent=str(values.get('intent','')).split(' ')[0]
    return {'read':'reading','page_qa':'reading','seat':'seat','find':'objects',
            'describe':'objects','chat':'conversation'}.get(intent,'auto')


def assistant_state(values):
    if values.get('speaking'):
        return 'Speaking'
    status=str(values.get('gemma','')).lower()
    if 'transcrib' in status:
        return 'Understanding'
    if values.get('task_busy'):
        return 'Processing'
    if any(word in status for word in ('looking','phrasing','thinking','working','routing')):
        return 'Processing'
    return 'Listening'


class RecordingLayout:
    def __init__(self,feature='auto'):
        if feature not in FEATURES:
            raise ValueError('Unknown recording feature')
        self.feature=feature
        self._base=Image.new('RGB',SIZE,BG)
        draw=ImageDraw.Draw(self._base)
        draw.line((40,92,1880,92),fill=LINE,width=1)
        draw.rounded_rectangle((40,220,1272,924),radius=18,fill='#0B1112',outline=LINE,width=1)
        draw.rounded_rectangle((1304,220,1880,924),radius=18,fill=SURFACE)
        draw.line((40,966,1880,966),fill=LINE,width=1)
        # Quiet geometric brand mark, built natively without new assets.
        draw.ellipse((42,33,68,59),outline=ACCENT,width=3)
        draw.arc((50,41,60,51),180,360,fill=ACCENT,width=2)
        put(self._base,'ARGUS',(82,25),30,bold=True,width=180)
        put(self._base,'ON-DEVICE ASSISTANT',(251,35),15,color=MUTED,width=400)

    def render(self,snap,frame,source='wide',recording=False,elapsed=0,diagnostics=False,
               face_count=None,privacy_ready=False):
        now=time.monotonic()
        image=self._base.copy(); draw=ImageDraw.Draw(image)
        values=snap.values
        feature=infer_feature(values,self.feature)
        state=assistant_state(values)
        draw.rounded_rectangle((1658,24,1880,70),radius=23,fill='#203D38')
        draw.ellipse((1678,41,1688,51),fill=ACCENT)
        put(image,state,(1702,31),22,color=ACCENT,width=160)

        # Letterbox preserves the entire camera view and page edges.
        if frame is not None and frame.size:
            scale=min(1216/frame.shape[1],640/frame.shape[0])
            size=(max(1,int(frame.shape[1]*scale)),max(1,int(frame.shape[0]*scale)))
            fitted=cv2.resize(frame,size,interpolation=cv2.INTER_AREA)
            rgb=Image.fromarray(cv2.cvtColor(fitted,cv2.COLOR_BGR2RGB))
            image.paste(rgb,(48+(1216-size[0])//2,228+(640-size[1])//2))
        else:
            put(image,'Waiting for camera',(400,505),34,color=MUTED,width=700)
        source_label='LEFT STEREO' if source=='left' else 'WIDE CAMERA'
        draw.rectangle((48,868,1264,916),fill='#182426')
        draw.ellipse((65,887,73,895),fill=ACCENT if frame is not None else AMBER)
        put(image,source_label,(88,879),18,color=WHITE,width=360,mono=True)
        privacy='FACE BLUR ACTIVE' if privacy_ready else 'PRIVACY FILTER UNAVAILABLE'
        if privacy_ready and face_count is not None:
            privacy+=f'  /  {face_count} detected'
        put(image,privacy,(620,880),16,color=MUTED if privacy_ready else AMBER,width=625)

        # A feature-specific evidence panel; OCR text is the real masked output.
        x=1336
        user=next((e.text for e in reversed(snap.events) if e.kind=='user'),'')
        last=next((e.text for e in reversed(snap.events) if e.kind in ('argus','safety')),'')
        response=values.get('speaking') or last
        ocr=str(values.get('reading_ocr_text') or '')
        if feature=='reading':
            put(image,'TEXT RECOGNITION',(x,246),17,color=ACCENT,width=510,mono=True)
            draw.line((x,283,1848,283),fill=LINE)
            put(image,str(values.get('reading_ocr_status') or 'Idle'),
                (x,300),23,color=MUTED,width=510,lines=2)
            put(image,ocr or 'No text captured',(x,385),29,
                color=WHITE if ocr else MUTED,width=510,lines=10)
            put(image,'LAST REQUEST',(x,820),15,color=ACCENT,width=510,mono=True)
            put(image,user or '—',(x,850),23,width=510,lines=2)
        else:
            put(image,'LIVE CONVERSATION',(x,246),17,color=ACCENT,width=510,mono=True)
            draw.line((x,283,1848,283),fill=LINE)
            put(image,'YOU',(x,308),16,color=MUTED,width=510,mono=True)
            put(image,user or '—',(x,342),29,width=510,lines=3)
            put(image,'ARGUS',(x,490),16,color=ACCENT,width=510,mono=True)
            put(image,response or '—',(x,526),31,width=510,lines=7)

        # Clear persistent capability boundary, kept compact in the footer.
        boundary = ('STEREO CALIBRATION LOADED' if values.get('calibrated') else '')
        if feature=='mapping' and not values.get('calibrated'):
            boundary='MAPPING UNAVAILABLE · CALIBRATION REQUIRED'
        if feature=='mapping' and source=='wide':
            boundary='WIDE-CAMERA SLAM NOT IMPLEMENTED'
        unavailable='UNAVAILABLE' in boundary or 'NOT IMPLEMENTED' in boundary
        put(image,boundary,(42,988),17,color=AMBER if unavailable else MUTED,width=950,mono=True)
        if recording:
            draw.ellipse((1538,998,1550,1010),fill='#F07779')
            elapsed=max(0,int(elapsed))
            put(image,f'RECORDING  {elapsed//60:02}:{elapsed%60:02}',(1566,987),21,width=310,mono=True)
        else:
            put(image,'PREVIEW · NOT RECORDING',(1480,987),18,color=MUTED,width=400,mono=True)
        hint=''
        if diagnostics:
            hint='Tab  diagnostics   ·   Esc twice  close'
        if feature=='reading' and response:
            hint='ARGUS  '+str(response)
        if getattr(snap,'values',{}).get('close_hint_until',0)>now:
            hint='Press Esc again to close ARGUS'
        put(image,hint,(42,1032),20,color=MUTED,width=1810)
        output=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)

        if diagnostics:
            from .dashboard import _fit
            tile=_fit(snap.frames.get('depth_color'),360,225)
            output[620:845,880:1240]=tile
            label=f"Depth {values.get('depth_ms',0):.0f}ms | {values.get('fast_hz',0):.1f}Hz"
            cv2.putText(output,label,(888,870),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)

        # Safety always overlays the composed scene, irrespective of mode.
        safety=next((e for e in reversed(snap.events) if e.kind=='safety' and now-e.when<2.5),None)
        if safety:
            alert=Image.fromarray(cv2.cvtColor(output,cv2.COLOR_BGR2RGB))
            d=ImageDraw.Draw(alert)
            d.rounded_rectangle((80,720,1230,842),radius=14,fill='#AF303C')
            put(alert,'SAFETY ALERT',(104,734),18,width=1000,bold=True)
            put(alert,safety.text,(104,768),32,width=1090,lines=1,bold=True)
            output=cv2.cvtColor(np.asarray(alert),cv2.COLOR_RGB2BGR)
        return output
