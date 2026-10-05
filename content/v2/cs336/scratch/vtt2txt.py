import re, os, glob, sys

def vtt_to_txt(vtt_path):
    raw = open(vtt_path, encoding='utf-8').read()
    cues = re.split(r'\n\s*\n', raw)
    seen = {}
    for cue in cues:
        lines = [l.strip() for l in cue.strip().split('\n') if l.strip()]
        if not lines:
            continue
        ts = None
        text_lines = []
        for l in lines:
            m = re.match(r'(\d{2,}):(\d{2}):(\d{2})\.(\d{2,3})\s*-->', l)
            if m:
                h, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
                ts = h * 3600 + mm * 60 + ss
            elif 'WEBVTT' in l or l.startswith('NOTE'):
                continue
            else:
                text_lines.append(re.sub(r'<[^>]+>', '', l))
        if ts is None:
            continue
        label = "%02d:%02d" % (ts // 60, ts % 60)
        text = ' '.join(text_lines)
        text = re.sub(r'\[.*?\]', '', text).strip()
        if label in seen:
            seen[label] += ' ' + text
        else:
            seen[label] = text
    order = sorted(seen, key=lambda x: (int(x.split(':')[0]), int(x.split(':')[1])))
    out = []
    for label in order:
        t = seen[label].strip()
        if t:
            out.append("[%s] %s" % (label, t))
    return chr(10).join(out)

base = os.path.dirname(os.path.abspath(__file__))
vtts = sorted(glob.glob('/home/hatch/workspace/stanford-frontier-ai/sources/cs336/subs/*en-US.vtt'))
vtts = [f for f in vtts if '.en-en-US' not in f and '.en-orig' not in f]
for f in vtts:
    nn = os.path.basename(f)[:2]
    txt = vtt_to_txt(f)
    open(os.path.join(base, nn + '.txt'), 'w').write(txt + chr(10))
    print(nn, len(txt.split()), 'words')
