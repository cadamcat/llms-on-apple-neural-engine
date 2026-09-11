"""G2 plots reuse the documentation canvas and portable scalar recomputation."""
import statistics
from .canvas import Canvas, T_SMALL, num
from g2.evidence import derive, FIELDS

COLORS = {'C': 'teal', 'G': 'blue', None: 'slate'}
BASE = 'results/historical/g2-w4a16-night/'


def panel(c, x, y, w, h, title, ymax, ticks, xticks, xlabel='', ymin=0):
    c.text(x, y-14, title, 12, 'ink', 'sb')
    for value in ticks:
        yy = y+h-h*(value-ymin)/(ymax-ymin)
        c.line(x,yy,x+w,yy,'grid')
        c.text(x-9,yy+4,f'{value:g}',T_SMALL,'muted',anchor='end')
    for xx, label in xticks:
        c.text(x+xx*w,y+h+18,label,T_SMALL,'muted',anchor='middle')
    if xlabel:
        c.text(x+w/2,y+h+36,xlabel,T_SMALL,'muted',anchor='middle')
    return lambda t:x+t*w, lambda v:y+h-h*(v-ymin)/(ymax-ymin)


def series(c, points, color, width=1.5, dash='', dots=False, gap=None):
    paths=[]; current=[]
    for point in points:
        if current and gap is not None and point[0]-current[-1][0] > gap:
            paths.append(current); current=[]
        current.append(point)
    if current:
        paths.append(current)
    for part in paths:
        if len(part)>1:
            shape='M'+' L'.join(f'{num(x)} {num(y)}' for x,y in part)
            c.path(shape,color,width,dash=dash)
    if dots:
        for x,y in points:
            c.dot(x,y,3.2,color)


def throughput(d):
    c=Canvas('G2: native W4A16 throughput','Seven MLP position counts, all three hosts per engine and their medians.',470)
    c.header('W4A16: seven sizes, three hosts per engine',
             'Gemma-derived first-layer MLP · same-source Q4_0 · M5 Pro',
             'Thin lines: individual hosts. Thick lines: median host rate. No confidence interval.')
    ns=[64,128,256,512,1024,2048,4096]
    x,y=panel(c,90,125,740,255,'MLP positions/s',35000,[0,10000,20000,30000,35000],
              [(i/6,str(n)) for i,n in enumerate(ns)],'Positions per request (log₂ spacing)')
    for engine in ('G','C'):
        for rnd in range(3):
            cells={r['positions']:r for r in d['p2'] if r['engine']==engine and r['round']==rnd and not r['diagnostic']}
            series(c,[(x(i/6),y(cells[n]['rate'])) for i,n in enumerate(ns)],COLORS[engine],.8,dots=True)
        series(c,[(x(i/6),y(r[engine]['positions_per_second'])) for i,r in enumerate(d['medians'])],COLORS[engine],3)
    c.legend(460,100,[('line','teal','ANE'),('line','blue','MLX GPU')])
    c.footnote(440,'One complete MLP component, not a token rate. Ventura screensaver reported; its effect was not isolated.')
    return c,d['medians']


def matched(d):
    c=Canvas('G2: matched-rate temperature observations','GPU minus ANE mean sensor temperature across three equal arrival rates and two orders.',430)
    c.header('Equal arrival rates: observed temperature differences',
             'Six-minute observation means · positive values mean the GPU-inference block read warmer',
             'All six starting groups matched; most blocks did not reach an approximate thermal platform.')
    for ix,(field,label) in enumerate([('cpu_mean_G_minus_C','CPU sensor mean Δ (°C)'),('gpu_mean_G_minus_C','GPU sensor mean Δ (°C)')]):
        x,y=panel(c,85+ix*425,135,310,205,label,4,[0,1,2,3,4],[(0,'0.25R'),(.5,'0.50R'),(1,'0.75R')])
        for rnd in (0,1):
            selected=[next(r for r in d['expected']['P3_pairs'] if r['group']==f'P3-{fraction}-r{rnd}') for fraction in ('0.25','0.5','0.75')]
            series(c,[(x(i/2),y(r[field])) for i,r in enumerate(selected)],'blue' if rnd==0 else 'teal',2,dash='' if rnd==0 else '5 4',dots=True)
    c.legend(265,98,[('line','blue','Order 1'),('line','teal','Order 2 (dashed)')])
    c.footnote(385,'R = 6.5925 requests/s from an independent pilot. Same input IDs and completed work within each pair.',
               'Observed means, not equilibrium or case temperature. Power sampling ended during the first 0.75R GPU block.')
    return c,d['expected']['P3_pairs']


def saturated(d):
    c=Canvas('G2: sustained temperature and fans','Four saturated blocks, each with sixteen minutes of service, shown separately for both temperatures and both fans.',670)
    c.header('Saturated service: temperatures and fan speeds',
             'Four separate blocks · each engine completes as much work as it can',
             'All four final five-minute windows met the recorded platform rule; starting pairs matched.')
    names=['SAT-0-C','SAT-1-G','SAT-2-G','SAT-3-C']
    c.legend(120,98,[('line','teal','ANE 0'),('line','blue','GPU 1'),('line','blue','GPU 2 (dashed)'),('line','teal','ANE 3 (dashed)')])
    points=[]
    for i,(field,label,limit,ticks) in enumerate([(FIELDS[0],'CPU sensor mean (°C)',90,[0,30,60,90]),
        (FIELDS[1],'GPU sensor mean (°C)',90,[0,30,60,90]),(FIELDS[2],'Fan 0 (RPM)',6000,[0,2000,4000,6000]),
        (FIELDS[3],'Fan 1 (RPM)',6000,[0,2000,4000,6000])]):
        x,y=panel(c,90+(i%2)*420,145+(i//2)*225,325,150,label,limit,ticks,
                  [(j/4,str(j*4)) for j in range(5)],'Minutes in observation')
        for j,name in enumerate(names):
            slot=d['slots'][name];start,end=slot['observe_start_ns'],slot['observe_end_ns']
            selected=[r for r in d['sensors'] if start<=r['sample_monotonic_ns']<end]
            pp=[(x((r['sample_monotonic_ns']-start)/(end-start)),y(r[field])) for r in selected]
            series(c,pp,COLORS[slot['config']['engine']],1.5,dash='5 4' if j>=2 else '',gap=325*10/960)
        points.append({'field':field,'source':'sensors.jsonl.gz','blocks':names})
    rates=[d['slots'][n]['service']['inference']['observation_throughput_per_second'] for n in names]
    c.footnote(597,'Requests/s, in block order: '+', '.join(f'{v:.3f}' for v in rates)+'. Unequal completed work; no equal-work energy claim.',
               'Sensor means are not case or ANE junction temperatures. Fan RPM is not acoustic noise.',
               'Ventura screensaver reported after the run; no animation-off control was captured.')
    return c,points


def coexist(d):
    c=Canvas('G2: GPU foreground tail latency','All eighteen coexistence slots: p95 and p99 for matrix and memory-access foregrounds, with thermal-start labels.',690)
    c.header('GPU foreground tails: all eighteen slots',
             'Foreground alone, with ANE inference, and with GPU inference · lower is better',
             'Each panel uses its own latency axis. Tails include completed jobs; all-arrival violations are tabulated separately.')
    c.legend(180,100,[('dot','slate','Alone'),('dot','teal','With ANE'),('dot','blue','With GPU')])
    source=[]
    for i,case in enumerate(('matrix_050','matrix_075','memory_050')):
        limit=70 if case.startswith('matrix') else 14
        ticks=[0,20,40,60] if limit==70 else [0,4,8,12]
        px=80+i*285
        for rnd in (0,1):
            group=next(g for g in d['starts'] if g['group_id']==f'P1-{case}-r{rnd}')
            title=f"{case.split('_')[0].title()} · {int(case.split('_')[1])/100:.2f}F · order {rnd+1}"
            x,y=panel(c,px,155+rnd*240,205,155,title,limit,ticks,[(0,'alone'),(.5,'ANE'),(1,'GPU')])
            c.text(px,155+rnd*240-31,group['status'],T_SMALL,'teal' if group['status']=='matched' else 'amber')
            for q in ('p95','p99'):
                pp=[]
                for j,engine in enumerate((None,'C','G')):
                    name=f'P1-{case}-r{rnd}-{engine or "alone"}';s=d['slots'][name]['service']['foreground']
                    value=s['response_seconds_completed_only'][q]*1000
                    pp.append((x(j/2),y(value)))
                    c.dot(x(j/2),y(value),3.2,COLORS[engine])
                    source.append({'slot':name,'quantile':q,'ms':value,'thermal_start':group['status']})
                series(c,pp,'muted',1.2,dash='4 3' if q=='p99' else '')
    c.footnote(605,'Solid: p95. Dashed: p99. Only the second 0.5F matrix group had matched thermal starts.',
               'Memory-access baseline violations were anomalous in both orders; no claim that ANE improves memory performance.',
               'Fixed offered rates do not measure maximum spare GPU capacity. Ventura effect remains unquantified.')
    return c,source


def memory(d):
    c=Canvas('G2: sampled process memory','All audited r4 samples: total owned RSS and component footprint, with P0 checkpoints in the evidence bundle.',470)
    c.header('Memory stayed inside the sampled run limits',
             'All r4 resource samples · process sums, not model weight size or whole-machine memory',
             'Multiple fresh hosts across phases; this is not one model held resident for eight hours.')
    origin=d['protocol']['origin_monotonic_ns']
    x,y=panel(c,90,135,740,220,'GiB',8,[0,2,4,6,8],[(i/4,str(i*120)) for i in range(5)],'Minutes from original protocol start')
    for field,color in [('all_owned_rss_bytes','blue'),('component_footprint_bytes','teal')]:
        pp=[(x((r['monotonic_ns']-origin)/60e9/480),y(r[field]/2**30)) for r in d['resources']]
        series(c,pp,color,1,gap=740*2/28800)
    c.line(x(0),y(4),x(1),y(4),'teal',1,dash='5 4')
    c.legend(250,100,[('line','blue','All-owned RSS'),('line','teal','Component footprint')])
    p1=[r for r in d['resources'] if r['phase']=='p1_gpu_coexistence']
    peak=max(p1,key=lambda r:r['all_owned_rss_bytes'])
    span=[(r['monotonic_ns']-origin)/60e9 for r in p1]
    native=max(r['last_footprint_bytes'] for r in d['native_growth'] if r['engine']=='C')
    c.footnote(410,'Limits: all-owned RSS 8 GiB; component footprint 4 GiB (dashed). Gaps above two seconds are not joined.',
               f'Minutes {min(span):.0f}–{max(span):.0f}: the rise-and-reset is the memory-access foreground '
               f'({peak["foreground_rss_bytes"]/2**30:.2f} of {peak["all_owned_rss_bytes"]/2**30:.2f} GiB at peak). '
               f'Native ANE P0 hosts ended at {native/2**20:.0f} MiB.',
               'No observed swap growth; initial swap was nonzero. Finite sampling does not establish indefinite residency.')
    return c,{'source':'resources.jsonl.gz','fields':['all_owned_rss_bytes','component_footprint_bytes'],'gap_seconds':2}


def load_fans(d):
    slots=d['slots']; floor=d['fan_floor']
    rate=lambda s:s['service']['inference']['observation_throughput_per_second']
    blocks=[]
    for name,slot in slots.items():
        if name.startswith('P3-'):
            values,statistic=slot['sensor_means'],'six-minute window mean'
        elif name.startswith('SAT-'):
            values,statistic=slot['tail_medians'],'final five-minute median'
        else:
            continue
        blocks.append({'slot':name,'engine':slot['config']['engine'],'statistic':statistic,
                       'requests_per_second':rate(slot),'gpu_sensor_c':values[FIELDS[1]],
                       'fan0_rpm':values[FIELDS[2]],'fan1_rpm':values[FIELDS[3]]})
    equal_rate=[b for b in blocks if b['slot'].startswith('P3-')]
    low=min(b['requests_per_second'] for b in equal_rate); top=max(b['requests_per_second'] for b in equal_rate)
    sat={e:[b['requests_per_second'] for b in blocks if b['slot'].startswith('SAT-') and b['engine']==e] for e in 'CG'}
    ane_max=statistics.mean(sat['C']); gpu_low=min(sat['G']); gpu_max=statistics.mean(sat['G'])
    cpu=[r['cpu_mean_G_minus_C'] for r in d['expected']['P3_pairs']]
    fan1_equal=[b['fan1_rpm'] for b in equal_rate]
    fan1_sat=[b['fan1_rpm'] for b in blocks if b['slot'].startswith('SAT-') and b['engine']=='G']
    c=Canvas('G2: load against temperature and fan speed',
             'Completed requests per second against GPU sensor temperature and fan 0 speed, for all six equal-rate pairs and all four saturated blocks; fan 1 values are given in the caption.',500)
    c.header('At equal load, fans stayed at idle on both engines',
             'Temperature readings differed by a few degrees; fan speed rose only in the saturated GPU blocks',
             f'GPU loads between {top:.1f} and {gpu_low:.1f} requests/s were not measured · one M5 Pro · first-layer MLP')
    w,h,top_y=325,220,140
    xticks=[(v/25,str(v)) for v in (0,5,10,15,20,25)]
    for px in (90,510):
        x0,x1=px+top/25*w,px+gpu_low/25*w
        c.tint(x0,top_y,x1-x0,h,'slate',0,stroke=False)
    left=panel(c,90,top_y,w,h,'GPU sensor mean (°C)',80,[40,50,60,70,80],xticks,'Completed requests/s',ymin=40)
    right=panel(c,510,top_y,w,h,'Fan 0 speed (RPM)',4000,[0,1000,2000,3000,4000],xticks,'Completed requests/s')
    for px,(x,y) in ((90,left),(510,right)):
        c.text((x(top/25)+x(gpu_low/25))/2,top_y+16,'GPU not measured',T_SMALL,'muted',anchor='middle')
        c.line(x(ane_max/25),top_y,x(ane_max/25),top_y+h,'teal',1,dash='4 3')
        c.text(x(ane_max/25)+4,top_y+h-8,'ANE maximum',T_SMALL,'teal')
    x,y=right
    c.line(510,y(floor['fan0_rpm']),510+w,y(floor['fan0_rpm']),'muted',1,dash='2 3')
    c.text(510+w,y(floor['fan0_rpm'])-6,'idle',T_SMALL,'muted',anchor='end')
    for block in sorted(blocks,key=lambda b:b['engine']!='G'):
        size=1 if block['engine']=='G' else .6
        xx,yy=left; c.dot(xx(block['requests_per_second']/25),yy(block['gpu_sensor_c']),5*size,COLORS[block['engine']])
        xx,yy=right
        c.dot(xx(block['requests_per_second']/25),yy(block['fan0_rpm']),5*size,COLORS[block['engine']])
    c.legend(390,100,[('dot','blue','GPU'),('dot','teal','ANE')])
    c.footnote(430,f'Six equal-rate pairs at {low:.1f}–{top:.1f} requests/s (window means, matched starts); saturated blocks at '
                   f'{ane_max:.1f} and {gpu_max:.1f} (final five-minute medians, platform rule met).',
               f'The ANE marker is drawn inside the GPU marker; dashed: median block-start fan 0 speed. '
               f'CPU sensor difference at equal load: {min(cpu):.1f}–{max(cpu):.1f} °C.',
               f'Fan 1 follows the same pattern: idle {floor["fan1_rpm"]:,.0f} RPM, {min(fan1_equal):,.0f}–{max(fan1_equal):,.0f} '
               f'at equal load, {min(fan1_sat):,.0f}–{max(fan1_sat):,.0f} in saturated GPU blocks.',
               'Sensor means are not case temperatures; RPM is not acoustic noise. Ventura screensaver reported; energy undetermined.')
    return c,{'blocks':blocks,'fan_floor':floor,'unmeasured_gpu_requests_per_second':[top,gpu_low]}


def tile(d):
    c=Canvas('G2: paired tile diagnostic','At 1024 positions, tile 64 and tile 256 run in the same native host in each of three rounds.',360)
    c.header('A larger ANE tile helps within the same host',
             'N1024 diagnostic · tile 64 versus tile 256 · three separate hosts',
             'Pairs overlap at this scale; individual ratios are shown. The main tile policy remains fixed.')
    x,y=panel(c,165,120,470,150,'MLP positions/s',7500,[0,2500,5000,7500],[(0,'tile 64'),(1,'tile 256')])
    points=[]
    for rnd,color in enumerate(('teal','blue','plum')):
        cells={r['tile']:r for r in d['p2'] if r['engine']=='C' and r['round']==rnd and r['positions']==1024}
        values=[cells[t]['rate'] for t in (64,256)]
        series(c,[(x(j),y(v)) for j,v in enumerate(values)],color,1.5,dots=True)
        ratio=values[1]/values[0]
        c.text(685,160+27*rnd,f'Round {rnd+1}: {ratio:.3f}×',12,color,'sb')
        points.append({'round':rnd,'tile64':values[0],'tile256':values[1],'ratio':ratio})
    c.footnote(325,'More positions per ANE stage call reduce the measured cost here; no claim about an untested fused graph.')
    return c,points


def generate(repo, out):
    d=derive(repo/BASE)
    all_sources=[BASE+p.name for p in sorted((repo/BASE).iterdir()) if p.name!='README.md']
    for filename,fn in [('g2-throughput.svg',throughput),('g2-load-fans.svg',load_fans),
                       ('g2-matched-rate-thermal.svg',matched),
                       ('g2-saturated-thermal.svg',saturated),('g2-coexistence-tails.svg',coexist),
                       ('g2-memory.svg',memory),('g2-tile.svg',tile)]:
        canvas,points=fn(d)
        (out/filename).write_text(canvas.finish())
        yield {'filename':filename,'sources':all_sources,'points_or_scope':points,
               'generation':'Standard-library Canvas; values checked by scripts/g2/evidence.py; no device calls.',
               'scope':'One M5 Pro, one first-layer MLP, r4 only; Ventura dynamic screensaver reported; energy undetermined.'}
