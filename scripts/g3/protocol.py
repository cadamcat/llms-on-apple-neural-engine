"""G3 token/clock contract, adapted from the recorded host checks."""

def require(value, name):
    if not value:
        raise ValueError(name)


def validate_request(row, command, input_count, *, require_graph=True):
    require(row['id']==command['id'] and row['mode']==command.get('mode','full'),'request_identity')
    require(row['input_tokens']==input_count,'request_input_length')
    count=row['output_tokens']
    require(0<count<=command['max_new_tokens'],'output_count')
    require(count==command['max_new_tokens'] or
            (command.get('stop_at_eos') and row['first_EOS_index']==count-1),'incomplete_output')
    times=row['token_times'];start=row['request_start_ns'];end=row['request_end_ns']
    require(len(times)==count and start<end,'clock_or_token_count')
    previous=start
    for i,t in enumerate(times):
        require(t['step']==i and previous<t['end_ns']<=end and t['latency_ns']==t['end_ns']-previous,'token_clock')
        previous=t['end_ns']
    require(row['first_token_ns']==times[0]['end_ns'] and row['last_token_ns']==times[-1]['end_ns'],'phase_clock')
    initial=row['initial_processed']
    require(0<=initial<input_count,'initial_KV')
    if command.get('reset',True):
        require(initial==0 and row['prefix_hit_tokens']==0,'reset_or_prefix_reuse')
    if row['mode']=='decode':
        require(initial==input_count-1 and command.get('reset') is False,'decode_continuation')
    require(row['final_KV_count']==input_count+count-1<=32768,'final_KV')
    require(row['decode_steps']==(count if row['mode']=='decode' else count-1),'decode_count')
    if command.get('forced_ids') is not None:
        require(row.get('generated_ids')==command['forced_ids'],'forced_ids')
    if require_graph:
        require(row['telemetry'],'missing_graph_telemetry')
        graphs=row['graphs'];cursor=0;position=initial;pending=None
        for step,t in enumerate(times):
            new=0;completed=0
            while cursor<len(graphs) and graphs[cursor]['monotonic_ns']<=t['end_ns']:
                g=graphs[cursor];cursor+=1
                require(start<=g['monotonic_ns'],'graph_before_request')
                if g['event']=='graph_begin':
                    require(pending is None and g['processed_before']==position,'graph_begin_state')
                    pending=g
                elif g['event']=='graph_end':
                    require(pending is not None and pending['graph']==g['graph'],'unpaired_graph')
                    require(g['valid_new_tokens']==pending['valid_new_tokens']>0 and
                            g['processed_after']==position+g['valid_new_tokens'],'graph_KV')
                    position=g['processed_after'];new+=g['valid_new_tokens'];completed+=1;pending=None
            require(pending is None and completed>0 and new==(input_count-initial if step==0 else 1),'sequential_device_work')
        require(cursor==len(graphs) and position==row['final_KV_count'],'unaccounted_graph_work')
    prefill=(row['first_token_ns']-start)/1e9 if row['mode']!='decode' else None
    decode=((row['last_token_ns']-row['first_token_ns']) if row['mode']!='decode' else row['last_token_ns']-start)/1e9
    return {'passed':True,'request_seconds':(end-start)/1e9,'prefill_seconds':prefill,
            'prefill_tokens_per_second':input_count/prefill if prefill else None,
            'decode_seconds':decode if row['decode_steps'] else None,
            'decode_tokens_per_second':row['decode_steps']/decode if row['decode_steps'] else None}
