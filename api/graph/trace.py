"""Recorded tool outcomes and deterministic, parent-facing trace descriptions."""

from copy import deepcopy


def tool_event(name, arguments, result, call_id=None):
    return {'tool': name, 'arguments': deepcopy(arguments), 'result': deepcopy(result), 'call_id': call_id}


def source_urls(value):
    urls = set()
    if isinstance(value, dict):
        if isinstance(value.get('source_url'), str) and value['source_url'].strip():
            urls.add(value['source_url'].strip())
        for item in value.values():
            urls.update(source_urls(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            urls.update(source_urls(item))
    return urls


def describe_event(event):
    result = event['result']
    prefix = f"Called {event['tool']} — "
    if not isinstance(result, dict):
        return prefix + str(result)
    details = []
    if result.get('reason'):
        details.append(result['reason'])
    if 'standards' in result:
        rows = result['standards']
        areas = {r['domain'] for r in rows if r.get('domain')}
        details.append(f"found {len(rows)} standards across {len(areas)} areas")
    if 'achievement_levels' in result:
        count = len(result['achievement_levels'])
        details.append(f"returned {count} achievement descriptions" if count else "no achievement descriptions returned")
    if 'progression' in result:
        details.append(f"returned {len(result['progression'])} skill connections")
    if 'programs' in result:
        details.append(f"found {len(result['programs'])} outside programmes")
    if 'chunks' in result:
        relevant = 'relevant ' if result.get('judged') is True and result.get('state') == 'available' else ''
        details.append(f"found {len(result['chunks'])} {relevant}passages")
    return prefix + '; '.join(details or [f"returned {result.get('state', 'a result')}"])
