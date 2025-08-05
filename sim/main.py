
def load_timeline(path):
    import json
    with open(path, 'r') as file:
        data = json.load(file)
        
    assert isinstance(data, dict), "Data is not a dictionary"
    assert 'timeline' in data, "Key 'timeline' not found in data"
    assert 'sim' in data, "Key 'sim' not found in data"
    assert 'aggregations' in data, "Key 'aggregations' not found in data"

    assert type(data['timeline']) is list, "Timeline is not a list"
    assert type(data['sim']) is dict, "Sim is not a dictionary"
    assert type(data['aggregations']) is list, "Aggregations is not a list"

    assert len(data['timeline']) > 0, "Timeline is empty"   
    assert len(data['aggregations']) > 0, "Aggregations is empty"

    assert type(data['timeline'][0]) is list, "element of timeline is not a list"
    assert type(data['aggregations'][0]) == int, "element of aggregations is not an int"
    assert type(data['timeline'][0][0]) == dict, "first element of timeline is not a dict"
    return data

