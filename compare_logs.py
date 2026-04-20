import json

with open("intercepted_requests.log", "r") as f:
    content = f.read()

sessions = content.split("=== Session ID: ")
for s in sessions:
    if not s.strip(): continue
    lines = s.split("\n")
    sid = lines[0].split(" | ")[0]
    time = lines[0].split(" | ")[1]
    body_str = "\n".join(lines[1:])
    try:
        body = json.loads(body_str)
        print(f"SID: {sid} | Time: {time}")
        for m in body:
            print(f"  {m['role']}: {m['content'][:50]}...")
    except:
        pass
