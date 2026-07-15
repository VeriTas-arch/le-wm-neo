import numpy as np

H_DELTA = {0: (-1,0), 1: (0,1), 2: (1,0), 3: (0,-1)}
TURN_TO_COLOR = {1:"red", 2:"blue", 0:"green"}

def turn_heading(h, d):
    return h if d==0 else ((h-1)%4 if d==1 else (h+1)%4)

def generate_maze(required_turns, size=15, rng=None):
    if rng is None: rng = np.random.default_rng()
    n = len(required_turns)
    g = np.ones((size,size), dtype=np.int32)
    path = []
    sr = rng.integers(3, size-3)
    path.append((sr, 0)); g[sr, 0] = 0
    heading = 1; r, c = sr, 0
    # inter_cols = evenly spaced + jitter
    base = np.linspace(3, size-4, n+2, dtype=int)[1:-1]
    inter_cols = [min(size-3, max(3, int(b + rng.integers(-1, 2)))) for b in base]
    if len(inter_cols) < n:
        inter_cols = base[:n]
    
    for i, ic in enumerate(inter_cols):
        while c < ic:
            c += 1; path.append((r,c)); g[r,c] = 0
        path_idx = len(path)-1
        turn = required_turns[i]
        
        # Carve dead ends: short branches in wrong directions
        for tt in [0,1,2]:
            th = turn_heading(heading, tt)
            dr, dc = H_DELTA[th]
            length = rng.integers(2, 5) if tt != turn else 0
            for step in range(1, length+1):
                nr, nc = r+dr*step, c+dc*step
                if 1<=nr<size-1 and 1<=nc<size-1:
                    g[nr,nc] = 0
        
        # Move into correct branch
        heading = turn_heading(heading, turn)
        dr, dc = H_DELTA[heading]
        for _ in range(rng.integers(3, 6)):
            nr, nc = r+dr, c+dc
            if 1<=nr<size-1 and 1<=nc<size-1:
                r,c = nr,nc; path.append((r,c)); g[r,c] = 0
    
    while c < size-1:
        c += 1; path.append((r,c)); g[r,c] = 0
    g[path[-1]] = 2
    
    # Detect intersections
    inters = []
    for i in range(1, len(path)-1):
        di = (path[i][0]-path[i-1][0], path[i][1]-path[i-1][1])
        do = (path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])
        if di != do:
            hi = None
            for h,(hr,hc) in H_DELTA.items():
                if (hr,hc)==di: hi=h
            ho = None
            for h,(hr,hc) in H_DELTA.items():
                if (hr,hc)==do: ho=h
            if hi is not None and ho is not None:
                if hi==ho: t=0
                elif (hi-1)%4==ho: t=1
                else: t=2
                inters.append((i, t))
    
    intersections = inters[:n]
    if len(intersections) < n:
        # Padding: take last n
        raise RuntimeError("retry")
    
    sh = None
    for h,(hr,hc) in H_DELTA.items():
        if (hr,hc)==(path[1][0]-path[0][0], path[1][1]-path[0][1]):
            sh=h
    return {"grid":g.tolist(),"path":path,"intersections":intersections,"start_heading":sh or 1}
