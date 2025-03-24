"""Functions for computing maximum weighted matching in a graph."""

from itertools import repeat


def matching_dict_to_set(matching):
    """
    Converts matching dictionary to a set of edges.
    
    The dictionary has pairs of vertices as keys and values.
    Converts this to a set of edges format where each edge is a tuple (u, v).
    """
    edges = set()
    for edge in matching.items():
        u, v = edge
        if (v, u) in edges or edge in edges:
            continue
        if u == v:
            raise ValueError(f"Self-loops cannot appear in matchings {edge}")
        edges.add(edge)
    return edges


def max_weight_matching(G, maxcardinality=False, weight="weight"):
    """
    Compute a maximum-weighted matching of G.
    
    A matching is a subset of edges where no node appears more than once.
    The weight of a matching is the sum of the weights of its edges.
    
    Parameters
    ----------
    G : NetworkX graph
        Undirected graph
    maxcardinality : bool, optional (default=False)
        If True, compute the maximum-cardinality matching with maximum
        weight among all maximum-cardinality matchings.
    weight : string, optional (default='weight')
        Edge data key corresponding to the edge weight.
        If key not found, uses 1 as weight.
    
    Returns
    -------
    matching : set
        A maximal matching of the graph.
    
    Notes
    -----
    This implementation uses the "blossom" method for finding augmenting paths
    and the "primal-dual" method for finding a matching of maximum weight.
    This implementation is based on Galil's 1986 paper and takes O(n³) time.
    """
    # Helper classes
    class NoNode:
        """Dummy value different from any node."""
        pass

    class Blossom:
        """Represents a non-trivial blossom or sub-blossom."""
        __slots__ = ["childs", "edges", "mybestedges"]
        
        def __init__(self):
            self.childs = []  # Ordered list of sub-blossoms
            self.edges = []   # List of connecting edges
            self.mybestedges = None  # List of least-slack edges
        
        def leaves(self):
            """Generate all leaf vertices of this blossom."""
            stack = list(self.childs)
            while stack:
                t = stack.pop()
                if isinstance(t, Blossom):
                    stack.extend(t.childs)
                else:
                    yield t

    # Setup initial values
    gnodes = list(G)
    if not gnodes:
        return set()  # Empty graph

    # Find maximum edge weight
    maxweight = 0
    allinteger = True
    for i, j, d in G.edges(data=True):
        wt = d.get(weight, 1)
        if i != j and wt > maxweight:
            maxweight = wt
        allinteger = allinteger and (str(type(wt)).split("'")[1] in ("int", "long"))

    # Initialize data structures
    mate = {}  # Matched vertex for each vertex
    label = {}  # Labels for blossoms (1=S, 2=T)
    labeledge = {}  # Edge through which a blossom got its label
    inblossom = dict(zip(gnodes, gnodes))  # Top-level blossom for each vertex
    blossomparent = dict(zip(gnodes, repeat(None)))  # Parent blossom
    blossombase = dict(zip(gnodes, gnodes))  # Base vertex for a blossom
    bestedge = {}  # Best edge from a blossom
    dualvar = dict(zip(gnodes, repeat(maxweight)))  # Dual variables
    blossomdual = {}  # Dual variables for blossoms
    allowedge = {}  # Edges with zero slack
    queue = []  # Queue of newly discovered S-vertices

    # Return 2 * slack of edge (v, w)
    def slack(v, w):
        return dualvar[v] + dualvar[w] - 2 * G[v][w].get(weight, 1)

    # Assign label to a blossom
    def assignLabel(w, t, v):
        b = inblossom[w]
        assert label.get(w) is None and label.get(b) is None
        label[w] = label[b] = t
        if v is not None:
            labeledge[w] = labeledge[b] = (v, w)
        else:
            labeledge[w] = labeledge[b] = None
        bestedge[w] = bestedge[b] = None
        if t == 1:
            # b became an S-vertex/blossom; add it to the queue
            if isinstance(b, Blossom):
                queue.extend(b.leaves())
            else:
                queue.append(b)
        elif t == 2:
            # b became a T-vertex/blossom; assign label S to its mate
            base = blossombase[b]
            assignLabel(mate[base], 1, base)

    # Trace back from vertices v and w to discover either a new blossom
    # or an augmenting path
    def scanBlossom(v, w):
        path = []
        base = NoNode
        while v is not NoNode:
            b = inblossom[v]
            if label[b] & 4:
                base = blossombase[b]
                break
            assert label[b] == 1
            path.append(b)
            label[b] = 5
            if labeledge[b] is None:
                # Base of blossom is single; stop tracing
                assert blossombase[b] not in mate
                v = NoNode
            else:
                assert labeledge[b][0] == mate[blossombase[b]]
                v = labeledge[b][0]
                b = inblossom[v]
                assert label[b] == 2
                v = labeledge[b][0]
            if w is not NoNode:
                v, w = w, v
        # Remove breadcrumbs
        for b in path:
            label[b] = 1
        return base

    # Construct a new blossom with given base
    def addBlossom(base, v, w):
        bb = inblossom[base]
        bv = inblossom[v]
        bw = inblossom[w]
        
        # Create blossom
        b = Blossom()
        blossombase[b] = base
        blossomparent[b] = None
        blossomparent[bb] = b
        
        # Make list of sub-blossoms and edges
        b.childs = path = []
        b.edges = [(v, w)]
        
        # Trace back from v to base
        while bv != bb:
            blossomparent[bv] = b
            path.append(bv)
            b.edges.append(labeledge[bv])
            v = labeledge[bv][0]
            bv = inblossom[v]
        
        # Add base blossom and reverse lists
        path.append(bb)
        path.reverse()
        b.edges.reverse()
        
        # Trace from w to base
        while bw != bb:
            blossomparent[bw] = b
            path.append(bw)
            b.edges.append((labeledge[bw][1], labeledge[bw][0]))
            w = labeledge[bw][0]
            bw = inblossom[w]
        
        # Set label and dual variable
        label[b] = 1
        labeledge[b] = labeledge[bb]
        blossomdual[b] = 0
        
        # Relabel vertices
        for v in b.leaves():
            if label[inblossom[v]] == 2:
                # This T-vertex now becomes an S-vertex
                queue.append(v)
            inblossom[v] = b
        
        # Compute b.mybestedges
        bestedgeto = {}
        for bv in path:
            if isinstance(bv, Blossom):
                if bv.mybestedges is not None:
                    nblist = bv.mybestedges
                    bv.mybestedges = None
                else:
                    nblist = [(v, w) for v in bv.leaves() for w in G.neighbors(v) if v != w]
            else:
                nblist = [(bv, w) for w in G.neighbors(bv) if bv != w]
            
            for k in nblist:
                i, j = k
                if inblossom[j] == b:
                    i, j = j, i
                bj = inblossom[j]
                if (bj != b and label.get(bj) == 1 and 
                    ((bj not in bestedgeto) or slack(i, j) < slack(*bestedgeto[bj]))):
                    bestedgeto[bj] = k
            bestedge[bv] = None
        
        b.mybestedges = list(bestedgeto.values())
        
        # Select bestedge[b]
        mybestedge = None
        mybestslack = None
        for k in b.mybestedges:
            kslack = slack(*k)
            if mybestedge is None or kslack < mybestslack:
                mybestedge = k
                mybestslack = kslack
        bestedge[b] = mybestedge

    # Expand a blossom
    def expandBlossom(b, endstage):
        for s in b.childs:
            blossomparent[s] = None
            if isinstance(s, Blossom):
                if endstage and blossomdual[s] == 0:
                    expandBlossom(s, endstage)
                else:
                    for v in s.leaves():
                        inblossom[v] = s
            else:
                inblossom[s] = s
        
        # If we expand a T-blossom during a stage, we need to relabel
        if (not endstage) and label.get(b) == 2:
            entrychild = inblossom[labeledge[b][1]]
            j = b.childs.index(entrychild)
            if j & 1:
                j -= len(b.childs)
                jstep = 1
            else:
                jstep = -1
            
            # Move along blossom until base
            v, w = labeledge[b]
            while j != 0:
                # Relabel the T-sub-blossom
                if jstep == 1:
                    p, q = b.edges[j]
                else:
                    q, p = b.edges[j - 1]
                label[w] = None
                label[q] = None
                assignLabel(w, 2, v)
                
                # Step to next S-blossom
                allowedge[(p, q)] = allowedge[(q, p)] = True
                j += jstep
                if jstep == 1:
                    v, w = b.edges[j]
                else:
                    w, v = b.edges[j - 1]
                
                # Step to next T-blossom
                allowedge[(v, w)] = allowedge[(w, v)] = True
                j += jstep
            
            # Relabel base T-blossom
            bw = b.childs[j]
            label[w] = label[bw] = 2
            labeledge[w] = labeledge[bw] = (v, w)
            bestedge[bw] = None
            
            # Continue until we reach entrychild
            j += jstep
            while b.childs[j] != entrychild:
                bv = b.childs[j]
                if label.get(bv) == 1:
                    j += jstep
                    continue
                
                if isinstance(bv, Blossom):
                    for v in bv.leaves():
                        if label.get(v):
                            break
                else:
                    v = bv
                
                if label.get(v):
                    assert label[v] == 2
                    assert inblossom[v] == bv
                    label[v] = None
                    label[mate[blossombase[bv]]] = None
                    assignLabel(v, 2, labeledge[v][0])
                j += jstep
        
        # Remove the expanded blossom
        label.pop(b, None)
        labeledge.pop(b, None)
        bestedge.pop(b, None)
        del blossomparent[b]
        del blossombase[b]
        del blossomdual[b]

    # Augment matching along path
    def augmentMatching(v, w):
        for s, j in ((v, w), (w, v)):
            # Match vertex s to vertex j
            while True:
                bs = inblossom[s]
                assert label[bs] == 1
                assert (labeledge[bs] is None and blossombase[bs] not in mate) or (
                    labeledge[bs][0] == mate[blossombase[bs]])
                
                # Augment through the S-blossom
                if isinstance(bs, Blossom):
                    augmentBlossom(bs, s)
                
                # Update mate[s]
                mate[s] = j
                
                # Follow the trail back
                if labeledge[bs] is None:
                    # Reached single vertex; done
                    break
                
                t = labeledge[bs][0]
                bt = inblossom[t]
                assert label[bt] == 2
                
                # Continue with the next S-vertex
                s, j = labeledge[bt]
                
                # Augment through the T-blossom
                assert blossombase[bt] == t
                if isinstance(bt, Blossom):
                    augmentBlossom(bt, j)
                
                # Update mate[j]
                mate[j] = s

    # Augment matching between vertex v and base vertex
    def augmentBlossom(b, v):
        # Find the right position within blossom
        t = v
        while blossomparent[t] != b:
            t = blossomparent[t]
        
        # Recursive augment for the first sub-blossom if needed
        if isinstance(t, Blossom):
            augmentBlossom(t, v)
        
        # Determine direction around the blossom
        i = j = b.childs.index(t)
        if i & 1:
            j -= len(b.childs)
            jstep = 1
        else:
            jstep = -1
        
        # Move along the blossom until we get to the base
        while j != 0:
            # Update the next pair of sub-blossoms
            j += jstep
            t = b.childs[j]
            if jstep == 1:
                w, x = b.edges[j]
            else:
                x, w = b.edges[j - 1]
            
            # Recursively handle internal blossoms if needed
            if isinstance(t, Blossom):
                augmentBlossom(t, w)
            
            j += jstep
            t = b.childs[j]
            if isinstance(t, Blossom):
                augmentBlossom(t, x)
            
            # Match the edge directly
            mate[w] = x
            mate[x] = w
        
        # Rotate the list of sub-blossoms to put the new base at the front
        b.childs = b.childs[i:] + b.childs[:i]
        b.edges = b.edges[i:] + b.edges[:i]
        blossombase[b] = blossombase[b.childs[0]]
        assert blossombase[b] == v

    # Main algorithm loop
    while True:
        # Initialize for a new stage
        label.clear()
        labeledge.clear()
        bestedge.clear()
        for b in blossomdual:
            b.mybestedges = None
        allowedge.clear()
        queue[:] = []
        
        # Label free vertices with S and add them to the queue
        for v in gnodes:
            if v not in mate and label.get(inblossom[v]) is None:
                assignLabel(v, 1, None)
        
        # Loop until we succeed in augmenting the matching
        augmented = 0
        while True:
            # Continue labeling until no more progress or augment found
            while queue and not augmented:
                v = queue.pop()
                assert label[inblossom[v]] == 1
                
                # Scan all neighbors of v
                for w in G.neighbors(v):
                    if w == v:
                        continue  # Skip self-loops
                    
                    bv = inblossom[v]
                    bw = inblossom[w]
                    if bv == bw:
                        continue  # Skip internal edges
                    
                    if (v, w) not in allowedge:
                        kslack = slack(v, w)
                        if kslack <= 0:
                            allowedge[(v, w)] = allowedge[(w, v)] = True
                    
                    if (v, w) in allowedge:
                        if label.get(bw) is None:
                            # w is free; label it with T and its mate with S
                            assignLabel(w, 2, v)
                        elif label.get(bw) == 1:
                            # w is an S-vertex; trace for an augmenting path or blossom
                            base = scanBlossom(v, w)
                            if base is not NoNode:
                                # Found a new blossom; add it
                                addBlossom(base, v, w)
                            else:
                                # Found an augmenting path
                                augmentMatching(v, w)
                                augmented = 1
                                break
                        elif label.get(w) is None:
                            # w is inside a T-blossom but not yet reached
                            assert label[bw] == 2
                            label[w] = 2
                            labeledge[w] = (v, w)
                    elif label.get(bw) == 1:
                        # Track least-slack non-allowable edge to a different S-blossom
                        if bestedge.get(bv) is None or kslack < slack(*bestedge[bv]):
                            bestedge[bv] = (v, w)
                    elif label.get(w) is None:
                        # Track least-slack edge to an unreached vertex
                        if bestedge.get(w) is None or kslack < slack(*bestedge[w]):
                            bestedge[w] = (v, w)
            
            if augmented:
                break  # Go to the next stage
            
            # No augmenting path found; update dual variables
            # Find the type of update with the minimum delta
            deltatype = -1
            delta = deltaedge = deltablossom = None
            
            # Delta1: minimum value of any vertex dual
            if not maxcardinality:
                deltatype = 1
                delta = min(dualvar.values())
            
            # Delta2: minimum slack on any edge to free vertex
            for v in G.nodes():
                if label.get(inblossom[v]) is None and bestedge.get(v) is not None:
                    d = slack(*bestedge[v])
                    if deltatype == -1 or d < delta:
                        delta = d
                        deltatype = 2
                        deltaedge = bestedge[v]
            
            # Delta3: half the minimum slack on any edge between S-blossoms
            for b in blossomparent:
                if (blossomparent[b] is None and label.get(b) == 1 and 
                    bestedge.get(b) is not None):
                    kslack = slack(*bestedge[b])
                    if allinteger:
                        d = kslack // 2
                    else:
                        d = kslack / 2.0
                    if deltatype == -1 or d < delta:
                        delta = d
                        deltatype = 3
                        deltaedge = bestedge[b]
            
            # Delta4: minimum z variable of any T-blossom
            for b in blossomdual:
                if (blossomparent[b] is None and label.get(b) == 2 and
                    (deltatype == -1 or blossomdual[b] < delta)):
                    delta = blossomdual[b]
                    deltatype = 4
                    deltablossom = b
            
            if deltatype == -1:
                # No further improvement possible
                assert maxcardinality
                deltatype = 1
                delta = max(0, min(dualvar.values()))
            
            # Update dual variables
            for v in gnodes:
                if label.get(inblossom[v]) == 1:
                    # S-vertex: 2*u = 2*u - 2*delta
                    dualvar[v] -= delta
                elif label.get(inblossom[v]) == 2:
                    # T-vertex: 2*u = 2*u + 2*delta
                    dualvar[v] += delta
            
            for b in blossomdual:
                if blossomparent[b] is None:
                    if label.get(b) == 1:
                        # S-blossom: z = z + 2*delta
                        blossomdual[b] += delta
                    elif label.get(b) == 2:
                        # T-blossom: z = z - 2*delta
                        blossomdual[b] -= delta
            
            # Take action based on the type of update
            if deltatype == 1:
                break  # No further improvement possible
            elif deltatype == 2 or deltatype == 3:
                # Use the least-slack edge to continue search
                v, w = deltaedge
                assert label[inblossom[v]] == 1
                allowedge[(v, w)] = allowedge[(w, v)] = True
                queue.append(v)
            elif deltatype == 4:
                # Expand the least-z blossom
                expandBlossom(deltablossom, False)
        
        if not augmented:
            # No more augmenting paths can be found
            break
        
        # Expand any S-blossoms with zero dual
        for b in list(blossomdual.keys()):
            if b not in blossomdual:
                continue  # Already expanded
            if blossomparent[b] is None and label.get(b) == 1 and blossomdual[b] == 0:
                expandBlossom(b, True)
    
    # Return the matching
    return matching_dict_to_set(mate)