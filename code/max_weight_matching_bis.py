from collections import deque


def max_weight_matching(G, weight="weight"):
    """Compute maximum weight matching in bipartite graph using Hungarian method."""
    mate = {}
    u_nodes = set(G.keys())
    v_nodes = set()
    for u in G:
        for v in G[u]:
            v_nodes.add(v)
    v_nodes = list(v_nodes - u_nodes)
    dual_u = {u: 0 for u in u_nodes}
    dual_v = {v: 0 for v in v_nodes}
    parent = {}

    def slack(u, v):
        return dual_u[u] + dual_v[v] - G[u][v]

    for u in u_nodes:
        if u not in mate:
            queue = deque([u])
            in_tree = {u: True}
            prev = {u: None}
            min_slack = {}
            for v in v_nodes:
                min_slack[v] = slack(u, v)
                parent[v] = u

            matched = False
            while not matched and queue:
                u1 = queue.popleft()
                for v in G[u1]:
                    if min_slack[v] == 0:
                        if v not in mate:
                            # Augment path
                            while u1 is not None:
                                u0 = parent[v]
                                mate[v] = u1
                                u1, v = mate[u1], u0
                            matched = True
                            break
                        else:
                            u2 = mate[v]
                            if u2 not in in_tree:
                                in_tree[u2] = True
                                queue.append(u2)
                                prev[u2] = v
                                for v2 in v_nodes:
                                    new_slack = slack(u2, v2)
                                    if min_slack[v2] > new_slack:
                                        min_slack[v2] = new_slack
                                        parent[v2] = u2
                if not matched:
                    delta = min(min_slack.values())
                    for u_node in in_tree:
                        dual_u[u_node] -= delta
                    for v in v_nodes:
                        if min_slack[v] == delta:
                            dual_v[v] += delta
                        else:
                            min_slack[v] -= delta
                    for v in v_nodes:
                        if min_slack[v] == 0:
                            if v not in mate:
                                # Augment path
                                u1 = parent[v]
                                while u1 is not None:
                                    u0 = parent[v]
                                    mate[v] = u1
                                    u1, v = mate[u1], u0
                                matched = True
                                break
                            else:
                                u2 = mate[v]
                                if u2 not in in_tree:
                                    in_tree[u2] = True
                                    queue.append(u2)
                                    prev[u2] = v
                                    for v2 in v_nodes:
                                        new_slack = slack(u2, v2)
                                        if min_slack[v2] > new_slack:
                                            min_slack[v2] = new_slack
                                            parent[v2] = u2
    return mate
