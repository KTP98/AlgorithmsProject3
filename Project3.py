import sys
import heapq
from itertools import groupby
from collections import defaultdict, deque

sys.setrecursionlimit(10 ** 6)




# Graph ADT used in the project

class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(list)
        self.distances = {}

    def add_node(self, value):
        self.nodes.add(value)

    def add_edge(self, from_node, to_node, distance):
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.distances[(from_node, to_node)] = distance

    def print_graph(self):
        for key, value in self.edges.items():
            print(key, ' : ', value)


# http://eddmann.com/posts/depth-first-search-and-breadth-first-search-in-python/

# An iterative DFS implementation 
def dfs(graph, start, visited=None):
    
    # Tracking total number of recursive calls
    global t 
    global path
    
    t += 1

    # Initialization with empty set
    if visited is None:
        visited = set() 

    # Mark start visited and add it to visited
    visited.add(start)
    path.append(start)
    
    # For key in adjancency list set of start but
    # not yet visited visit the key
 
    for key in graph.edges[start] - visited: # Python suports set subtraction 
        if key not in path:
            dfs(graph, key, visited) # DFS recursive call
        
    return visited


def bfs(graph, start):
    visited, queue = set(), [start]
    p =[]
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            p.append(vertex)
            queue.extend(graph.edges[vertex] - visited)
    return p


# find the paths in a given graph using BFS
def bfs_paths(graph, start, goal):
    queue = [(start, [start])]
    while queue:
        (vertex, path) = queue.pop(0)
        for next in graph.edges[vertex] - set(path):
            if next == goal:
                yield path + [next]               
            else:
                queue.append((next, path + [next]))
                
                
# find the paths in a given graph using BFS
def dfs_paths(graph, start, goal):
    stack = [(start, [start])]
    while stack:
        (vertex, path) = stack.pop()
        for next in graph.edges[vertex] - set(path):
            if next == goal:
                yield path + [next]
            else:
                stack.append((next, path + [next]))





class Tracker(object):
    """Keeps track of the current time, current source, component leader,
    finish time of each node and the explored nodes.
    
    'self.leader' is informs of {node: leader, ...}."""

    def __init__(self):
        self.current_time = 0
        self.current_source = None
        self.leader = {}
        self.finish_time = {}
        self.explored = set()


def dfst(graph_dict, node, tracker):
    """Inner loop explores all nodes in a SCC. Graph represented as a dict,
    {tail: [head_list], ...}. Depth first search runs recursively and keeps
    track of the parameters"""

    tracker.explored.add(node)
    tracker.leader[node] = tracker.current_source
    for head in graph_dict[node]:
        if head not in tracker.explored:
            dfst(graph_dict, head, tracker)
    tracker.current_time += 1
    tracker.finish_time[node] = tracker.current_time


def dfs_loop(graph_dict, nodes, tracker):
    """Outer loop checks out all SCCs. Current source node changes when one
    SCC inner loop finishes."""

    for node in nodes:
        if node not in tracker.explored:
            tracker.current_source = node
            dfst(graph_dict, node, tracker)


def graph_reverse(graph):
    """Given a directed graph in forms of {tail:[head_list], ...}, compute
    a reversed directed graph, in which every edge changes direction."""

    reversed_graph = defaultdict(list)
    for tail, head_list in graph.items():
        for head in head_list:
            reversed_graph[head].append(tail)
    return reversed_graph


def scc(graph):
    """First runs dfs_loop on reversed graph with nodes in decreasing order,
    then runs dfs_loop on original graph with nodes in decreasing finish
    time order(obtained from first run). Return a dict of {leader: SCC}."""

    out = defaultdict(list)
    tracker1 = Tracker()
    tracker2 = Tracker()
    nodes = set()
    reversed_graph = graph_reverse(graph)
    for tail, head_list in graph.items():
        nodes |= set(head_list)
        nodes.add(tail)
    nodes = sorted(list(nodes), reverse=True)
    dfs_loop(reversed_graph, nodes, tracker1)
    sorted_nodes = sorted(tracker1.finish_time,
                          key=tracker1.finish_time.get, reverse=True)
    dfs_loop(graph, sorted_nodes, tracker2)
    for lead, vertex in groupby(sorted(tracker2.leader, key=tracker2.leader.get),
                                key=tracker2.leader.get):
        out[lead] = list(vertex)
    return out


def dijkstra(graph, initial):
  visited = {initial: 0}
  path = {}

  nodes = set(graph.nodes)

  while nodes: 
    min_node = None
    for node in nodes:
      if node in visited:
        if min_node is None:
          min_node = node
        elif visited[node] < visited[min_node]:
          min_node = node

    if min_node is None:
      break

    nodes.remove(min_node)
    current_weight = visited[min_node]

    for edge in graph.edges[min_node]:
      weight = current_weight + graph.distance[(min_node, edge)]
      if edge not in visited or weight < visited[edge]:
        visited[edge] = weight
        path[edge] = min_node

  return visited, path



# found code for printing shortest path from https://gist.github.com/mdsrosa/c71339cb23bc51e711d8
def shortest_path(graph, origin, destination):
    visited, paths = dijkstra(graph, origin)
    print("Shortest Path Tree: " + str(visited))
    print("\n")
    print("Paths: " + str(paths))
    print("\n")
    full_path = deque()
    _destination = paths[destination]

    while _destination != origin:
        full_path.appendleft(_destination)
        _destination = paths[_destination]

    full_path.appendleft(origin)
    full_path.append(destination)

    return visited[destination], list(full_path)


# Prim's algorithm produce a MST
def prim( nodes, edges ):
    nodes = list(nodes)
    conn = defaultdict( list )
    #for n1,n2,c in edges:
    #    conn[ n1 ].append( (c, n1, n2) )
    #    conn[ n2 ].append( (c, n2, n1) )
    for key in edges:
        conn[key[0]].append((edges[key], key[0], key[1]))
        
    mst = []
    used = set( [nodes[ 0 ]] )
    usable_edges = conn[ nodes[0] ][:]
    heapq.heapify( usable_edges )
 
    while usable_edges:
        cost, n1, n2 = heapq.heappop( usable_edges )
        if n2 not in used:
            used.add( n2 )
            mst.append( ( n1, n2, cost ) )
 
            for e in conn[ n2 ]:
                if e[ 2 ] not in used:
                    heapq.heappush( usable_edges, e )
    return mst




# For question 1 we are to execute the DFS and BFS algorithms on the given graph
def question1():
    print("--------------------------------------------------------------------------")
    print("Question 1: Run DFS and BFS on the graph provided.")
    # DFS graph, disconnected 
    graph = Graph()

    graph.nodes = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'N', 'O', 'P'}

    graph.edges = {'A': set(['B', 'E', 'F']), 'B': set(['A', 'C', 'F']), 'C': set(['B', 'D', 'G']), 'D': set(['C', 'G']), \
                     'E': set(['A', 'F', 'I']), 'F': set(['A', 'B', 'E', 'I']), 'G': set(['C', 'D', 'J']), 'H': set(['K', 'L']), \
                     'I': set(['E', 'F', 'J', 'M']), 'J': set(['G', 'I']), 'K': set(['H', 'L', 'O']), 'L': set(['H', 'K', 'P']), \
                     'M': set(['I', 'N']), 'N': set(['M']), 'O': set(['K']), 'P': set(['L'])}

    global t
    global path
    path = []
    t = 0
    v = dfs(graph, 'A')
    print("DFS output starting with A:")
    print("Path: ", path)
    print("\n")
    path = []
    t = 0
    v = dfs(graph, 'H')
    print("DFS output starting with H:")
    print("Path: ", path)     
    print("\n")
    v = bfs(graph, 'A')
    print("BFS output starting with A:")
    print("Path: ", v)
    print("\n")
    v = bfs(graph, 'H')
    print("BFS output starting with H:")
    print("Path: ", v)
    
    
    print("--------------------------------------------------------------------------")
    print("\n")
    
    

def question2():
    print("--------------------------------------------------------------------------")
    print("Question 2: Find all of the paths between two nodes using DFS and BFS")
    
    # DFS graph, disconnected 
    graph = Graph()

    graph.nodes = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'N', 'O', 'P'}

    graph.edges = {'A': set(['B', 'E', 'F']), 'B': set(['A', 'C', 'F']), 'C': set(['B', 'D', 'G']), 'D': set(['C', 'G']), \
                     'E': set(['A', 'F', 'I']), 'F': set(['A', 'B', 'E', 'I']), 'G': set(['C', 'D', 'J']), 'H': set(['K', 'L']), \
                     'I': set(['E', 'F', 'J', 'M']), 'J': set(['G', 'I']), 'K': set(['H', 'L', 'O']), 'L': set(['H', 'K', 'P']), \
                     'M': set(['I', 'N']), 'N': set(['M']), 'O': set(['K']), 'P': set(['L'])}
        
    paths = tuple(bfs_paths(graph, 'A', 'N'))
    print('The paths from A to N using BFS are: ')
    print(paths)
    print("\n")

    paths = tuple(bfs_paths(graph, 'H', 'O'))
    print('The paths from H to O using BFS are: ')
    print(paths) 
    print("\n")
    
    paths = tuple(dfs_paths(graph, 'A', 'N'))
    print('The paths from A to N using DFS are:' )
    print(paths)
    print("\n")
    
    paths = tuple(dfs_paths(graph, 'H', 'O'))
    print('The paths from H to O using DFS are:' )
    print(paths)
    
    print("--------------------------------------------------------------------------")
    print("\n")
    
    
 
def question3():
    print("--------------------------------------------------------------------------")
    print("Question 3: Find all the strongly connected components in the graph provided.")
    
    dgraph = Graph() 

    dgraph.nodes = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'}
    
    dgraph.edges = {'1': ['3'], '2': ['1'], '3': ['2', '5'], '4': ['1', '2', '12'], '5': ['6', '8'], '6': ['7', '8', '10'], 
                    '7': ['10'], '8': ['9', '10'], '9': ['5', '11'], '10': ['9', '11'], '11': ['12'], '12': []}
    
    groups = scc(dgraph.edges)
   
    top_5 = heapq.nlargest(5, groups, key=lambda x: len(groups[x]))
    result = []
    for i in range(5):
        try:
            result.append(len(groups[top_5[i]]))
        except:
            result.append(0)
            
    for key in groups:
        print(groups[key])

    print("--------------------------------------------------------------------------")
    print("\n")



def question4():
    print("--------------------------------------------------------------------------")
    print("Question 4: Apply Dijkstra's algorithm to the graph provided.")
    
    graph = Graph()
    graph.nodes = {'A' ,'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'}
    graph.edges = {'A': ['B', 'C', 'D'], 'B': ['A', 'C', 'F', 'H'], 'C': ['A', 'B', 'D', 'E', 'F'], 'D': ['A', 'C', 'E', 'I'], \
                   'E': ['C', 'D', 'F', 'G'], 'F': ['B', 'C', 'E', 'G', 'H'], 'G': ['E', 'F', 'H', 'I'], 'H': ['B', 'F', 'G', 'I'], \
                   'I': ['D', 'G', 'H']}
    graph.distance = {('A', 'B'):22, ('A', 'C'):9, ('A', 'D'):12, ('B', 'C'):35, ('B', 'H'):34, ('B', 'F'):36, ('C', 'F'):42, \
                        ('B', 'A'):22, ('C', 'A'):9, ('D', 'A'):12, ('C', 'B'):35, ('H', 'B'):34, ('F', 'B'):36, ('F', 'C'):42, \
                        ('C', 'D'):4, ('C' ,'E'):65, ('D', 'E'):33, ('F', 'E'):18, ('F', 'G'):39, ('F', 'H'):24, ('E', 'G'):23, \
                        ('D', 'C'):4, ('E' ,'C'):65, ('E', 'D'):33, ('E', 'F'):18, ('G', 'F'):39, ('H', 'F'):24, ('G', 'E'):23, \
                        ('G', 'H'):25, ('G', 'I'):21, ('H', 'I'):19, ('D', 'I'):30, \
                        ('H', 'G'):25, ('I', 'G'):21, ('I', 'H'):19, ('I', 'D'):30}
    print("Shortest path from A to G (cost, [path]): " + str(shortest_path(graph, 'A', 'G')))

    print("--------------------------------------------------------------------------")
    print("\n")



def question5():
    print("--------------------------------------------------------------------------")
    print("Question 5: Print a Minimum Spanning Tree from the given weighted graph.")
    
    wgraph = Graph()

    wgraph.nodes = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'}

    wgraph.edges = {'A': ['B', 'C', 'D'], 'B': ['A', 'C', 'F', 'H'], 'C': ['A', 'B', 'D', 'E', 'F'], 'D': ['A', 'C', 'E', 'I'], \
                    'E': ['C', 'D', 'F', 'G'], 'F': ['B', 'C', 'E', 'G', 'H'], 'G': ['E', 'F', 'H', 'I'], 'H': ['B', 'F', 'G', 'I'], \
                    'I': ['D', 'G', 'H']}

    # if the edges have a weight associated, create a tuple ('from - to', weight) example ('AB', 10)
    wgraph.distances = {('A', 'B'):22, ('A', 'C'):9, ('A', 'D'):12, ('B', 'C'):35, ('B', 'H'):34, ('B', 'F'):36, ('C', 'F'):42, \
                        ('B', 'A'):22, ('C', 'A'):9, ('D', 'A'):12, ('C', 'B'):35, ('H', 'B'):34, ('F', 'B'):36, ('F', 'C'):42, \
                        ('C', 'D'):4, ('C' ,'E'):65, ('D', 'E'):33, ('F', 'E'):18, ('F', 'G'):39, ('F', 'H'):24, ('E', 'G'):23, \
                        ('D', 'C'):4, ('E' ,'C'):65, ('E', 'D'):33, ('E', 'F'):18, ('G', 'F'):39, ('H', 'F'):24, ('G', 'E'):23, \
                        ('G', 'H'):25, ('G', 'I'):21, ('H', 'I'):19, ('D', 'I'):30, \
                        ('H', 'G'):25, ('I', 'G'):21, ('I', 'H'):19, ('I', 'D'):30}


    
    print("Prim: ", prim(wgraph.nodes, wgraph.distances))
    
    print("--------------------------------------------------------------------------")
    print("\n")
        

def main():
    
    question1()
    question2()
    question3()
    question4()
    question5()
    
if __name__ == '__main__':
    main()
