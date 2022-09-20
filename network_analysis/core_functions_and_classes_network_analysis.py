# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 08:50:02 2021

@author: Sebastian Molina-Obando
"""

from collections import defaultdict

#Initializing the Graph Class
class Graph:
    def __init__(self):
        self.nodes = set()
        self.nodes_num = []
        self.edges = defaultdict(list)
        self.distances = {}

    
    def addNode(self,value):
        self.nodes.add(value)
    def addNode_num(self, value):
        self.nodes_num.append(value)
    
    def addEdge(self, fromNode, toNode, distance):
        self.edges[fromNode].append(toNode)
        self.distances[(fromNode, toNode)] = distance
        

        

#Implementing Dijkstra's Algorithm

#Single Source Shortest Path (SSSP) - Dijkstra's algorithm
#code source: https://pythonwife.com/dijkstras-algorithm-in-python/

def dijkstra(graph, initial):
    visited = {initial : 0}
    path = defaultdict(list)

    nodes = set(graph.nodes)

    while nodes:
        minNode = None
        for node in nodes:
            if node in visited:
                if minNode is None:
                    minNode = node
                elif visited[node] < visited[minNode]:
                    minNode = node
        if minNode is None:
            break

        nodes.remove(minNode)
        currentWeight = visited[minNode]

        for edge in graph.edges[minNode]:
            weight = currentWeight + graph.distances[(minNode, edge)]
            if edge not in visited or weight < visited[edge]:
                visited[edge] = weight
                path[edge].append(minNode)
    
    return visited, path