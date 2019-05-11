from utils import dis
from utils import *
from data_parser import data_pt
import heapq
import math
from data import ExperimentData
import copy

agent_match = {} # dict mapping agent index to facility index matched
coalition_size = 0 # Hopefully not 1 as it can cause issues
open_facilities = set()
#match_add_agents = 0


#TODO
class agent_node:
    #index: int # agent index
    #point: data_pt # agent point
    #dist: float # distance from facility to agent
    def __init__(self, index, point, dist):
        self.index = index
        self.point = point
        self.dist = dist
        self.prev = None
        self.next = None # contains the reference to the next node


#@dataclass(order=True)
class facility:

    def __init__(self, index, data_list, distances):
        self.index = index
        self.point = data_list[index]
        self.agent_list_len = len(data_list)

        # Generate distances and sort agents by distance
        agents = []
        for i in range(len(data_list)):
            agents.append(agent_node(i, data_list[i], distances[self.point][data_list[i]]))
        agents.sort(key=lambda agent: agent.dist)

        # Convert sorted list to linked list
        agent_list_end = agents[0]
        agents[0].prev = None
        self.agent_list = agent_list_end
        for i in range(1, len(data_list)):
            agent_list_end.next = agents[i]
            agents[i].prev = agent_list_end
            agent_list_end = agents[i]
            if i == coalition_size - 1: # Initially point at S-1
                self.cur_pointer = agent_list_end
                # self.cur_pointer_pos = i

    #def __cmp__(self, other):
    def __lt__(self, other):
        if self.cur_pointer.dist == other.cur_pointer.dist:
            return self.index - other.index < 0
        return self.cur_pointer.dist - other.cur_pointer.dist < 0

    def __str__(self):
        #return "%d %.2f" % (self.index, self.cur_pointer.dist)
        str = []
        agent = self.agent_list
        while agent is not None:
            str.append("(%d, %.2f)" % (agent.index, agent.dist))
            agent = agent.next
        return "%d %.2f %s" % (self.index, self.cur_pointer.dist, "[" + ', '.join(str) + "]")

    def remove_node(self, agent_node, update_pointer = True):
        if update_pointer and self.cur_pointer:
            self.cur_pointer = self.cur_pointer.next
            # if self.cur_pointer is None:
                # self.cur_pointer_pos = -1
        if agent_node == self.agent_list:
            self.agent_list = agent_node.next
            if self.agent_list:
                self.agent_list.prev = None
        else:
            prev = agent_node.prev
            next = agent_node.next
            prev.next = next
            if next:
                next.prev = prev

    def remove_matched_agents(self, start, update_pointer = True):
        """
        Remove all agents matched to other facilities from start to cur_pointer.
        Stops when every agent is considered and cur_pointer itself is unmatched or is None.
        # Consider any agent up to the old cur_pointer, but if a new cur_pointer is unmatched, it stops there.
        :param match_here: whether unmatched agents should be matched to this facility
        :param update_pointer: whether pointer should be moved back by 1 per matched agent
        :return: bool of whether any matched agents were found and removed
        """
        found_match = False
        while start is not None and (start != self.cur_pointer or start.index in agent_match): # stops when start is cur_pointer and not matched
            if start.index in agent_match:
                found_match = True
                next = start.next
                self.remove_node(start, update_pointer)
                start = next
            else:
                start = start.next
        return found_match

    def process(self):
        """
        Perform actions once the facility is removed from PriorityQueue.
        :return: bool representing whether the facility should be added back to pq
        """
        if self.index in open_facilities: # Already open
            found_match = self.remove_matched_agents(self.cur_pointer, True)
            if not found_match: # match agent here
                agent_match[self.cur_pointer.index] = self.index
                #print("Agent %s matched to facility %d with distance %.2f" % (self.cur_pointer.index, self.index, self.cur_pointer.dist))
                self.cur_pointer = self.cur_pointer.next
                #global match_add_agents
                #match_add_agents += 1
        else: # Not open yet
            found_match = self.remove_matched_agents(self.agent_list, True)
            if not found_match: # all S agents before cur_pointer (inclusive) are unmatched, open facility
                open_facilities.add(self.index)
                agent = self.agent_list
                #agents_matched = [] # for prints
                while agent != self.cur_pointer.next:
                    if agent.index in agent_match: # bug in code
                        raise RuntimeError
                    agent_match[agent.index] = self.index
                    #agents_matched.append(agent.index)
                    agent = agent.next
                self.cur_pointer = self.cur_pointer.next
                #print("Opened facility %d" % self.index)
                #print("Agents %s matched to facility %d" % (','.join(str(x) for x in agents_matched), self.index))
        return self.cur_pointer is not None


def ball_growing_procedure(data_list, k, rho=1, distances=None,
                           client_list=None, remaining_k=-1):
    """
    Performs the ball-growing algorithm that gives 2.414-approx.
    :param data_list: list of data points (serve as both agents and facilities)
    :param k: number of centers to be opened (determines blocking coalition size)
    :param rho: parameter that gives size of blocking coalition (see utils)
    :param client_list: list of indexes of available facilities to choose
        default: 0, 1, ..., len(data_list)-1
    :param remaining_k: number of remaining centers to be opened
        default: k
    :return: list of indexes of opened facilities
    """
    global agent_match
    global coalition_size
    global open_facilities
    #global match_add_agents
    agent_match = {}
    open_facilities = set()
    #
    # match_add_agents = 0

    if not distances:
        distances = calc_distances(data_list)
    if not client_list:
        client_list = range(len(data_list))
    if remaining_k == -1:
        remaining_k = k
    num = len(data_list)
    coalition_size = math.ceil(num / k)
    pq = []
    for i in client_list:
        pq.append(facility(i, data_list, distances))
    heapq.heapify(pq)

    while len(pq) > 0 and len(open_facilities) < remaining_k:
        cur_facility = heapq.heappop(pq)
        push_back = cur_facility.process()
        if push_back:
            heapq.heappush(pq, cur_facility)

    facility_indexes = list(open_facilities)
    return facility_indexes


def ball_growing_repeated(data_list, k, alpha=1, distances=None):
    """
    Runs the ball-growing algorithm repeatedly, until exactly k centers are opened.
    :param data_list: list of data points (serve as both agents and facilities)
    :param k: number of centers to be opened (determines blocking coalition size)
    :param alpha: parameter that gives size of blocking coalition (see utils)
    :return: - k-center objective value
             - k-median objective value
             - minimum beta value
    """

    if not distances:
        distances = calc_distances(data_list)
    remaining_indexes = set(range(len(data_list)))
    facility_indexes = []
    remaining_k = k

    while remaining_k > 0:
        facility_indexes += ball_growing_procedure(data_list, k, alpha, distances=distances, client_list=list(remaining_indexes), remaining_k=remaining_k)
        remaining_indexes = remaining_indexes.difference(set(facility_indexes))
        remaining_k = k - len(facility_indexes)

    selected_center = []
    for facility in facility_indexes:
        selected_center.append(copy.deepcopy(data_list[facility]))

    return selected_center