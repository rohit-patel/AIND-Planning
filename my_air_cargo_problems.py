from aimacode.logic import PropKB
from aimacode.planning import Action
from aimacode.search import (
    Node, Problem,
)
from aimacode.utils import expr
from lp_utils import (
    FluentState, encode_state, decode_state,
)
from my_planning_graph import PlanningGraph

from functools import lru_cache


class AirCargoProblem(Problem):
    def __init__(self, cargos, planes, airports, initial: FluentState, goal: list):
        """

        :param cargos: list of str
            cargos in the problem
        :param planes: list of str
            planes in the problem
        :param airports: list of str
            airports in the problem
        :param initial: FluentState object
            positive and negative literal fluents (as expr) describing initial state
        :param goal: list of expr
            literal fluents required for goal test
        """
        self.state_map = initial.pos + initial.neg
        self.initial_state_TF = encode_state(initial, self.state_map)
        Problem.__init__(self, self.initial_state_TF, goal=goal)
        self.cargos = cargos
        self.planes = planes
        self.airports = airports
        self.actions_list = self.get_actions()

    def get_actions(self):
        """
        This method creates concrete actions (no variables) for all actions in the problem
        domain action schema and turns them into complete Action objects as defined in the
        aimacode.planning module. It is computationally expensive to call this method directly;
        however, it is called in the constructor and the results cached in the `actions_list` property.

        Returns:
        ----------
        list<Action>
            list of Action objects
        """

        def load_actions():
            """Create all concrete Load actions and return a list

            :return: list of Action objects
            """
            
            all_cargos_planes_airports = ((cargo,plane, airport) for cargo in self.cargos for plane in self.planes for airport in self.airports)
            load_actions_list = []
            for cargo,plane,airport in all_cargos_planes_airports:
                    precond_pos = [expr("At({}, {})".format(cargo, airport)),expr("At({}, {})".format(plane, airport))
                                   ]
                    precond_neg = []
                    effect_add = [expr("In({}, {})".format(cargo, plane))]
                    effect_rem = [expr("At({}, {})".format(cargo, airport))]
                    load = Action(expr("Load({}, {}, {})".format(cargo, plane, airport)),
                                 [precond_pos, precond_neg],
                                 [effect_add, effect_rem])
                    load_actions_list.append(load)    
            return load_actions_list
            
        def unload_actions():
            """Create all concrete Unload actions and return a list

            :return: list of Action objects
            """
            all_cargos_planes_airports = ((cargo,plane, airport) for cargo in self.cargos for plane in self.planes for airport in self.airports)
            unload_actions_list = []
            for cargo,plane,airport in all_cargos_planes_airports:
                    precond_pos = [expr("In({}, {})".format(cargo, plane)),expr("At({}, {})".format(plane, airport))
                                   ]
                    precond_neg = []
                    effect_add = [expr("At({}, {})".format(cargo, airport))]
                    effect_rem = [expr("In({}, {})".format(cargo, plane))]
                    unload = Action(expr("Unload({}, {}, {})".format(cargo, plane, airport)),
                                 [precond_pos, precond_neg],
                                 [effect_add, effect_rem])
                    unload_actions_list.append(unload)    
            return unload_actions_list

        def fly_actions():
            """Create all concrete Fly actions and return a list

            :return: list of Action objects
            """
            flys = []
            for fr in self.airports:
                for to in self.airports:
                    if fr != to:
                        for p in self.planes:
                            precond_pos = [expr("At({}, {})".format(p, fr)),
                                           ]
                            precond_neg = []
                            effect_add = [expr("At({}, {})".format(p, to))]
                            effect_rem = [expr("At({}, {})".format(p, fr))]
                            fly = Action(expr("Fly({}, {}, {})".format(p, fr, to)),
                                         [precond_pos, precond_neg],
                                         [effect_add, effect_rem])
                            flys.append(fly)
            return flys

        return load_actions() + unload_actions() + fly_actions()

    def actions(self, state: str) -> list:
        """ Return the actions that can be executed in the given state.

        :param state: str
            state represented as T/F string of mapped fluents (state variables)
            e.g. 'FTTTFF'
        :return: list of Action objects
        """
        possible_actions = []
        fluentstate = decode_state(state, self.state_map)
        positive_fluent_set = set(fluentstate.pos)
        negative_fluent_set = set(fluentstate.neg)
        for action in self.get_actions():
            if set(action.precond_pos).issubset(positive_fluent_set) and set(action.precond_neg).issubset(negative_fluent_set):
                possible_actions.append(action)         
        return possible_actions

    def result(self, state: str, action: Action):
        """ Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).

        :param state: state entering node
        :param action: Action applied
        :return: resulting state after action
        """
        positive_fluent_set = set(self.state_map[i] for i in range(len(self.initial)) if state[i]=='T')
        negative_fluent_set = set(self.state_map[i] for i in range(len(self.initial)) if state[i]=='F')
        updated_positive_fluent_set = set(action.effect_add).union(positive_fluent_set) - set(action.effect_rem)
        updated_negative_fluent_set = set(action.effect_rem).union(negative_fluent_set) - set(action.effect_add)
        new_state = FluentState(updated_positive_fluent_set,updated_negative_fluent_set)
        return encode_state(new_state, self.state_map)

    def goal_test(self, state: str) -> bool:
        """ Test the state to see if goal is reached

        :param state: str representing state
        :return: bool
        """
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        for clause in self.goal:
            if clause not in kb.clauses:
                return False
        return True

    def h_1(self, node: Node):
        # note that this is not a true heuristic
        h_const = 1
        return h_const

    @lru_cache(maxsize=8192)
    def h_pg_levelsum(self, node: Node):
        """This heuristic uses a planning graph representation of the problem
        state space to estimate the sum of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
        """
        pg = PlanningGraph(self, node.state)
        pg_levelsum = pg.h_levelsum()
        return pg_levelsum

    @lru_cache(maxsize=8192)
    def h_ignore_preconditions(self, node: Node):
        """This heuristic estimates the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions by ignoring the preconditions required for an action to be
        executed.
        """
        def get_relevant_actions_sorted(action_set):
            relevant_actions = []
            for action in action_set:
                effects_in_goal = set(action.effect_add).intersection(states_to_achieve)
                if bool(effects_in_goal):
                    relevant_actions.append([action,len(effects_in_goal)])
            return sorted(relevant_actions, key = lambda x: x[1])

        count = 0
        states_to_achieve = set(self.goal) - set(self.state_map[i] for i in range(len(self.initial)) if node.state[i]=='T')
        action_set = self.get_actions()      
        
        #Greedy algorithm to approximate minimum number of unconstrained moves
        while states_to_achieve:
            sorted_relevant_actions = get_relevant_actions_sorted(action_set)
            states_to_achieve = states_to_achieve - set(sorted_relevant_actions[0][0].effect_add)
            action_set = set([i for i in zip(*sorted_relevant_actions)][0])
            count +=1
        return count


def air_cargo_p1() -> AirCargoProblem:
    cargos = ['C1', 'C2']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]
    neg = [expr('At(C2, SFO)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('At(C1, JFK)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('At(P1, JFK)'),
           expr('At(P2, SFO)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p2() -> AirCargoProblem:
    cargos = ['C1', 'C2', 'C3']
    planes = ['P1', 'P2', 'P3']
    airports = ['JFK', 'SFO', 'ATL']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(C3, ATL)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           expr('At(P3, ATL)'),
           ]
    neg = []
    for cargo in cargos:
        for plane in planes:
            inexpression = expr('In({0},{1})'.format(cargo,plane))
            if not(inexpression in pos):
                neg.append(inexpression)
        for airport in airports:
            atexpression = expr('At({0},{1})'.format(cargo,airport))
            if not(atexpression in pos):
                neg.append(atexpression)
    for plane, airport in ((plane,airport) for plane in planes for airport in airports):
            atexpression = expr('At({0},{1})'.format(plane,airport))
            if not(atexpression in pos):
                neg.append(atexpression)
        
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C3, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p3() -> AirCargoProblem:
    cargos = ['C1', 'C2', 'C3', 'C4']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO', 'ATL', 'ORD']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(C3, ATL)'),
           expr('At(C4, ORD)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]
    neg = []
    for cargo in cargos:
        for plane in planes:
            inexpression = expr('In({0},{1})'.format(cargo,plane))
            if not(inexpression in pos):
                neg.append(inexpression)
        for airport in airports:
            atexpression = expr('At({0},{1})'.format(cargo,airport))
            if not(atexpression in pos):
                neg.append(atexpression)
    for plane, airport in ((plane,airport) for plane in planes for airport in airports):
            atexpression = expr('At({0},{1})'.format(plane,airport))
            if not(atexpression in pos):
                neg.append(atexpression)
        
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C3, JFK)'),
            expr('At(C4, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)
