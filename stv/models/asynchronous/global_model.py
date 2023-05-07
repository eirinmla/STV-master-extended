import itertools

from stv.parsers.formula_parser import AtlFormula, CtlFormula, UpgradeFormula, PathQuantifier, SimpleExpression, CoalitionExpression, SimpleExpressionOperator
from typing import List, Dict, Any, Set, Tuple
from enum import Enum
import time
from stv.models.asynchronous.global_state import GlobalState
from stv.models.asynchronous.local_model import LocalModel
from stv.models.asynchronous.local_transition import LocalTransition, SharedTransition
from stv.models import SimpleModel
from stv.logics.atl import ATLIrModel
from stv.comparing_strats import StrategyComparer
from stv.parsers import FormulaParser, TemporalOperator, UpgradeType


count = 0 

class LogicType(Enum):
    ATL = "ATL"
    CTL = "CTL"
    UCL = "UCL"


class GlobalModel:
    """
    Represents global model.

    :param local_models:
    :param reduction:

    :ivar _model:
    :ivar _local_models:
    :ivar _reduction:
    :ivar _persistent:
    :ivar _states:
    :ivar _agents_count:
    :ivar _states_dict:
    :ivar _stack1:
    :ivar _stack2:
    :ivar _G:
    :ivar coalition:
    :ivar _stack1_dict:
    :ivar _transitions_count
    """

    def __init__(self,
                 local_models: List[LocalModel], reduction: List[str],
                 bounded_vars: List[str], persistent: List[str],
                 coalition: List[str], goal: List[str],
                 logicType: LogicType, formula: str,
                 show_epistemic: bool, semantics: str, initial, name: str = ""):
        self._model: SimpleModel = None
        self._local_models: List[LocalModel] = local_models
        self._reduction: List[str] = reduction
        self._bounded_vars: Dict[str, str] = dict([(x.split(' ')[0], x.split(' ')[1]) for x in bounded_vars])
        # print(f"LOG: bounded_vars = {self._bounded_vars}")
        self._persistent: List[str] = persistent
        self._coalition: List[str] = coalition
        self._goal: List[str] = goal
        self._logicType = logicType
        self._formula = formula
        self._semantics = semantics
        self._initial = initial
        self._name = name
        if self.isAtl():
            self._formula_obj = self._parseAtlFormula()
        elif self.isCtl():
            self._formula_obj = self._parseCtlFormula()
        elif self.isUCL():
            self._formula_obj = self._parseUpgradeFormula()
        self._states: List[GlobalState] = []
        self._agents_count: int = 0
        self._states_dict: Dict[str, int] = dict()
        self._stack1: List[Any] = []
        self._stack2: List[int] = []
        self._G: List = []
        if self.isAtl():
            self.coalition: List = self._formula_obj.agents
        elif self.isCtl():
            self.coalition: List = self._getCtlCoalition()
        elif self.isUCL():
            self.coalition: List = self._formula_obj.agents
        self._stack1_dict: Dict[str, int] = dict()
        self._transitions_count: int = 0
        self._epistemic_states_dictionaries: List[Dict[str, Set[int]]] = []
        self._show_epistemic = show_epistemic

    def _parseAtlFormula(self):
        formula_parser = FormulaParser()
        return formula_parser.parseAtlFormula(self._formula)

    def _parseCtlFormula(self):
        formula_parser = FormulaParser()
        return formula_parser.parseCtlFormula(self._formula)
    
    def _parseUpgradeFormula(self):
        formula_parser = FormulaParser()
        return formula_parser.parseUpgradeFormula(self._formula)

    def _getCtlCoalition(self):
        coalition: List[str] = []
        if self._formula_obj.pathQuantifier == PathQuantifier.A:
            # Empty set of agents
            coalition.append("__ctl_pseudo_agent__")
        elif self._formula_obj.pathQuantifier == PathQuantifier.E:
            # Set of all agents
            for localModel in self._local_models:
                coalition.append(localModel.agent_name)
        return coalition

    def isAtl(self):
        return self._logicType == LogicType.ATL

    def isCtl(self):
        return self._logicType == LogicType.CTL
    
    def isUCL(self):
        return self._logicType == LogicType.UCL

    def get_agent(self):
        return self.agent_name_to_id(self.coalition[0])

    def get_coalition(self):
        result = []
        for a_str in self.coalition:
            result.append(self.agent_name_to_id(a_str))

        return result

    @property
    def formula(self):
        """Formula string"""
        return self._formula

    @property
    def model(self):
        """The model."""
        return self._model

    @property
    def local_models(self):
        return self._local_models

    @property
    def states_count(self):
        return len(self._states)

    @property
    def transitions_count(self):
        return self._transitions_count

    @property
    def name(self):
        return self._name

    def generate(self, reduction: bool = False):
        """
        Generates model.
        :param reduction: Should reductions be used.
        :return: None.
        """
        self._agents_count = len(self._local_models)
        self._epistemic_states_dictionaries: List[Dict[str, Set[int]]] = [{} for _ in range(self._agents_count)]
        self._model = SimpleModel(self._agents_count)
        self._add_index_to_transitions()
        # self._compute_dependent_transitions()
        self._compute_shared_transitions()
        self._coalition = self._formula_obj.agents
        self._model.coalition = self.agent_name_coalition_to_ids(self._coalition)
        if reduction:
            self._add_to_stack(GlobalState.initial_state(self._agents_count, self._initial))
            self._iter_por()
        else:
            if self._semantics == "asynchronous":
                self._compute_asynchronous()
            else:
                self._compute_synchronous()

        # self._model.states = self._states
        self._prepare_epistemic_relation()

    def generate_part(self, agents: List[int]):
        agent_id = 0
        for el in self._local_models[:]:
            if agent_id in agents:
                self._local_models.remove(el)

            agent_id += 1

        self._agents_count = len(self._local_models)
        self._model = SimpleModel(self._agents_count)
        self._add_index_to_transitions()
        self._compute_shared_transitions()
        if self._semantics == "asynchronous":
            self._compute_asynchronous()
        else:
            self._compute_synchronous()

    def generate_local_models(self):
        for local_model in self._local_models:
            local_model.generate()

    def _prepare_epistemic_relation(self):
        """
        Prepares epistemic relation for the model.
        Should be called after creating the model.
        :return: None
        """
        # for ep in self._epistemic_states:
        #     epistemic_state, state_id, agent_id = ep
        #     epistemic_state["actions"] = set()
        #     for action in self._model.get_partial_strategies(state_id, agent_id):
        #         epistemic_state["actions"].add(action)
        #     self._add_to_epistemic_dictionary(epistemic_state, state_id, agent_id)

        i = self.get_agent()
        for _, epistemic_class in self._epistemic_states_dictionaries[i].items():
            self.model.add_epistemic_class(i, epistemic_class)

    def _add_index_to_transitions(self):
        for agent_id in range(self._agents_count):
            for i in range(len(self._local_models[agent_id].transitions)):
                for j in range(len(self._local_models[agent_id].transitions[i])):
                    self._local_models[agent_id].transitions[i][j].i = i
                    self._local_models[agent_id].transitions[i][j].j = j

    def _compute_shared_transitions(self):
        replace = []
        for agent_id in range(self._agents_count):
            for i in range(len(self._local_models[agent_id].transitions)):
                for j in range(len(self._local_models[agent_id].transitions[i])):
                    transition = self._local_models[agent_id].transitions[i][j]
                    if not transition.shared:
                        continue

                    shared_transition = self._create_shared_transition(transition, agent_id)
                    shared_transition.transition_list.sort(key=lambda tran: tran.agent_id)
                    replace.append((agent_id, i, j, shared_transition))

        for rep in replace:
            agent_id, i, j, shared_transition = rep
            self._local_models[agent_id].transitions[i][j] = shared_transition

    def _create_shared_transition(self, transition: LocalTransition, agent_id: int) -> SharedTransition:
        shared_transition = SharedTransition(transition)
        for agent_id2 in range(self._agents_count):
            if agent_id == agent_id2:
                continue

            if self._local_models[agent_id2].has_action(transition.action):
                for transition2 in self._local_models[agent_id2].get_transitions():
                    if transition2.action == transition.action:
                        shared_transition.add_transition(transition2)
                        break

        return shared_transition

    def _available_transitions_in_state_for_agent(self, state: GlobalState, agent_id: int) -> List[LocalTransition]:
        """
        Computes a list of transitions available for the specified agent in the given state.
        :param state: Global state.
        :param agent_id: Agent identifier.
        :return: List of local transitions.
        """
        agent_state_id: int = state.local_states[agent_id]
        all_transitions: List[LocalTransition] = self._local_models[agent_id].private_transitions_from_state(
            agent_state_id)
        all_transitions += self._local_models[agent_id].shared_transitions_from_state(agent_state_id)
        return list(filter(lambda transition: transition.check_conditions(state), all_transitions))

    def _enabled_transitions_in_state(self, state: GlobalState) -> List[List[LocalTransition]]:
        """
        Computes all enabled transitions for the given global state.
        :param state:
        :return:
        """
        all_transitions = [self._available_transitions_in_state_for_agent(state, agent_id) for agent_id in
                           range(self._agents_count)]
        result = [self._enabled_transitions_for_agent(agent_id, all_transitions) for agent_id in
                  range(self._agents_count)]

        return result

    def _enabled_transitions_for_agent(self, agent_id: int, all_transitions: List[List[LocalTransition]]):
        """
        Computes enabled transitions for given agent based on the transitions from the global state.
        :param agent_id: Agent identifier.
        :param all_transitions: List containing all of the transitions going out from specific global state.
        :return:
        """
        result = []
        for transition in all_transitions[agent_id]:
            if not transition.shared:
                result.append(transition)
                continue

            if self._check_if_shared_transition_is_enabled(transition, agent_id, all_transitions):
                result.append(transition)

        return result

    def _check_if_shared_transition_is_enabled(self, transition: LocalTransition, agent_id: int,
                                               all_transitions: List[List[LocalTransition]]) -> bool:
        is_ok = True
        for agent_id2 in range(len(self._local_models)):
            if agent_id2 == agent_id:
                continue

            if self._local_models[agent_id2].has_action(transition.action):
                is_ok = False
                for transition2 in all_transitions[agent_id2]:
                    if transition2.shared and transition2.action == transition.action:
                        is_ok = True
                        break

            if not is_ok:
                return False

        return True

    def _enabled_transitions_in_state_single_item_set(self, state: GlobalState) -> Set[Tuple[int, int, int]]:
        enabled = self._enabled_transitions_in_state(state)
        result = set()
        for agent_id in range(self._agents_count):
            for transition in enabled[agent_id]:
                result.add(transition.to_tuple())
                if not transition.shared:
                    continue
                for agent_id2 in range(agent_id + 1, self._agents_count):
                    i = 0
                    for transition2 in enabled[agent_id2]:
                        if transition2.shared and transition2.action == transition.action:
                            enabled[agent_id2].pop(i)
                            break
                        i += 1
        return result

    def _new_state_after_private_transition(self, state: GlobalState, transition: LocalTransition) -> GlobalState:
        agent_id = transition.agent_id
        new_state = GlobalState.copy_state(state, self._persistent)
        new_state.set_local_state(agent_id, self._local_models[agent_id].get_state_id(transition.state_to))
        new_state = self._copy_props_to_state(new_state, transition)
        return new_state

    def _check_correct_synchronous_transitions(self, transitions: List[LocalTransition]):
        agents = set()
        for transition in transitions:
            agent_id = transition.agent_id
            if agent_id in agents:
                return False

            agents.add(agent_id)

        return True

    def _new_state_after_synchronous_transitions(self, state: GlobalState,
                                                 transitions: List[LocalTransition]) -> GlobalState:
        new_state = GlobalState.copy_state(state, self._persistent)
        for transition in transitions:
            agent_id = transition.agent_id
            new_state.set_local_state(agent_id, self._local_models[agent_id].get_state_id(transition.state_to))
            new_state = self._copy_props_to_state(new_state, transition)

        return new_state

    def _new_state_after_shared_transition(self, state: GlobalState, actual_transition) -> Tuple[
        GlobalState, List[int]]:
        new_state = GlobalState.copy_state(state, self._persistent)
        agents = []
        for act_tran in actual_transition:
            new_state.set_local_state(act_tran[0], self._local_models[act_tran[0]].get_state_id(
                act_tran[1].state_to))
            new_state = self._copy_props_to_state(new_state, act_tran[1])
            agents.append(act_tran[0])
        return new_state, agents

    def _new_state_after_shared_transitions_list(self, state: GlobalState,
                                                 transitions: List[LocalTransition]) -> GlobalState:
        new_state = GlobalState.copy_state(state, self._persistent)
        for transition in transitions:
            new_state.set_local_state(transition.agent_id,
                                      self._local_models[transition.agent_id].get_state_id(transition.state_to))
            new_state = self._copy_props_to_state(new_state, transition)
        return new_state

    def _compute_next_for_state(self, state: GlobalState, current_state_id: int):
        all_transitions = self._enabled_transitions_in_state(state)
        visited = []
        for agent_id in range(len(self._local_models)):
            self._compute_next_for_state_for_agent(state, current_state_id, agent_id, visited, all_transitions)

    def _compute_synchronous_next_for_state(self, state: GlobalState, current_state_id: int):
        all_transitions = self._enabled_transitions_in_state(state)
        private_transitions = []
        for agent_id in range(len(self._local_models)):
            for tran in all_transitions[agent_id]:
                if not tran.shared:
                    private_transitions.append(tran)

        for i in range(2, len(private_transitions) + 1):
            for transitions in itertools.combinations(private_transitions, i):
                # TODO check if transitions dont change variables from other conditions
                # if not self._check_synchronous_transitions(transitions):
                #     continue
                if not self._check_correct_synchronous_transitions(transitions):
                    continue
                new_state = self._new_state_after_synchronous_transitions(state, transitions)
                new_state_id = self._add_state(new_state)
                self._add_synchronous_transitions(current_state_id, new_state_id, transitions)

    def _check_synchronous_transitions(self, transitions: List[LocalTransition]):
        conditions = set()
        props = set()
        for tran in transitions:
            if len(tran.conditions) > 0:
                conditions.add(tran.conditions[0][0])

            props.update(list(tran.props.keys()))

        for prop in props:
            if prop in conditions:
                return False
        for cond in conditions:
            if cond in props:
                return False

        return True

    def _compute_next_for_state_for_agent(self, state: GlobalState, current_state_id: int, agent_id: int,
                                          visited: List[str],
                                          all_transitions: List[List[LocalTransition]]):
        for transition in all_transitions[agent_id]:
            if transition.shared and transition.action not in visited:
                visited.append(transition.action)
                actual_transition = [(agent_id, transition)]
                for n_a_id in range(agent_id + 1, self._agents_count):
                    for n_tr in all_transitions[n_a_id]:
                        if n_tr.shared and n_tr.action == transition.action:
                            actual_transition.append((n_a_id, n_tr))
                            break
                new_state, agents = self._new_state_after_shared_transition(state, actual_transition)
                new_state_id = self._add_state(new_state)
                self._add_transition(current_state_id, new_state_id, transition)
            elif not transition.shared:
                new_state = self._new_state_after_private_transition(state, transition)
                new_state_id = self._add_state(new_state)
                self._add_transition(current_state_id, new_state_id, transition)

    def _copy_props_to_state(self, state: GlobalState, transition: LocalTransition) -> GlobalState:
        for prop in transition.props:
            op, val = transition.props[prop]
            self._check_bounded_vars(prop, val)

            if type(val) is str:
                if val[0] == "?":
                    prop_name = val[1:]
                    if prop_name in transition.props:
                        state.set_prop(prop, transition.props[prop_name][1])
                    elif prop_name in state.props:
                        state.set_prop(prop, state.props[prop_name])
                elif val[0] == "%":
                    prop_name = val[1:]
                    if prop_name in transition.props:
                        state.average_prop(prop, transition.props[prop_name][1])
                    elif prop_name in state.props:
                        state.average_prop(prop, state.props[prop_name])
                elif val[0] == "^":
                    prop_name = val[1:]
                    if prop_name in transition.props:
                        state.max_prop(prop, transition.props[prop_name][1])
                    elif prop_name in state.props:
                        state.max_prop(prop, state.props[prop_name])
                if op == "+":
                    prop_val = state.props[val]
                    state.change_prop(prop, prop_val)
            elif type(val) is int:
                if op == "+":
                    state.change_prop(prop, val)
                elif op == "-":
                    state.change_prop(prop, -val)
                else:
                    state.set_prop(prop, val)
            else:
                state.set_prop(prop, val)
        return state

    def _check_bounded_vars(self, prop_name, prop_val):
        if not self._bounded_vars:
            return
        if prop_name in self._bounded_vars:
            min_val, max_val = self._bounded_vars[prop_name].strip("{").strip("}").split('..')
            if prop_val < int(min_val) or prop_val > int(max_val):
                print(f"WARN: Assigning an int out of bound values in '{prop_name}={prop_val}'")

    def _state_find(self, state: GlobalState) -> int:
        if state.to_str() in self._states_dict:
            return self._states_dict[state.to_str()]

        return -1

    def _is_in_G(self, state: GlobalState) -> bool:
        for st in self._G:
            if st.equal(state):
                return True
        return False

    def _find_state_on_stack1(self, state: GlobalState) -> int:
        str_state: str = state.to_str()

        if str_state in self._stack1_dict:
            return self._stack1_dict[str_state]

        return -1

    def _add_to_stack(self, state: GlobalState) -> bool:
        str_state: str = state.to_str()

        if str_state in self._stack1_dict:
            return False
        else:
            self._stack1.append(state)
            self._stack1_dict[state.to_str()] = len(self._stack1) - 1
            return True

    def _pop_from_stack(self):
        self._stack1_dict[self._stack1[-1].to_str()] = -1
        self._stack1.pop()

    def _iter_por(self):
        """
        Iterative partial order reduction algorithm.
        :return: None.
        """
        dfs_stack: List[int] = [1]
        while len(dfs_stack) > 0:
            dfs: int = dfs_stack.pop()
            if dfs == 1:
                g: GlobalState = self._stack1[-1]
                # print("State:", g)
                reexplore: bool = False
                i: int = self._find_state_on_stack1(g)
                if i != -1 and i != len(self._stack1) - 1:
                    if len(self._stack2) == 0:
                        depth: int = 0
                    else:
                        depth: int = self._stack2[-1]
                    if i > depth:
                        reexplore = True
                    else:
                        self._pop_from_stack()
                        continue

                if not reexplore and self._is_in_G(g):
                    self._pop_from_stack()
                    continue

                self._G.append(g)
                g_state_id: int = self._add_state(g)
                E_g: Set[Tuple[int, int, int]] = set()
                en_g: Set[Tuple[int, int, int]] = self._enabled_transitions_in_state_single_item_set(g)

                # print("State:", g)
                # print("en_g:")
                # for tup in en_g:
                #     a: LocalTransition = self._local_models[tup[0]].transitions[tup[1]][tup[2]]
                #     print(a)
                # print()

                dfs_stack.append(-1)
                if len(en_g) > 0:
                    if not reexplore:
                        E_g = self._ample(g)

                    if len(E_g) == 0:
                        E_g = en_g

                    if E_g == en_g:
                        self._stack2.append(len(self._stack1))

                    for tup in E_g:
                        a: LocalTransition = self._local_models[tup[0]].transitions[tup[1]][tup[2]]
                        g_p: GlobalState = self._successor(g, a)
                        g_p_state_id: int = self._add_state(g_p)

                        # print("State g_p:", g_p)

                        self._add_transition(g_state_id, g_p_state_id, a)
                        if self._add_to_stack(g_p):
                            # print("State g_p:", g_p)
                            # print("State added")
                            dfs_stack.append(1)
            elif dfs == -1:
                if len(self._stack2) == 0:
                    depth: int = 0
                else:
                    depth: int = self._stack2[-1]
                if depth == len(self._stack1):
                    self._stack2.pop()
                self._pop_from_stack()

    def _ample(self, state: GlobalState) -> Set[Tuple[int, int, int]]:
        """
        Computes ample set for given state.
        :param state: Global state.
        :return: Ample set.
        """
        V = self._enabled_transitions_in_state_single_item_set(state)
        while len(V) > 0:
            alpha = V.pop()
            V.add(alpha)
            X = {alpha}
            U = {alpha}
            DIS = set()
            while len(X) > 0 and len(X.difference(V)) == 0:
                DIS.update(self._enabled_for_x(X))
                X = self._dependent_for_x(X, DIS, U)
                U.update(X)
            if len(X) == 0 and not self._check_for_k(state,
                                                     U):  # and not self._check_for_cycle(state, U):# and not self._check_for_k(state, U):
                return U
            V.difference_update(U)
        return set()

    def _check_for_cycle(self, state: GlobalState, X: Set[Tuple[int, int, int]]) -> bool:
        for tup in X:
            transition = self._local_models[tup[0]].transitions[tup[1]][tup[2]]
            successor_state = self._successor(state, transition)
            if self._find_state_on_stack1(successor_state) != -1:
                return True
        return False

    def _check_for_k(self, state: GlobalState, X: Set[Tuple[int, int, int]]) -> bool:
        for tup in X:
            transition = self._local_models[tup[0]].transitions[tup[1]][tup[2]]
            successor_state = self._successor(state, transition)
            # print(self._reduction)
            for agent_id in self.agent_name_coalition_to_ids(self._coalition):
                if state.local_states[agent_id] != successor_state.local_states[agent_id]:
                    return True

                for prop in self._reduction:
                    if prop in state.props and prop not in successor_state.props:
                        return True
                    if prop not in state.props and prop in successor_state.props:
                        return True
                    if prop not in state.props:
                        continue
                    if state.props[prop] != successor_state.props[prop]:
                        return True
        return False

    def _enabled_for_x(self, X: Set[Tuple[int, int, int]]) -> Set[Tuple[int, int, int]]:
        result: Set[Tuple[int, int, int]] = set()

        for tup in X:
            transition = self._local_models[tup[0]].transitions[tup[1]][tup[2]]
            if isinstance(transition, SharedTransition):
                for transition2 in transition.transition_list:
                    for tr in self._local_models[transition2.agent_id].get_transitions():
                        if tr.state_from != transition2.state_from:
                            result.add(tr.to_tuple())
            else:
                for tr in self._local_models[transition.agent_id].get_transitions():
                    if tr.state_from != transition.state_from:
                        result.add(tr.to_tuple())

        return result

    def _dependent_for_x(self, X: Set[Tuple[int, int, int]], DIS: Set[Tuple[int, int, int]],
                         U: Set[Tuple[int, int, int]]) -> Set[Tuple[int, int, int]]:  # !!!!
        result = set()
        for tup in X:
            transition = self._local_models[tup[0]].transitions[tup[1]][tup[2]]
            if isinstance(transition, SharedTransition):
                for transition2 in transition.transition_list:
                    for tr in self._local_models[transition2.agent_id].get_transitions():
                        if tr.to_tuple() not in DIS and tr.to_tuple() not in U:
                            result.add(tr.to_tuple())
            else:
                for tr in self._local_models[transition.agent_id].get_transitions():
                    if tr.to_tuple() not in DIS and tr.to_tuple() not in U:
                        result.add(tr.to_tuple())

        return result

    def _successor(self, state: GlobalState, transition: LocalTransition) -> GlobalState:
        if not isinstance(transition, SharedTransition):
            return self._new_state_after_private_transition(state, transition)
        else:
            return self._new_state_after_shared_transitions_list(state, transition.transition_list)

    def _add_state(self, state: GlobalState) -> int:
        # state.add_local_state_props(self._local_models)
        state_id = self._state_find(state)
        if state_id == -1:
            state_id = len(self._states)
            state.id = state_id
            self._states.append(state)
            self._states_dict[state.to_str()] = state_id
            self._model.states.append(state.to_obj())
            for agent_id in self._model.coalition:
                epistemic_state = self._get_epistemic_state(state, agent_id)
                # self._epistemic_states.append((epistemic_state, state_id, agent_id))
                self._add_to_epistemic_dictionary(epistemic_state, state_id, agent_id)

        state.id = state_id
        return state_id

    def _get_epistemic_state(self, state: GlobalState, agent_id: int) -> hash:
        """
        Compute epistemic representation of the given state.
        :param state: State to compute.
        :param agent_id: Id of the agent for which epistemic representation should be computed.
        :return: Epistemic representation of the given state.
        """

        if state.id == 0:
            return {'local_state': -1}

        epistemic_state = {'local_states': [state.local_states[agent_id]]}
        if agent_id in self.get_coalition():
            epistemic_state['local_states'] = [state.local_states[ag_id] for ag_id in self.get_coalition()]
        # if agent_id in self.coalition:
        #
        props = {}

        agent_name: str = self._local_models[agent_id].agent_name

        for prop in state.props:
            if prop[0:len(agent_name)] == agent_name:
                props[prop] = state.props[prop]
            elif prop in self._local_models[agent_id].local or prop in self._local_models[agent_id].interface:
                props[prop] = state.props[prop]
            elif agent_id in self.get_coalition():
                for ag_id in self.get_coalition():
                    if prop in self._local_models[ag_id].local:
                        props[prop] = state.props[prop]
                        break

        epistemic_state['props'] = props
        return epistemic_state

    def _add_to_epistemic_dictionary(self, state: hash, new_state_id: int, agent_id: int):
        """
        Adds state to the epistemic dictionary.
        :param state:
        :param new_state_id:
        :param agent_id:
        :return: None
        """
        state_str = ' '.join(str(state[e]) for e in state)
        if state_str not in self._epistemic_states_dictionaries[agent_id]:
            self._epistemic_states_dictionaries[agent_id][state_str] = {new_state_id}
        else:
            self._epistemic_states_dictionaries[agent_id][state_str].add(new_state_id)

    def _add_transition(self, state_from: int, state_to: int, transition: LocalTransition):
        self._transitions_count += 1
        self._model.add_transition(state_from, state_to, self._create_list_of_actions(transition))

    def _add_synchronous_transitions(self, state_from: int, state_to: int, transitions: List[LocalTransition]):
        self._transitions_count += 1
        actions = ['*' for _ in range(self._agents_count)]
        for tran in transitions:
            actions[tran.agent_id] = tran.action

        self._model.add_transition(state_from, state_to, actions)

    def _create_list_of_actions(self, transition: LocalTransition) -> List[str]:
        actions = ['*' for _ in range(self._agents_count)]

        if isinstance(transition, SharedTransition):
            for tr in transition.transition_list:
                actions[tr.agent_id] = tr.prot_name
        else:
            actions[transition.agent_id] = transition.prot_name

        return actions

    def _compute_asynchronous(self):
        """
        Compute global model using asynchronous semantics.
        :return:
        """
        state: GlobalState = GlobalState.initial_state(len(self._local_models), self._initial)
        self._add_state(state)
        current_state_id: int = 0
        while current_state_id < len(self._states):
            state = self._states[current_state_id]

            self._compute_next_for_state(state, current_state_id)

            current_state_id += 1

    def _compute_synchronous(self):
        """
        Compute global model using synchronous semantics.
        :return:
        """
        state: GlobalState = GlobalState.initial_state(len(self._local_models), self._initial)
        self._add_state(state)
        i: int = 0
        while i < len(self._states):
            state = self._states[i]
            current_state_id = i
            i += 1

            self._compute_next_for_state(state, current_state_id)
            self._compute_synchronous_next_for_state(state, current_state_id)

    def agent_name_to_id(self, agent_name: str) -> int:
        for agent_id in range(len(self._local_models)):
            if self._local_models[agent_id].agent_name == agent_name:
                return agent_id
        raise Exception("Cannot be an empty coalition or an agent who is not in the model.")

    def agent_name_coalition_to_ids(self, agent_names: List[str]) -> List[int]:
        agent_ids: List[int] = []
        for agent_name in agent_names:
            agent_ids.append(self.agent_name_to_id(agent_name))
        return agent_ids

    def print(self):
        for model in self._local_models:
            model.print()

    def set_coalition(self, coalition: List[str]):
        self.coalition = self.agent_name_coalition_to_ids(coalition)

    def get_winning_states(self) -> Set[int]:
        winning_states = set()
        for state in self._states:
            ok = False

            if "Voter1_vote" in state.props and state.props["Voter1_vote"] == 1:
                ok = True

            if ok:
                winning_states.add(state.id)
        return winning_states


    def verify_approximation(self, perfect_inf: bool):
        if perfect_inf:
            atl_model = self._model.to_atl_perfect()
        else:
            atl_model = self._model.to_atl_imperfect()

        winning_states = set(self.get_formula_winning_states())
        coalition = self.agent_name_coalition_to_ids(self._coalition)
        result = []
        start = time.process_time()
        if self._formula_obj.temporalOperator == TemporalOperator.F:
            result = atl_model.minimum_formula_many_agents(coalition,
                                                           winning_states)
        elif self._formula_obj.temporalOperator == TemporalOperator.G:
            result = atl_model.maximum_formula_many_agents(coalition,
                                                           winning_states)
        elif self._formula_obj.temporalOperator == TemporalOperator.FG:
            result = atl_model.minimum_formula_many_agents(coalition,
                                                           atl_model.maximum_formula_many_agents(
                                                               coalition,
                                                               winning_states)
                                                           )
        elif self._formula_obj.temporalOperator == TemporalOperator.GF:
            result = atl_model.maximum_formula_many_agents(coalition,
                                                           atl_model.minimum_formula_many_agents(
                                                               coalition,
                                                               winning_states))
        elif self._formula_obj.temporalOperator == TemporalOperator.X:
            result = atl_model.next_formula_many_agents(coalition,
                                                           winning_states)
        # print(result)
        end = time.process_time()

        return 0 in result, end - start, result, atl_model.strategy

    def agents_to_dict(self):
        agents_id_dict = {}
        counter = 0
        while counter < 2:
            agents_id_dict[self._local_models[counter].agent_name] = counter
            counter += 1
        return agents_id_dict

    def updating_model(self): # entry-point
        """return value : set of states where the formula holds"""
        result = self.updating_model_upgrade_formula(self._formula_obj)
        return result
        
    def updating_model_upgrade_formula(self, upgrade_formula, gm=None):
        """Sends the parts of the formula to their respective methods, divided in upgrade_list 
            and coalition_expression. Generates new global model if none is given to hold with
            the recursion. """
        #print("upgrade_formula", upgrade_formula)
        if isinstance(upgrade_formula, UpgradeFormula) and  upgrade_formula.upgradeList and gm==None:
            gm =GlobalModel([it for it in self._local_models],
                [it for it in self._reduction],
                [it for it in self._bounded_vars],
                [it for it in self._persistent],
                [it for it in self._coalition],
                [it for it in self._goal],
                None, # Skip parsing
                "",
                False,
                self._semantics,
                self._initial)
            gm._formula_obj = upgrade_formula
            gm.coalition = upgrade_formula.agents
            gm._logicType = LogicType.UCL
            gm.generate()
            print("Viritual Model is generated")

            updated_gm = gm.updating_model_upgrade_list(upgrade_formula.upgradeList, gm)
            result = gm.updating_model_coalition_expression(upgrade_formula.coalitionExpression, updated_gm)
            print("result", upgrade_formula, result)
            return result
        
        elif isinstance(upgrade_formula, UpgradeFormula) and upgrade_formula.upgradeList and gm!=None:
            updated_gm = gm.updating_model_upgrade_list(upgrade_formula.upgradeList, gm)
            result = gm.updating_model_coalition_expression(upgrade_formula.coalitionExpression, updated_gm)
            print("result", upgrade_formula, result)
            return result

        else: 
            if isinstance(upgrade_formula, CoalitionExpression):
                result = self.updating_model_coalition_expression(upgrade_formula)
            else:
                result = self.updating_model_coalition_expression(upgrade_formula.coalitionExpression)
            return result

    def updating_model_upgrade_list(self, upgrade_list, gm):
        """Checks which upgradeType the upgrade has, calls the correct methods. 
            return value : a updated model with more or less transitions depending 
            on positive or negative upgrade. """
        #print("upgrade_list", upgrade_list)
        new_transitions = []
        for upgrade in upgrade_list.upgrades:
            if upgrade.type == UpgradeType.P:
                new_transitions = self.updating_model_upgrade_positive(upgrade)
                print("New transitions: ", new_transitions)
                updated_gm = gm._model.updated_model(new_transitions)
            elif upgrade.type == UpgradeType.N:
                preserved_transitions = self.updating_model_upgrade_negative(upgrade)
                remaining_transitions, removed_transitions = self.transitions_to_remove(preserved_transitions) # including preserved forcing actions and other non-forcing actions
                print("All preserved transitions:", remaining_transitions)
                print("Removed transitions", removed_transitions)
                updated_gm = gm._model.updated_model_negative(removed_transitions)
                self.test_negative_clash(remaining_transitions)
        #print("disse transitions blir lagt til av gangen", new_transitions)
        return updated_gm
    
    def updating_model_upgrade_positive(self, upgrade):
        """ Merges the new transitions from all updates within the upgrade,
            tests executability condition before returning values. 
            return value : list of all new transitions in upgrade. """
        #print("upgrade", upgrade)
        new_transitions_list = []
        granted_powers_dict = {}

        for update in upgrade.updates:
            from_states_ids, new_transitions = self.updating_model_update_positive(update)
            if str(update.agent) in granted_powers_dict:
                for element in from_states_ids:
                    granted_powers_dict[str(update.agent)].append(element)
            else: 
                granted_powers_dict[str(update.agent)] = (from_states_ids)

            new_transitions_list += new_transitions

        self.test_clashfreeness(granted_powers_dict)
        return new_transitions_list

    def updating_model_upgrade_negative(self, upgrade):
        """ Merges the preserved transitions from updates in the same upgrade,
            determines whether or not the transitions with 
            joint forcing actions should be preserved.
            return value : list of all preserved transitions in upgrade. """
        preserved_transitions_list = []
        maybe_list = {0:[], 1:[]}
        for update in upgrade.updates:
            preserved_transitions, maybe_preserved_transitions = self.updating_model_update_negative(update)
            for key, value in maybe_preserved_transitions.items():
                for v in value: 
                    maybe_list[key].append(v)
            preserved_transitions_list += preserved_transitions
        for element in maybe_list[0]:
            for el in maybe_list[1]:
                if element == el:
                    preserved_transitions_list.append(element)
        return preserved_transitions_list


    def updating_model_update_negative(self, update):
        """ Determine which transitions with forcing actions should be preserved 
            by the update, and which transitions that might be preserved depending
            on the other updates in the same upgrade 
            return value : list of transitions that will be preserved by the udate
            and a list of transtions that should be preserved by the update but 
            might not be because the other agent also has a forcing action in the
            same transition """
        preserved_transitions = []
        maybe_preserved_transitions = {}
        forcing_actions_agent1, forcing_actions_agent2 = self.get_forcing_actions()
        from_state_ids = self.updating_model_upgrade_formula(update.fromState)
        to_state_ids = self.updating_model_upgrade_formula(update.toState)
        agent = self.agent_name_to_id(str(update.agent))
        if agent == 0:
            for element in forcing_actions_agent1:
                for state in from_state_ids:
                    for s in to_state_ids:
                        if (state, s) == element[0] and element not in forcing_actions_agent2:
                            preserved_transitions.append(element)
                        elif (state, s) == element[0] and element in forcing_actions_agent2:
                            if 0 not in maybe_preserved_transitions.keys():
                                maybe_preserved_transitions[0] = [element]
                            else:
                                maybe_preserved_transitions[0].append(element)
        elif agent == 1:
            for element in forcing_actions_agent2:
                for state in from_state_ids:
                    for s in to_state_ids:
                        if (state, s) == element[0] and element not in forcing_actions_agent1:
                            preserved_transitions.append(element)
                        elif (state, s) == element[0] and element in forcing_actions_agent1:
                            if 1 not in  maybe_preserved_transitions.keys():
                                maybe_preserved_transitions[1] = [element]
                            else:
                                maybe_preserved_transitions[1].append(element)
        return preserved_transitions, maybe_preserved_transitions


    def updating_model_update_positive(self, update):
        """Retrieve state IDs for states that holds formula in from_state and to_state,
            calls method that generates new transitions with dictatorial powers for agent
            in update, 
            return value : list of fromstate IDs for checking executablitiy condition and
                           list of new transitions from current update. """
        #print("update", update)
        from_state_ids = self.updating_model_upgrade_formula(update.fromState)
        to_state_ids = self.updating_model_upgrade_formula(update.toState)
        print("from states and to states in update", from_state_ids, to_state_ids)
        action_pairs = self.get_positive_transitions(from_state_ids, to_state_ids, update.agent)
        new_transitions = action_pairs
        return from_state_ids, new_transitions

    def updating_model_coalition_expression(self, coalition_expression, current_model=None):
        """Generates a ATLIrModel, verifies the part of the formula given and returns states 
            where the part of the model holds. """
        #print("coalition_expression", coalition_expression)
        if coalition_expression.coalitionAgents:
            coalition = self.agent_name_coalition_to_ids(coalition_expression.coalitionAgents)
            print("coalition", coalition)
            winning_states = set(self.updating_model_simple_expression(coalition_expression.simpleExpression, self))
            #print("winning_states", winning_states)
            if current_model != ATLIrModel: 
                current_model = self._model.to_atl_perfect()
            result = current_model.ucl_next(coalition, winning_states)
            print("temporary result from coalition expression", result)
            return result
        else:
            result = self.updating_model_simple_expression(coalition_expression.simpleExpression, self)
            return result

    def updating_model_simple_expression(self, simple_expression, gm):
        """Determine what class the given part of the formula is an instance of and 
            returns a set of state ids where the part of the formula holds. """
        #print("simple_expression", simple_expression)
        if isinstance(simple_expression, UpgradeFormula):
            return self.updating_model_upgrade_formula(simple_expression, gm)
        elif isinstance(simple_expression, CoalitionExpression):
            return self.updating_model_coalition_expression(simple_expression, gm)    
        
        if isinstance(simple_expression, SimpleExpression) and simple_expression.operator != SimpleExpressionOperator.EQ:
            if isinstance(simple_expression.left, UpgradeFormula):
                left = self.updating_model_upgrade_formula(simple_expression.left, gm)
            elif isinstance(simple_expression.left, CoalitionExpression):
                left =  self.updating_model_coalition_expression(simple_expression.left, gm) 
            elif isinstance(simple_expression.left, SimpleExpression):
                if simple_expression.left.operator == SimpleExpressionOperator.NOT:
                    left = self.get_resulting_states(self.updating_model_simple_expression(simple_expression.left.right, gm), simple_expression.left.operator)
                else:
                    left = self.get_states_with_props(simple_expression.left)
            else:
                left = set()
            if isinstance(simple_expression.right, UpgradeFormula):
                right = self.updating_model_upgrade_formula(simple_expression.right, gm)
            elif isinstance(simple_expression.right, CoalitionExpression):
                right =  self.updating_model_coalition_expression(simple_expression.right, gm)
            elif isinstance(simple_expression.right, SimpleExpression):
                if simple_expression.right.operator == SimpleExpressionOperator.NOT:
                    right = self.get_resulting_states(self.updating_model_simple_expression(simple_expression.right.right, gm), simple_expression.right.operator)
                else:
                    right = self.get_states_with_props(simple_expression.right)
            return self.get_resulting_states(right, simple_expression.operator, left)

        if isinstance(simple_expression, SimpleExpression):
            if simple_expression.operator == SimpleExpressionOperator.NOT:
                states_with_props = self.updating_model_simple_expression(simple_expression)
            else:
                states_with_props = self.get_states_with_props(simple_expression)
            return states_with_props
        else: 
            return self.updating_model_upgrade_formula(simple_expression, gm)

    def get_states_with_props(self, expr) -> List[int]:
        """Retrieves state IDs to states where the propositions given value holds.
            Return value : list of state IDs."""
        result = []
        for state in self._states:
            if expr.left not in state.props and expr.right == "False": # if prop not stated in state and value is false, append state.id because props that is not stated is allways false in the state.
               state.set_prop(expr.left, False)
            if expr.evaluate(state.props):
                result.append(state.id)
        print("Props:", expr, "States with props:", result)
        return result

    def get_resulting_states(self, right, operator, left=set()):
        """Processes sets of state IDs with SimpleExpressionOperator
            between them and returns a set of states that holds with
            the operator. """
        result = set()
        if operator == SimpleExpressionOperator.NOT:
            all_states = set(state.id for state in self._states)
            result = all_states - right
        elif operator == SimpleExpressionOperator.AND:
            result = set(left).intersection(set(right)) 
        elif operator == SimpleExpressionOperator.OR:
            result = set(left).union(set(right))
        print("get_resulting_states", left, operator, right, " = ", result)
        return result
        
    def get_positive_transitions(self, from_states, to_states, agent):
        """returns new transitions for all actions counterpart has in the 
        state where the agent is granted dictatorial powers"""    
        global count
        agents_id_dict = self.agents_to_dict()
        transitions = []
        if agents_id_dict.get(str(agent)) == 0:
            for element in from_states:
                for elem in to_states:
                    for el in self._model.get_possible_strategies_for_coalition(element, [1]): 
                        transitions.append([element, elem, [f"dict_powers{count}", el[0]]])
                    count += 1
        elif agents_id_dict.get(str(agent)) == 1:
            for element in from_states:
                for elem in to_states:
                    for el in self._model.get_possible_strategies_for_coalition(element, [0]): 
                        transitions.append([element, elem, [el[0],f"dict_powers{count}"]])
                    count += 1
        else:
            raise Exception("Only works when two agents")
        return transitions


    def test_clashfreeness(self, granted_powers_dict):
        """ Tests positive executability condition, no two agents
            can be granted dictatorial powers from the same state.
            Raises exception if it does not hold."""
        values = set()
        value_count = 0
        if len(granted_powers_dict.keys()) > 1: 
            for _, value in granted_powers_dict.items():
                value_count += len(value)
                for val in value:
                    values.add(val)
        if len(values) != value_count:
            raise Exception("Two or more updates are clashing.")

    def test_negative_clash(self, remaining_transitions):
        """ Tests negative executability condition, all states needs at least one out-going transition. 
            There has to be enough transitions to cover the cartisian product of the actions 
            in the specific state. Raises exception if it does not hold. """
        state_action_dict, action_pairs_counter = self.get_state_actions()
        from_states_list = []
        for element in remaining_transitions:
            from_states_list.append(element[0][0])
        for state in self._states:
            if state.id not in from_states_list: 
                raise Exception("There has to be at least one out-going arrow per state.")
            
        state_action_dict = {}
        action_pairs_counter = 0
        for state in self._states:
            actions_agent1 = [i[0] for i in self._model.get_possible_strategies_for_coalition(state.id, [0])]
            actions_agent2 = [i[0] for i in self._model.get_possible_strategies_for_coalition(state.id, [1])]
            state_action_dict[state.id] = []
            for elm in actions_agent1:
                for el in actions_agent2:
                    state_action_dict[state.id].append((elm, el))
                    action_pairs_counter += 1

        remaining_transitions_dict = {}
        count = 0
        test_count = 0
        while count < len(self._states):
            for el in remaining_transitions:
                if el[0][0] == count:
                    if count not in remaining_transitions_dict.keys():
                        remaining_transitions_dict[count] = [(el[1])]
                    else: remaining_transitions_dict[count].append((el[1]))
            count += 1

        for key, value in state_action_dict.items():
            for k, v in remaining_transitions_dict.items():
                if key == k:
                    test_count += len(set(value).intersection(set(v)))
                   
        if action_pairs_counter > test_count:
            raise Exception("There is not enough arrows preserved to be a valid concurrent game model.")
             
            
    def transitions_to_remove(self, preserved_transitions): 
        """Determine which transitions to remove based on which are forcing
            and which are preserved. 
            Return value : list of remaining transitions with both forcing and 
            non-forcing actions, list of transitions that will not be preserved 
            in the upgrade. """
        all_transitions = self._model.get_full_transitions()
        forcing_act1, forcing_act2 = self.get_forcing_actions()
        forcing_actions = []
        for element in forcing_act1:
            if element not in forcing_actions:
                forcing_actions.append(element)
        for element in forcing_act2:
            if element not in forcing_actions:
                forcing_actions.append(element)

        forcing_actions_dict = {}
        for element in forcing_actions:
            if element[0] not in forcing_actions_dict:
                forcing_actions_dict[element[0]] = [element[1]]
            else: 
                forcing_actions_dict[element[0]].append(element[1])

        temp_removed_transitions = []
        for element in forcing_actions:
            if element not in preserved_transitions:
                temp_removed_transitions.append(element)
        
        removed_transitions = []
        for element in forcing_actions:
            if element not in preserved_transitions:
                removed_transitions.append([element[0], [element[1][0], element[1][1]]])
                
        remaining_transitions = []
        for key, value in all_transitions.items():
            for val in value:
                if [key, val] not in temp_removed_transitions:
                    remaining_transitions.append([key, val])

        return remaining_transitions, removed_transitions

    def get_state_actions(self):
        """Retrieves data of which actions the agents have in the different states, 
            returns a dictionary with state ID as key and action pairs in list as value
            returns a counter for minimum amount of action_pairs in a CGM with the 
            current states and actions."""
        state_action_dict = {}
        action_pairs_counter = 0
        for state in self._states:
            actions_agent1 = [i[0] for i in self._model.get_possible_strategies_for_coalition(state.id, [0])]
            actions_agent2 = [i[0] for i in self._model.get_possible_strategies_for_coalition(state.id, [1])]
            state_action_dict[state.id] = []
            for elm in actions_agent1:
                for el in actions_agent2:
                    state_action_dict[state.id].append((elm, el))
                    action_pairs_counter += 1
            
        return state_action_dict, action_pairs_counter

    def get_equal_states(self):
        """Checks if states have the same props,
            return value: list of sets of states,
            the index is the same as state ID 
            and in each set is all state IDs
            for states with same props. """
        states_with_same_props = []
        for state in self._states:
            temp = set()
            for s in self._states:
                if state.props == s.props:
                    temp.add(s.id)
            states_with_same_props.append(temp)
        return states_with_same_props

    def get_forcing_actions(self):
        """Determines forcing actions. 
            Return values : list of transitions with forcing actions 
                            for each agent. """
        dict_actions = self._model.get_full_transitions()
        same_props = self.get_equal_states()
        forcing_actions_agent1 = []
        forcing_actions_agent2 = []
        count = 0
        while count < self._agents_count:
            from_act_to_all = {}
            for state in self._states:
                actions = self._model.get_possible_strategies_for_coalition(state.id, [count])
                for action in actions:
                    all_to_states = []
                    for transition in self._model._graph[state.id]:
                        if transition.actions[count] == action[0]:
                            all_to_states.append(transition.next_state)
                    if state.id not in from_act_to_all.keys():
                        from_act_to_all[state.id] = [{action[0]: all_to_states}]
                    else: from_act_to_all[state.id].append({action[0]: all_to_states})
            for from_state, value in from_act_to_all.items():
                for d in value:
                    for to_states in d.values():
                        if len(to_states) == 1:
                            if count == 0:
                                for i in dict_actions[(from_state,to_states[0])]:
                                    forcing_actions_agent1.append([(from_state, to_states[0]), i])
                            elif count == 1: 
                                for i in dict_actions[(from_state,to_states[0])]:
                                    forcing_actions_agent2.append([(from_state, to_states[0]), i])
                        else: 
                            for props in same_props:
                                if len(props.intersection(set(to_states))) == len(to_states):
                                    for to_state in to_states:
                                        if count == 0:
                                            for i in dict_actions[(from_state,to_state)]:
                                                if [(from_state, to_state), i] not in forcing_actions_agent1:
                                                    forcing_actions_agent1.append([(from_state, to_state), i]) 
                                        elif count == 1:
                                            for i in dict_actions[(from_state,to_state)]:
                                                if [(from_state, to_state), i] not in forcing_actions_agent2:
                                                    forcing_actions_agent2.append([(from_state, to_state), i]) 
            count += 1
        return forcing_actions_agent1, forcing_actions_agent2

    def verify_approximation_ucl(self): 
        """Method called from main.py. Estimates time
        for verification and returns boolean answer to the
        model checking problem, time for verification 
        and the set of states where the formula is true. """
        start = time.process_time()
        result = self.updating_model()
        end = time.process_time()    

        return 0 in result, end - start, result
    
    def verify_domino(self):
        agent_id = self.get_agent()
        strategy_comparer = StrategyComparer(self._model, self.get_actions()[agent_id])
        start = time.process_time()
        result, strategy = strategy_comparer.domino_dfs(0, set(self.get_formula_winning_states()), [agent_id],
                                                        strategy_comparer.basic_h)
        end = time.process_time()
        # print(strategy)
        return result, end - start

    def get_actions(self):
        actions = []
        for local in self._local_models:
            actions.append(local.actions)
            #actions[-1].add("")
        return actions

    def get_fake_formula_winning_states(self, revote: int, cand: int):
        expr = self._formula_obj.expression
        result = []
        for state in self._states:
            if expr.evaluate(state.props):
                if state.props["VoterC1_revote"] != revote:
                    result.append(state.id)
                elif state.props["VoterC1_vote"] == cand:
                    ok = True
                    for ep in self.model.epistemic_class_for_state(state.id, self.get_agent()):
                        ep_state = self._states[ep]
                        if ep_state.props["VoterC1_vote"] != state.props["VoterC1_vote"]:
                            ok = False
                            break

                    if ok:
                        result.append(state.id)

        # print(result)
        return result

    def get_formula_winning_states(self) -> List[int]:
        expr = self._formula_obj.expression
        print(expr)
        result = []
        for state in self._states:
            if expr.evaluate(state.props):
                result.append(state.id)

        return result

    def get_real_formula_winning_states(self) -> List[int]:
        expr = self._formula_obj.expression
        result = []
        for state in self._states:
            if expr.evaluate(state.props):
                result.append(state.id)

        return result

    def coalition_ids_to_str(self, coalition: List[int]) -> List[str]:
        result = []
        for agent_id in coalition:
            result.append(self._local_models[agent_id].agent_name)
        return result

    def save_to_file(self, filename: str):
        model_file = open(filename, "w")
        # model_dump = self.model.dump_for_agent(self.get_agent())
        model_dump = self.model.dump_for_coalition(self.get_coalition())
        model_file.write(model_dump)
        winning_states = self.get_formula_winning_states()
        # winning_states = self.get_winning_states()
        model_file.write(f"{len(winning_states)}\n")
        for state_id in winning_states:
            model_file.write(f"{state_id}\n")

        model_file.write("0\n")
        model_file.close()

    def classic_save_to_file(self, filename: str):
        model_file = open(filename, "w")
        model_dump = self.model.dump_for_agent(self.get_agent())
        model_file.write(model_dump)
        winning_states = self.get_real_formula_winning_states()
        model_file.write(f"{len(winning_states)}\n")
        for state_id in winning_states:
            model_file.write(f"{state_id}\n")

        model_file.write("0\n")
        model_file.close()

    def classic_save_to_file_coal(self, filename: str):
        model_file = open(filename, "w")
        model_dump = self.model.dump_for_coalition(self.get_coalition())
        model_file.write(model_dump)
        winning_states = self.get_real_formula_winning_states()
        model_file.write(f"{len(winning_states)}\n")
        for state_id in winning_states:
            model_file.write(f"{state_id}\n")

        model_file.write("0\n")
        model_file.close()

    def selene_save_to_file(self, filename: str, params: []):
        model_file = open(filename, "w")
        # model_dump = self.model.dump_for_agent(self.get_agent())
        model_dump = self.model.dump_for_coalition(self.get_coalition())
        model_file.write(model_dump)
        model_file.write(f"{len(params)}\n")
        for p in params:
            winning_states = self.get_fake_formula_winning_states(p[0], p[1])
            model_file.write(f"{len(winning_states)}\n")
            for state_id in winning_states:
                model_file.write(f"{state_id}\n")

        model_file.write("0\n")
        model_file.close()

    def __str__(self):
        result = f"SEMANTICS: {self._semantics}\n\n"
        for local_model in self._local_models:
            result += f"{local_model}\n"

        result += f"PERSISTENT: [{', '.join(self._persistent)}]\n"
        result += f"INITIAL: [{', '.join(f'{key}={self._initial[key]}' for key in self._initial)}]\n"
        result += f"FORMULA: {self._formula}\n"
        return result


if __name__ == "__main__":
    from stv.models.asynchronous.parser import GlobalModelParser
    from stv.parsers import FormulaParser

    filename = f"sai_2ai_2mmq_mim"

    # model = GlobalModelParser().parse(f"stv/models/asynchronous/specs/generated/{filename}.txt")
    model = GlobalModelParser().parse(f"specs/generated/{filename}.txt")

    start = time.process_time()
    model.generate(reduction=False)
    end = time.process_time()

    print(f"Generation time: {end - start}, #states: {model.states_count}, #transitions: {model.transitions_count}")
