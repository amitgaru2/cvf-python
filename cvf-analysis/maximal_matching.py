import os
import copy
import string

from cvf_analysis import CVFAnalysis, PartialCVFAnalysisMixin, logger


class Configuration:
    def __init__(self, p=None, m=False):
        self._p = p
        self._m = m

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, val):
        self._p = val

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, val):
        self._m = val

    def __eq__(self, other):
        return self.p == other.p and self.m == other.m

    def __hash__(self):
        return hash((self.p, self.m))

    def __repr__(self):
        return f"<p: {self.p}, m: {self.m}>"


class MaximalMatchingFullAnalysis(CVFAnalysis):
    results_prefix = "maximal_matching"
    results_dir = os.path.join("results", results_prefix)

    def possible_pvalues_of_node(self, position):
        return set([None, *self.graph_based_on_indx[position]])

    def _gen_configurations(self):
        self.configurations = {
            tuple([Configuration(p=None, m=False) for i in range(len(self.nodes))])
        }
        # perturb each state at a time for all states in configurations and accumulate the same in the configurations for next state to perturb
        for _, n in enumerate(self.nodes):
            node_pos = self.node_positions[n]
            config_copy = copy.deepcopy(self.configurations)
            for val in self.possible_pvalues_of_node(node_pos):
                for cc in config_copy:
                    cc = list(cc)
                    cc[node_pos] = Configuration(p=val, m=False)
                    self.configurations.add(tuple(cc))
                    cc[node_pos] = Configuration(p=val, m=True)
                    self.configurations.add(tuple(cc))

        logger.info("No. of Configurations: %s", len(self.configurations))

    def _check_if_only_one_eligible_process(self, state):
        """check invariant"""

        def _pr_married(j, config):
            for i in self.graph_based_on_indx[j]:
                if state[i].p == j and config.p == i:
                    return True
            return False

        for j, config in enumerate(state):
            # update m.j
            if config.m != _pr_married(j, config):
                return False

            # accept a proposal
            if config.m == _pr_married(j, config):
                if config.p is None:
                    for i in self.graph_based_on_indx[j]:
                        if state[i].p == j:
                            return False

                    for k in self.graph_based_on_indx[j]:
                        if state[k].p is None and k < j and not state[k].m:
                            return False
                else:
                    i = config.p
                    if state[i].p != j and (state[i].m or j <= i):
                        return False

        return True

    def _find_invariants(self):
        for state in self.configurations:
            if self._check_if_only_one_eligible_process(state):
                self.invariants.add(state)

        logger.info("No. of Invariants: %s", len(self.invariants))

    def _is_program_transition(self, perturb_pos, start_state, dest_state) -> bool:
        j = perturb_pos
        state = start_state
        config = state[perturb_pos]
        dest_config = dest_state[perturb_pos]

        def _pr_married(j, config):
            for i in self.graph_based_on_indx[j]:
                if state[i].p == j and config.p == i:
                    return True
            return False

        # update m.j
        if start_state[perturb_pos].m != _pr_married(j, config):
            if dest_config.m == _pr_married(j, config):
                return True

        # accept a proposal
        if config.m == _pr_married(j, config):
            if config.p is None:
                for i in self.graph_based_on_indx[j]:
                    if state[i].p == j and dest_config.p == i:
                        return True

                # make a proposal
                for i in self.graph_based_on_indx[j]:
                    if state[i].p == j:
                        break
                else:
                    max_k = -1
                    for k in self.graph_based_on_indx[j]:
                        if state[k].p is None and k < j and not state[k].m:
                            if k > max_k:
                                max_k = k

                    if max_k >= 0 and dest_config.p == max_k:
                        return True
            else:
                # withdraw a proposal
                i = config.p
                if state[i].p != j and (state[i].m or j <= i):
                    if dest_config.p is None:
                        return True

        return False

    def _get_program_transitions(self, start_state):
        program_transitions = set()
        for position, _ in enumerate(start_state):
            possible_config_p_val = self.possible_pvalues_of_node(position) - {
                start_state[position].p
            }
            for perturb_p_val in possible_config_p_val:
                perturb_state = list(copy.deepcopy(start_state))
                perturb_state[position].p = perturb_p_val
                perturb_state = tuple(perturb_state)
                if self._is_program_transition(position, start_state, perturb_state):
                    program_transitions.add(perturb_state)

            possible_config_m_val = {True, False} - {start_state[position].m}
            for perturb_m_val in possible_config_m_val:
                perturb_state = list(copy.deepcopy(start_state))
                perturb_state[position].m = perturb_m_val
                perturb_state = tuple(perturb_state)
                if self._is_program_transition(position, start_state, perturb_state):
                    program_transitions.add(perturb_state)

        return {"program_transitions": program_transitions}

    def _evaluate_perturbed_pr_married(self, position, state):
        if state[position].p is None:
            return [False]
        return [True, False]

    def _get_cvfs(self, start_state):
        cvfs_in = dict()
        cvfs_out = dict()

        def _add_to_cvf(perturb_state, position):
            if start_state in self.invariants:
                cvfs_in[perturb_state] = position
            else:
                cvfs_out[perturb_state] = position

        for position, _ in enumerate(start_state):
            config = start_state[position]
            for a_pr_married_value in self._evaluate_perturbed_pr_married(
                position, start_state
            ):
                if config.m is not a_pr_married_value:
                    perturb_state = copy.deepcopy(start_state)
                    perturb_state[position].m = a_pr_married_value
                    _add_to_cvf(perturb_state, position)
                else:
                    if config.p is None:
                        for nbr in self.graph_based_on_indx[position]:
                            perturb_state = copy.deepcopy(start_state)
                            perturb_state[position].p = nbr
                            perturb_state[position].m = a_pr_married_value
                            _add_to_cvf(perturb_state, position)
                    else:
                        perturb_state = copy.deepcopy(start_state)
                        perturb_state[position].p = None
                        perturb_state[position].m = a_pr_married_value
                        _add_to_cvf(perturb_state, position)

        return {"cvfs_in": cvfs_in, "cvfs_out": cvfs_out}


class MaximalMatchingPartialAnalysis(PartialCVFAnalysisMixin, MaximalMatchingFullAnalysis):

    def _get_program_transitions(self, start_state):
        program_transitions = []
        pt_per_node = []

        for position, _ in enumerate(start_state):
            possible_config_p_val = self.possible_pvalues_of_node(position) - {
                start_state[position].p
            }
            for perturb_p_val in possible_config_p_val:
                perturb_state = list(copy.deepcopy(start_state))
                perturb_state[position].p = perturb_p_val
                perturb_state = tuple(perturb_state)
                if self._is_program_transition(position, start_state, perturb_state):
                    pt_per_node.append(perturb_state)
                    # program_transitions.add(perturb_state)

            possible_config_m_val = {True, False} - {start_state[position].m}
            for perturb_m_val in possible_config_m_val:
                perturb_state = list(copy.deepcopy(start_state))
                perturb_state[position].m = perturb_m_val
                perturb_state = tuple(perturb_state)
                if self._is_program_transition(position, start_state, perturb_state):
                    pt_per_node.append(perturb_state)
                    # program_transitions.add(perturb_state)

            if pt_per_node:
                program_transitions.append(pt_per_node)
                pt_per_node = []

        result = self.generate_random_samples(program_transitions, self.K_sampling)

        return {"program_transitions": set(result)}

    # def _get_program_transitions(self, start_state):
    #     """v2"""
    #     program_transitions = set()
        
    #     ignore_node_p_perturb_values = {i: {start_state[i].p} for i in range(len(start_state))}
    #     ignore_node_m_perturb_values = {i: {start_state[i].m} for i in range(len(start_state))}
        
    #     generator = self.generate_next_random_node_position(len(start_state))
        
    #     for _ in range(self.K_sampling):
    #         try:
    #             position = next(generator)
    #         except StopIteration:
    #             break
            
    #         perturb_p_values = self.possible_pvalues_of_node(position) - ignore_node_p_perturb_values[position]
    #         perturb_m_values = {True, False} - ignore_node_m_perturb_values[position]
    
    #         if not perturb_p_values and not perturb_m_values:
    #             try:
    #                 generator.send(True)
    #             except StopIteration:
    #                 break
    #         else:
    #             for perturb_value in perturb_p_values:
    #                 perturb_state = list(copy.deepcopy(start_state))
    #                 perturb_state[position].p = perturb_value
    #                 perturb_state = tuple(perturb_state)
    #                 if self._is_program_transition(position, start_state, perturb_state):
    #                     program_transitions.add(perturb_state)
    #                     ignore_node_p_perturb_values[position].add(perturb_value)
    #                     break
    #                 else:
    #                     ignore_node_p_perturb_values[position].add(perturb_value)
    #             else:
    #                 for perturb_value in perturb_m_values:
    #                     perturb_state = list(copy.deepcopy(start_state))
    #                     perturb_state[position].m = perturb_value
    #                     perturb_state = tuple(perturb_state)
    #                     if self._is_program_transition(position, start_state, perturb_state):
    #                         program_transitions.add(perturb_state)
    #                         ignore_node_m_perturb_values[position].add(perturb_value)
    #                         break
    #                     else:
    #                         ignore_node_m_perturb_values[position].add(perturb_value)
        
    #     return {"program_transitions": program_transitions}       


    def _get_cvfs(self, start_state):
        def _flat_list_to_dict(lst):
            result = {}
            for item in lst:
                result[item[0]] = item[1]
            return result
    
        cvfs_in = []
        cvfs_out = []
        cvfs_in_per_node = []
        cvfs_out_per_node = []

        def _add_to_cvf(perturb_state, position):
            if start_state in self.invariants:
                cvfs_in_per_node.append((perturb_state, position))
            else:
                cvfs_out_per_node.append((perturb_state, position))

        for position, _ in enumerate(start_state):
            config = start_state[position]
            for a_pr_married_value in self._evaluate_perturbed_pr_married(
                position, start_state
            ):
                if config.m is not a_pr_married_value:
                    perturb_state = copy.deepcopy(start_state)
                    perturb_state[position].m = a_pr_married_value
                    _add_to_cvf(perturb_state, position)
                else:
                    if config.p is None:
                        for nbr in self.graph_based_on_indx[position]:
                            perturb_state = copy.deepcopy(start_state)
                            perturb_state[position].p = nbr
                            perturb_state[position].m = a_pr_married_value
                            _add_to_cvf(perturb_state, position)
                    else:
                        perturb_state = copy.deepcopy(start_state)
                        perturb_state[position].p = None
                        perturb_state[position].m = a_pr_married_value
                        _add_to_cvf(perturb_state, position)

            if cvfs_in_per_node:
                cvfs_in.append(cvfs_in_per_node)
                cvfs_in_per_node = []

            if cvfs_out_per_node:
                cvfs_out.append(cvfs_out_per_node)
                cvfs_out_per_node = []

        result_cvfs_in = _flat_list_to_dict(self.generate_random_samples(cvfs_in, self.K_sampling))
        result_cvfs_out = _flat_list_to_dict(self.generate_random_samples(cvfs_out, self.K_sampling))

        return {"cvfs_in": result_cvfs_in, "cvfs_out": result_cvfs_out}