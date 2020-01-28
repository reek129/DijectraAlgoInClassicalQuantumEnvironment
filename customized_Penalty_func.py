# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 13:24:56 2019

@author: Reek's
"""
import logging
from math import fsum
import warnings

import numpy as np
from docplex.mp.constants import ComparisonType
from docplex.mp.model import Model
from qiskit.quantum_info import Pauli

from qiskit.aqua import AquaError
from qiskit.aqua.operators import WeightedPauliOperator

logger = logging.getLogger(__name__)

def get_operator(mdl, auto_penalty=True, default_penalty=1e5):
    """ Generate Ising Hamiltonian from a model of DOcplex.
    Args:
        mdl (docplex.mp.model.Model): A model of DOcplex for a optimization problem.
        auto_penalty (bool): If true, the penalty coefficient is automatically defined
                             by "_auto_define_penalty()".
        default_penalty (float): The default value of the penalty coefficient for the constraints.
            This value is used if "auto_penalty" is False.
    Returns:
        tuple(operators.WeightedPauliOperator, float): operator for the Hamiltonian and a
        constant shift for the obj function.
    """

    _validate_input_model(mdl)

    # set the penalty coefficient by _auto_define_penalty() or manually.
    if auto_penalty:
        penalty = _auto_define_penalty(mdl, default_penalty)
    else:
        penalty = default_penalty

    # set a sign corresponding to a maximized or minimized problem.
    # sign == 1 is for minimized problem. sign == -1 is for maximized problem.
    sign = 1
    if mdl.is_maximized():
        sign = -1

    # assign variables of the model to qubits.
    q_d = {}
    index = 0
    for i in mdl.iter_variables():
        if i in q_d:
            continue
        q_d[i] = index
        index += 1

    # initialize Hamiltonian.
    num_nodes = len(q_d)
    pauli_list = []
    shift = 0
    zero = np.zeros(num_nodes, dtype=np.bool)

    # convert a constant part of the object function into Hamiltonian.
    shift += mdl.get_objective_expr().get_constant() * sign

    # convert linear parts of the object function into Hamiltonian.
    l_itr = mdl.get_objective_expr().iter_terms()
    for j in l_itr:
        z_p = np.zeros(num_nodes, dtype=np.bool)
        index = q_d[j[0]]
        weight = j[1] * sign / 2
        z_p[index] = True

        pauli_list.append([-weight, Pauli(z_p, zero)])
        shift += weight

    # convert quadratic parts of the object function into Hamiltonian.
    q_itr = mdl.get_objective_expr().iter_quads()
    for i in q_itr:
        index1 = q_d[i[0][0]]
        index2 = q_d[i[0][1]]
        weight = i[1] * sign / 4

        if index1 == index2:
            shift += weight
        else:
            z_p = np.zeros(num_nodes, dtype=np.bool)
            z_p[index1] = True
            z_p[index2] = True
            pauli_list.append([weight, Pauli(z_p, zero)])

        z_p = np.zeros(num_nodes, dtype=np.bool)
        z_p[index1] = True
        pauli_list.append([-weight, Pauli(z_p, zero)])

        z_p = np.zeros(num_nodes, dtype=np.bool)
        z_p[index2] = True
        pauli_list.append([-weight, Pauli(z_p, zero)])

        shift += weight

    # convert constraints into penalty terms.
    for constraint in mdl.iter_constraints():
        if type(constraint).__name__ == "IndicatorConstraint":
            constraint = constraint.get_linear_constraint()
        constant = constraint.right_expr.get_constant()

        # constant parts of penalty*(Constant-func)**2: penalty*(Constant**2)
        shift += penalty * constant ** 2

        # linear parts of penalty*(Constant-func)**2: penalty*(-2*Constant*func)
        for __l in constraint.left_expr.iter_terms():
            z_p = np.zeros(num_nodes, dtype=np.bool)
            index = q_d[__l[0]]
            weight = __l[1]
            z_p[index] = True

            pauli_list.append([penalty * constant * weight, Pauli(z_p, zero)])
            shift += -penalty * constant * weight

        # quadratic parts of penalty*(Constant-func)**2: penalty*(func**2)
        for __l in constraint.left_expr.iter_terms():
            for l_2 in constraint.left_expr.iter_terms():
                index1 = q_d[__l[0]]
                index2 = q_d[l_2[0]]
                weight1 = __l[1]
                weight2 = l_2[1]
                penalty_weight1_weight2 = penalty * weight1 * weight2 / 4

                if index1 == index2:
                    shift += penalty_weight1_weight2
                else:
                    z_p = np.zeros(num_nodes, dtype=np.bool)
                    z_p[index1] = True
                    z_p[index2] = True
                    pauli_list.append([penalty_weight1_weight2, Pauli(z_p, zero)])

                z_p = np.zeros(num_nodes, dtype=np.bool)
                z_p[index1] = True
                pauli_list.append([-penalty_weight1_weight2, Pauli(z_p, zero)])

                z_p = np.zeros(num_nodes, dtype=np.bool)
                z_p[index2] = True
                pauli_list.append([-penalty_weight1_weight2, Pauli(z_p, zero)])

                shift += penalty_weight1_weight2

    # Remove paulis whose coefficients are zeros.
    qubit_op = WeightedPauliOperator(paulis=pauli_list)

    return qubit_op, shift

def _validate_input_model(mdl):
    """ Check whether an input model is valid. If not, raise an AquaError
    Args:
         mdl (docplex.mp.model.Model): A model of DOcplex for a optimization problem.
    Raises:
        AquaError: Unsupported input model
    """
    valid = True

    # validate an object type of the input.
    if not isinstance(mdl, Model):
        raise AquaError('An input model must be docplex.mp.model.Model.')

    # raise an error if the type of the variable is not a binary type.
    for var in mdl.iter_variables():
        if not var.is_binary():
            logger.warning('The type of Variable %s is %s. It must be a binary variable. ',
                           var, var.vartype.short_name)
            valid = False

    # raise an error if the constraint type is not an equality constraint.
    for constraint in mdl.iter_constraints():
        if type(constraint).__name__ == "IndicatorConstraint":
            constraint = constraint.get_linear_constraint()
        if not constraint.sense == ComparisonType.EQ:
            logger.warning('Constraint %s is not an equality constraint.', constraint)
            valid = False
        
    if not valid:
        raise AquaError('The input model has unsupported elements.')


def _auto_define_penalty(mdl, default_penalty=1e5):
    """ Automatically define the penalty coefficient.
    This returns object function's (upper bound - lower bound + 1).
    Args:
        mdl (docplex.mp.model.Model): A model of DOcplex for a optimization problem.
        default_penalty (float): The default value of the penalty coefficient for the constraints.
    Returns:
        float: The penalty coefficient for the Hamiltonian.
    """

    # if a constraint has float coefficient, return 1e5 for the penalty coefficient.
    terms = []
    for constraint in mdl.iter_constraints():
        if type(constraint).__name__ == "IndicatorConstraint":
            constraint = constraint.get_linear_constraint()
        terms.append(constraint.right_expr.get_constant())
        terms.extend(term[1] for term in constraint.left_expr.iter_terms())
        
    if any(isinstance(term, float) and not term.is_integer() for term in terms):
        logger.warning('Using %f for the penalty coefficient because a float coefficient exists '
                       'in constraints. \nThe value could be too small. '
                       'If so, set the penalty coefficient manually.', default_penalty)
        return default_penalty

    # (upper bound - lower bound) can be calculate as the sum of absolute value of coefficients
    # Firstly, add 1 to guarantee that infeasible answers will be greater than upper bound.
    penalties = [1]
    # add linear terms of the object function.
    penalties.extend(abs(i[1]) for i in mdl.get_objective_expr().iter_terms())
    # add quadratic terms of the object function.
    penalties.extend(abs(i[1]) for i in mdl.get_objective_expr().iter_quads())

    return fsum(penalties)

