# MIT License
#
# Copyright (c) 2024 DALabNOVA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
SLIM_GSGP Class for Evolutionary Computation using PyTorch.
"""

import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from slim_gsgp.algorithms.GP.representations.tree import Tree as GP_Tree
from slim_gsgp.algorithms.GSGP.representations.tree import Tree
from slim_gsgp.algorithms.SLIM_GSGP.representations.individual import Individual
from slim_gsgp.algorithms.SLIM_GSGP.representations.population import Population
from slim_gsgp.utils.diversity import gsgp_pop_div_from_vectors
from slim_gsgp.utils.logger import logger
from slim_gsgp.utils.utils import verbose_reporter


class SLIM_GSGP:

    def __init__(
        self,
        pi_init,
        initializer,
        selector,
        inflate_mutator,
        deflate_mutator,
        ms,
        crossover,
        find_elit_func,
        p_m=1,
        p_xo=0,
        p_inflate=0.3,
        p_deflate=0.7,
        pop_size=100,
        seed=0,
        operator="sum",
        copy_parent=True,
        two_trees=True,
        settings_dict=None,
    ):
        """
        Initialize the SLIM_GSGP algorithm with given parameters.

        Parameters
        ----------
        pi_init : dict
            Dictionary with all the parameters needed for candidate solutions initialization.
        initializer : Callable
            Function to initialize the population.
        selector : Callable
            Function to select individuals.
        inflate_mutator : Callable
            Function for inflate mutation.
        deflate_mutator : Callable
            Function for deflate mutation.
        ms : Callable
            Mutation step function.
        crossover : Callable
            Crossover function.
        find_elit_func : Callable
            Function to find elite individuals.
        p_m : float
            Probability of mutation. Default is 1.
        p_xo : float
            Probability of crossover. Default is 0.
        p_inflate : float
            Probability of inflate mutation. Default is 0.3.
        p_deflate : float
            Probability of deflate mutation. Default is 0.7.
        pop_size : int
            Size of the population. Default is 100.
        seed : int
            Random seed for reproducibility. Default is 0.
        operator : {'sum', 'prod'}
            Operator to apply to the semantics, either "sum" or "prod". Default is "sum".
        copy_parent : bool
            Whether to copy the parent when mutation is not possible. Default is True.
        two_trees : bool
            Indicates if two trees are used. Default is True.
        settings_dict : dict
            Additional settings passed as a dictionary.

        """
        self.pi_init = pi_init
        self.selector = selector
        self.p_m = p_m
        self.p_inflate = p_inflate
        self.p_deflate = p_deflate
        self.crossover = crossover
        self.inflate_mutator = inflate_mutator
        self.deflate_mutator = deflate_mutator
        self.ms = ms
        self.p_xo = p_xo
        self.initializer = initializer
        self.pop_size = pop_size
        self.seed = seed
        self.operator = operator
        self.copy_parent = copy_parent
        self.two_trees = two_trees
        self.settings_dict = settings_dict
        self.find_elit_func = find_elit_func

        Tree.FUNCTIONS = pi_init["FUNCTIONS"]
        Tree.TERMINALS = pi_init["TERMINALS"]
        Tree.CONSTANTS = pi_init["CONSTANTS"]

        GP_Tree.FUNCTIONS = pi_init["FUNCTIONS"]
        GP_Tree.TERMINALS = pi_init["TERMINALS"]
        GP_Tree.CONSTANTS = pi_init["CONSTANTS"]

    def _calculate_generation_diversity(self, population):
        """
        Calculate population diversity based on the operator type.
        
        Parameters
        ----------
        population : Population
            The population for which to calculate diversity.
            
        Returns
        -------
        float
            The calculated diversity value.
        """
        if self.operator == "sum":
            semantics_stack = torch.stack([
                torch.sum(ind.train_semantics, dim=0) 
                for ind in population.population
            ])
        else:  # operator == "prod"
            semantics_stack = torch.stack([
                torch.prod(ind.train_semantics, dim=0) 
                for ind in population.population
            ])
        
        return float(gsgp_pop_div_from_vectors(semantics_stack))

    def _prepare_logging_info(self, population, log_level):
        """
        Prepare additional logging information based on the log level.
        
        Parameters
        ----------
        population : Population
            The current population.
        log_level : int
            The logging level (1-4).
            
        Returns
        -------
        list
            List of additional information for logging.
        """
        base_info = [self.elite.test_fitness, self.elite.nodes_count]
        
        if log_level == 1:
            return base_info + [log_level]
        
        elif log_level == 2:
            gen_diversity = self._calculate_generation_diversity(population)
            return base_info + [
                gen_diversity,
                np.std(population.fit),
                log_level
            ]
        
        elif log_level == 3:
            nodes_info = " ".join([str(ind.nodes_count) for ind in population.population])
            fitness_info = " ".join([str(f) for f in population.fit])
            return base_info + [nodes_info, fitness_info, log_level]
        
        elif log_level == 4:
            gen_diversity = self._calculate_generation_diversity(population)
            nodes_info = " ".join([str(ind.nodes_count) for ind in population.population])
            fitness_info = " ".join([str(f) for f in population.fit])
            return base_info + [
                gen_diversity,
                np.std(population.fit),
                nodes_info,
                fitness_info,
                log_level
            ]
        
        return base_info + [log_level]

    def _log_evolution_to_csv(self, log_path, generation, timing, population, run_info):
        """
        Log evolution progress to a dedicated CSV file for analysis.
        
        Parameters
        ----------
        log_path : str
            Base path for logging files.
        generation : int
            Current generation number.
        timing : float
            Time taken for this generation.
        population : Population
            Current population.
        run_info : list
            Information about the current run.
        """
        if log_path is None:
            return
            
        # Create evolution log file path
        base_path = Path(log_path)
        evolution_csv_path = base_path.parent / f"{base_path.stem}_evolution.csv"
        
        # Prepare evolution data
        evolution_data = {
            'generation': generation,
            'seed': self.seed,
            'time_seconds': timing,
            'elite_fitness': float(self.elite.fitness),
            'elite_test_fitness': float(self.elite.test_fitness) if hasattr(self.elite, 'test_fitness') else None,
            'elite_nodes': self.elite.nodes_count,
            'population_size': len(population.population),
            'avg_population_fitness': np.mean(population.fit) if population.fit else None,
            'std_population_fitness': np.std(population.fit) if population.fit else None,
            'avg_nodes_count': float(population.nodes_count) / len(population.population) if population.population else None,
            'diversity': self._calculate_generation_diversity(population),
            'operator': self.operator
        }
        
        # # Add run info if available
        # if run_info:
        #     for i, info in enumerate(run_info):
        #         evolution_data[f'run_info_{i}'] = info
        
        # Write to CSV
        file_exists = evolution_csv_path.exists()
        
        with open(evolution_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = list(evolution_data.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header only if file is new
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(evolution_data)

    def _copy_individual(self, parent, reconstruct):
        """
        Create a copy of an individual.
        
        Parameters
        ----------
        parent : Individual
            The parent individual to copy.
        reconstruct : bool
            Whether to reconstruct the individual.
            
        Returns
        -------
        Individual
            A copy of the parent individual.
        """
        offspring = Individual(
            collection=parent.collection if reconstruct else None,
            train_semantics=parent.train_semantics,
            test_semantics=parent.test_semantics,
            reconstruct=reconstruct,
        )
        
        # Copy all relevant attributes
        offspring.nodes_collection = parent.nodes_collection
        offspring.nodes_count = parent.nodes_count
        offspring.depth_collection = parent.depth_collection
        offspring.depth = parent.depth
        offspring.size = parent.size
        
        return offspring

    def solve(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        curr_dataset,
        run_info,
        n_iter=20,
        elitism=True,
        log=0,
        verbose=0,
        test_elite=False,
        log_path=None,
        ffunction=None,
        max_depth=17,
        n_elites=1,
        reconstruct=True,
        n_jobs=1):
        """
        Solve the optimization problem using SLIM_GSGP.

        Parameters
        ----------
        X_train : array-like
            Training input data.
        X_test : array-like
            Testing input data.
        y_train : array-like
            Training output data.
        y_test : array-like
            Testing output data.
        curr_dataset : str or int
            Identifier for the current dataset.
        run_info : dict
            Information about the current run.
        n_iter : int
            Number of iterations. Default is 20.
        elitism : bool
            Whether elitism is used during evolution. Default is True.
        log : int or str
            Logging level (e.g., 0 for no logging, 1 for basic, etc.). Default is 0.
        verbose : int
            Verbosity level for logging outputs. Default is 0.
        test_elite : bool
            Whether elite individuals should be tested. Default is False.
        log_path : str
            File path for saving log outputs. Default is None.
        ffunction : function
            Fitness function used to evaluate individuals. Default is None.
        max_depth : int
            Maximum depth for the trees. Default is 17.
        n_elites : int
            Number of elite individuals to retain during selection. Default is True.
        reconstruct : bool
            Indicates if reconstruction of the solution is needed. Default is True.
        n_jobs : int
            Maximum number of concurrently running jobs for joblib parallelization. Default is 1.

        """

        if test_elite and (X_test is None or y_test is None):
            raise Exception('If test_elite is True you need to provide a test dataset')

        # setting the seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # starting time count
        start = time.time()

        # creating the initial population
        population = Population(
            [
                Individual(
                    collection=[
                        Tree(
                            tree,
                            train_semantics=None,
                            test_semantics=None,
                            reconstruct=True,
                        )
                    ],
                    train_semantics=None,
                    test_semantics=None,
                    reconstruct=True,
                )
                for tree in self.initializer(**self.pi_init)
            ]
        )

        # calculating initial population semantics
        population.calculate_semantics(X_train)

        # evaluating the initial population
        population.evaluate(ffunction, y=y_train, operator=self.operator, n_jobs=n_jobs)

        end = time.time()

        # setting up the elite(s)
        self.elites, self.elite = self.find_elit_func(population, n_elites)

        # calculating the testing semantics and the elite's testing fitness if test_elite is true
        if test_elite:
            population.calculate_semantics(X_test, testing=True)
            self.elite.evaluate(
                ffunction, y=y_test, testing=True, operator=self.operator
            )

        # Log initial population results
        if log != 0 and log_path is not None:
            # Log to evolution CSV
            self._log_evolution_to_csv(log_path, 0, end - start, population, run_info)

        # displaying the results on console if verbose level is more than 0
        if verbose != 0:
            verbose_reporter(
                curr_dataset,
                0,
                self.elite.fitness,
                self.elite.test_fitness,
                end - start,
                self.elite.nodes_count,
            )

        # begining the evolution process
        for it in range(1, n_iter + 1, 1):
            # starting an empty offspring population
            offs_pop, start = [], time.time()

            # adding the elite to the offspring population, if applicable
            if elitism:
                offs_pop.extend(self.elites)

            # filling the offspring population
            while len(offs_pop) < self.pop_size:

                # choosing between crossover and mutation

                if random.random() < self.p_xo:

                    p1, p2 = self.selector(population), self.selector(population)
                    while p1 == p2:
                        # choosing parents
                        p1, p2 = self.selector(population), self.selector(population)
                    pass  # future work on slim_gsgp implementations should invent crossover
                else:
                    # so, mutation was selected. Now deflation or inflation is selected.
                    if random.random() < self.p_deflate:

                        # selecting the parent to deflate
                        p1 = self.selector(population)

                        # Parent cannot be deflated, handle according to copy_parent setting
                        if p1.size == 1:
                            if self.copy_parent:
                                off1 = self._copy_individual(p1, reconstruct)
                            else:
                                # Inflate instead of deflate
                                ms_ = self.ms()
                                off1 = self.inflate_mutator(
                                    p1,
                                    ms_,
                                    X_train,
                                    max_depth=self.pi_init["init_depth"],
                                    p_c=self.pi_init["p_c"],
                                    X_test=X_test,
                                    reconstruct=reconstruct,
                                )
                        else:
                            # Normal deflation
                            off1 = self.deflate_mutator(p1, reconstruct=reconstruct)

                    # Inflation mutation was selected
                    else:
                        p1 = self.selector(population)
                        ms_ = self.ms()

                        # Check if parent can be inflated
                        if max_depth is not None and p1.depth == max_depth:
                            # Parent cannot be inflated
                            if self.copy_parent:
                                off1 = self._copy_individual(p1, reconstruct)
                            else:
                                off1 = self.deflate_mutator(p1, reconstruct=reconstruct)
                        else:
                            # Normal inflation
                            off1 = self.inflate_mutator(
                                p1,
                                ms_,
                                X_train,
                                max_depth=self.pi_init["init_depth"],
                                p_c=self.pi_init["p_c"],
                                X_test=X_test,
                                reconstruct=reconstruct,
                            )

                            # Check if offspring exceeds max depth after inflation
                            if max_depth is not None and off1.depth > max_depth:
                                if self.copy_parent:
                                    off1 = self._copy_individual(p1, reconstruct)
                                else:
                                    off1 = self.deflate_mutator(p1, reconstruct=reconstruct)

                    # adding the new offspring to the offspring population
                    offs_pop.append(off1)

            # removing any excess individuals from the offspring population
            if len(offs_pop) > population.size:

                offs_pop = offs_pop[: population.size]

            # turning the offspring population into a Population
            offs_pop = Population(offs_pop)
            # calculating the offspring population semantics
            offs_pop.calculate_semantics(X_train)

            # evaluating the offspring population
            offs_pop.evaluate(ffunction, y=y_train, operator=self.operator, n_jobs=n_jobs)

            # replacing the current population with the offspring population P = P'
            population = offs_pop
            self.population = population

            end = time.time()

            # setting the new elite(s)
            self.elites, self.elite = self.find_elit_func(population, n_elites)

            # calculating the testing semantics and the elite's testing fitness if test_elite is true
            if test_elite:
                self.elite.calculate_semantics(X_test, testing=True)
                self.elite.evaluate(
                    ffunction, y=y_test, testing=True, operator=self.operator
                )

            # Log generation results
            if log != 0 and log_path is not None:           
                # Log to evolution CSV
                self._log_evolution_to_csv(log_path, it, end - start, population, run_info)

            # displaying the results on console if verbose level is more than 0
            if verbose != 0:
                verbose_reporter(
                    run_info[-1],
                    it,
                    self.elite.fitness,
                    self.elite.test_fitness,
                    end - start,
                    self.elite.nodes_count,
                )
