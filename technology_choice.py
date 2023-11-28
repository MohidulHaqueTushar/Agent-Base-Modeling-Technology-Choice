# -*- coding: utf-8 -*-
"""
A technology choice model inspired by Brian Arthur in a population with a network structure 
following an Erdos-Renyi will be analyzed with different numbers of possible technologies to investigated 
if the number of possible technologies has an effect on the resulting market concentration.


Author: Md Mohidul Haque
"""

"""import modules"""
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

""" Agent class. Contains an intependent decision maker within a simulation"""
class Agent():
    def __init__(self, S, id_number, choice_function_exponent):
        """
        Constructor method.

        Parameters
        ----------
        S : Simulation object
            The simulation the agent belongs to.
        id_number : int
            Unique ID number of the agent.
        choice_function_exponent : numeric, optional
            Exponent of the Generalized Eggenberger-Polya process choice 
            function. Values >1 will lead to winner-take-all dynamics, Values 
            <1 lead to equalization dynamics. The default is 2.

        Returns
        -------
        None.

        """
        self.id_number = id_number
        self.Simulation = S
        self.technology = None
        self.choice_function_exponent = choice_function_exponent
        
    def choose(self):
        """
        Method for choosing a technology to adopt.

        Returns
        -------
        int or None
            Previous technology.
        int or None
            New technology.

        """
        """ Obtain distribution of technologies used by direct neighbors"""
        neighbors = self.get_neighbors()
        tech_list = self.Simulation.get_technologies_list()
        tech_frequency = {tech: 0 for tech in tech_list}
        for A in neighbors:
            tech = A.get_technology()
            if tech is not None:
                tech_frequency[tech] += 1

        """ Compute choice probabilities based on the distribution in the 
            immediate neighborhood. The form of the transformation may tend to 
            the technology used by the majority (if self.choice_function_exponent > 1)
            or overrepresent to those used by the minority (if 
            self.choice_function_exponent < 1)"""
        tech_probability = [tech_frequency[tech]**self.choice_function_exponent \
                                                    for tech in tech_list]
        if np.sum(tech_probability) > 0:
            """ Select and adopt a technology"""
            tech_probability = np.asarray(tech_probability) / np.sum(tech_probability)
            old_tech = self.technology
            self.technology = np.random.choice(tech_list, p=tech_probability)
            """ Report the change back"""
            return old_tech, self.technology
        else:
            """ Report that no change was possible"""
            return None, None
        
    def get_technology(self):
        """
        Getter method for the technology the agent uses.

        Returns
        -------
        int
            Current technology. The technologies are characterized as ints.

        """
        return self.technology
        
    def set_technology(self, tech):
        """
        Setter method for the technology the agent uses.

        Parameters
        ----------
        tech : int
            New technology the agent should adopt. The technologies are 
            characterized as ints.

        Returns
        -------
        None.

        """
        self.technology = tech
        
        
    def get_neighbors(self):
        """
        Method for returning a list of neighbor agents

        Returns
        -------
        List of Agent objects:
            List of Agents that are direct neighbors
        """
        return [self.Simulation.G.nodes[N]["agent"] for N in \
                        nx.neighbors(self.Simulation.G, self.id_number)]

""" Simulation class. Contains the entire run of one simulation for one 
    parameter setting.
"""
class Simulation():
    def __init__(self, 
                 n_agents=1000, 
                 n_technologies=3, 
                 n_initial_adopters=2, 
                 reconsideration_probability=0.2, 
                 choice_function_exponent=2, 
                 network_type="Erdos-Renyi",
                 t_max=5000):
        """
        Constructor method.

        Parameters
        ----------
        n_agents : int, optional
            Number of agents. The default is 1000.
        n_technologies : int, optional
            Number of technologies. The default is 3.
        n_initial_adopters : int, optional
            Number of initial adopters of each technology. The default is 2.
        reconsideration_probability : float, optional
            Probability for agents that have already chosen to reconsider their 
            choice when given the chance. The default is 0.2.
        choice_function_exponent : numeric, optional
            Exponent of the Generalized Eggenberger-Polya process choice 
            function. Values >1 will lead to winner-take-all dynamics, Values 
            <1 lead to equalization dynamics. The default is 2.
        network_type : str, optional
            Network type. Can be Erdos-Renyi, Barabasi-Albert, or Watts-Strogatz. 
            The default is "Erdos-Renyi".
        t_max : int, optional
            Number of time periods. The default is 5000.

        Returns
        -------
        None.

        """
        """ Define parameters"""
        self.n_agents = n_agents      
        self.t_max = t_max  
        self.n_technologies = n_technologies
        self.n_initial_adopters = n_initial_adopters        
        self.reconsideration_probability = reconsideration_probability  
        self.choice_function_exponent = choice_function_exponent
        
        """ Prepare technology list"""
        self.technologies_list = list(range(self.n_technologies))
        """ Prepare technology frequency dict. Each technology initialized with
            number zero."""
        self.tech_frequency = {tech: 0 for tech in self.technologies_list}
        
        """ Generate network"""
        if network_type == "Erdos-Renyi":
            self.G = nx.erdos_renyi_graph(n=self.n_agents, p=0.1)
        elif network_type == "Barabasi-Albert":
            self.G = nx.barabasi_albert_graph(n=self.n_agents, m=40)
        elif network_type == "Watts-Strogatz":
            self.G = nx.connected_watts_strogatz_graph(n=self.n_agents, k=40, p=0.15)
        else:
            assert False, "Unknown network type {:s}".format(network_type)
        
        """ Create agents and place them on the network"""
        self.agents_list = []
        
        for i in range(self.G.order()):
            A = Agent(self, i, self.choice_function_exponent)
            self.agents_list.append(A)
            self.G.nodes[i]["agent"] = A
        
        """ Seed technologies in random agents"""
        n_early_adopters = self.n_technologies*self.n_initial_adopters
        early_adopters = list(np.random.choice(self.agents_list, 
                                               replace=False, 
                                               size=n_early_adopters))
        for i in range(self.n_technologies):
            for j in range(self.n_initial_adopters):
                A = early_adopters.pop()
                A.set_technology(self.technologies_list[i])
            self.tech_frequency[i] += self.n_initial_adopters
        
        """ Prepare history variables and record initial values"""
        self.history_tech_frequency = {tech: \
                                [self.tech_frequency[tech] / self.n_agents] \
                                for tech in self.technologies_list}
        self.history_t = [0]

    def run(self):
        """
        Run method. Governs the course of the simulation.

        Returns
        -------
        None.

        """
        """ Time iteration"""
        for t in range(0, self.t_max + 1):
            """ Select one agent in each time step"""
            A = np.random.choice(self.agents_list)
            """ The agent will choose a technology if they have none, otherwise
                they may reconsider depending on self.reconsideration_probability"""
            tech = A.get_technology()
            if (tech is None) or \
                    (np.random.random() < self.reconsideration_probability):
                old, new = A.choose()
                if old is not None:
                    self.tech_frequency[old] -= 1
                if new is not None:
                    self.tech_frequency[new] += 1
            """ Record current state"""
            for i in range(self.n_technologies):
                self.history_tech_frequency[i].append( \
                                    self.tech_frequency[i] / self.n_agents)
            self.history_t.append(t)

    def get_technologies_list(self):
        """
        Getter method for technologies list

        Returns
        -------
        list of int
            List of technologies. Each technology is identified as an int.

        """
        return self.technologies_list

    def return_results(self, show_plot=False):
        """
        Method for returning and visualizing results
        
        Parameters
        ----------
        show_plot : bool, optional
            Should plots be shown. Default is False.

        Returns
        -------
        simulation_history : dict
            Recorded data on the simulation run.

        """

        """ Prepare return dict"""
        simulation_history = {"history_t": self.history_t,
                              "history_tech_frequency": self.history_tech_frequency}
        
        if show_plot:
            """ Create figure showing the development of usage shares of the 
                technologies"""
            fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False)
            for tech in self.history_tech_frequency.keys():
                ax[0][0].plot(self.history_t, 
                              self.history_tech_frequency[tech], 
                              label="Technology "+str(tech))
            ax[0][0].set_ylim(0, 1)
            ax[0][0].set_xlim(0, self.t_max+1)
            ax[0][0].set_ylabel("Frequqncy")
            ax[0][0].set_xlabel("Time")
            ax[0][0].legend()
            
            """ Save (as pdf) and show figure"""
            plt.tight_layout()
            plt.savefig("technology_choice_simulation.pdf")
            plt.show()
            
        return simulation_history


""" Class for series of verifiable simulations, returning reproducible 
    statistical results"""
class Series():
    def __init__(self, n_technologies=3, number_of_replications=100):
        """
        Constructor method.

        Parameters
        ----------
        n_technologies : int, optional
            Total number of available technologies. Can be any integer number. 
            The default value is 3.
        number_of_replications: int, optional
            Number of replications. The default is 100.

        Returns
        -------
        None.

        """
        self.n_technologies = n_technologies
        self.number_of_replications = number_of_replications
        self.list_nhhis = []
        
    def run(self):
        """
        Run method. Creates and runs the series of simulations and collects
        the results.
        To save time we can adjust the size of the simulations (lowering the 
        number of time periods and agents compared to the default). Note that 
        this changes the experiment and therefore also the results.

        Returns
        -------
        None.

        """
        
        for i in range(self.number_of_replications):
            """ Create and run one replication at a time."""
            """ To save time we can adjust the size of the simulations here
                (lowering the number of time periods and agents compared to 
                the default). Note that this changes the experiment and 
                therefore also the results."""
            S = Simulation(n_technologies=self.n_technologies, 
                           t_max=500,
                           n_agents=100)
            S.run()
            results = S.return_results()

            """ Extract market shares in the last period from result variable"""
            history_market_shares = results["history_tech_frequency"]
            market_shares = []
            for key in history_market_shares.keys():
                history = history_market_shares[key]
                ms = history[-1]
                market_shares.append(ms)
            
            """ Compute statistics (HHI)"""
            nhhi = hhi(market_shares)
            
            """ Collect results into class level lists"""
            self.list_nhhis.append(nhhi)
        
    def collect_results(self):
        """
        Method for returning result lists.

        Returns
        -------
        list of float
            List of HHI concentration measures at the end of the simulation.

        """
        return self.list_nhhis


""" Class for the experiment: Running and comparing two series of simulations"""
class Experiment():
    def __init__(self, n_technologies_of_first_series=5, n_technologies_of_second_series=10):
        """
        Constructor method. Generates and runs simulations with Erdos-Renyi network structures
        for two different number of available technologies.
        
        Parameters
        ----------
        n_technologies_of_first_series : int, optional
                Number of technologies for first series, default value 5.
    
        n_technologies_of_second_series : int, optional
                Number of technologies for second series, default value 10.


        Returns
        -------
        None.

        """
        self.n_technologies_of_first_series = n_technologies_of_first_series
        self.n_technologies_of_second_series = n_technologies_of_second_series
        self.FirstSeries = Series(n_technologies = self.n_technologies_of_first_series)
        self.FirstSeries.run()
        self.SecondSeries = Series(n_technologies = self.n_technologies_of_second_series)
        self.SecondSeries.run()
    
    def analyse_results(self, hist_on_hhi = True):
        """
        Method for computing, printing, and plotting statistics about the experiment
        
        Parameters
        ----------
        hist_on_hhi : bool, optional
                Default value True

        Returns
        -------
        None.

        """
        hhi_FS = self.FirstSeries.collect_results()
        hhi_SS = self.SecondSeries.collect_results()
        
        print("Simulation with Erdos-Renyi network where number of technologies are {}:"\
              .format(self.n_technologies_of_first_series))
        print("Hirschman-Herfindahl-Idx = ", np.mean(hhi_FS), "+/-", np.std(hhi_FS))
        
        print("\nSimulation with Erdos-Renyi network where number of technologies are {}:"\
              .format(self.n_technologies_of_second_series))
        print("Hirschman-Herfindahl-Idx = ", np.mean(hhi_SS), "+/-", np.std(hhi_SS))
        
        if hist_on_hhi:
            """
            ploting concentration measure
            """
            fig, ax = plt.subplots(nrows = 1, ncols = 2, squeeze = False)
            ax[0,0].hist(hhi_FS, bins = 15, rwidth = 0.85)
            ax[0,0].set_title("Hirschman-Herfindahl-Idx for two Series")
            ax[0,0].set_xlabel("First Series HHI")
            ax[0,0].set_ylabel("Frequency")
            ax[0,1].hist(hhi_SS, bins = 15, rwidth = 0.85)
            ax[0,1].set_xlabel("Second Series HHI")
            ax[0,1].set_ylabel("Frequency")
            plt.tight_layout()
            plt.savefig("Histogram of Hirschman-Herfindahl-Idx for two Series.pdf")
            plt.show()

""" Function for computing concentration measures HHI. 
    Could also have been implemented as methods of the Series class"""
        
def hhi(market_shares):
    """
    Function for computing the Hirschman-Herfindahl index of a set of 
    market shares.
        

    Parameters
    ----------
    market_shares : list-type of numeric
        Market shares.

    Returns
    -------
    nhhi : float
        Hirschman-Herfindahl Index.

    """
    """The HHI assumes market shares adding up to 100%. For small numbers of 
    time periods (i.e., if not all agents have made a choice yet), the 
    market shares here have to be renormalized as follows"""
    sum_market_shares = np.sum(market_shares)
    market_shares = [ms / sum_market_shares for ms in market_shares]
    
    n = len(market_shares)
    hhi = 0
    for ms in market_shares:
        hhi += ms**2
    nhhi = (hhi - 1 / n) / (1 - 1 / n)
    return nhhi


""" Main entry point"""    
if __name__ == '__main__':
    EX = Experiment()
    EX.analyse_results()

