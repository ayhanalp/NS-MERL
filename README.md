# NS-MERL
Novelty Seeking Multiagent Evolutionary Learning
Original MERL code can be found here: https://github.com/ShawK91/MERL.git

This repo implements Novelty Seeking MERL that was presented at the Genetic and Evolutionary Computation Conference 2023, Lisbon, Portugal.

The simple novelty seeking reward function can be found in rover_domain.py (state_only).

**Abstract:**
Coevolving teams of agents promises effective solutions for many coordination tasks such as search and rescue missions or deep ocean exploration. Good team performance in such domains generally relies on agents discovering complex joint policies, which is particularly difficult when the fitness functions are sparse (where many joint policies return the same or even zero fitness values). In this paper, we introduce Novelty Seeking Multiagent Evolutionary Reinforcement Learning (NS-MERL), which enables agents to more efficiently explore their joint strategy space. The key insight of NS-MERL is to promote good exploratory behaviors for individual agents using a dense, novelty-based fitness function. Though the overall team-level performance is still evaluated via a sparse fitness function, agents using NS-MERL more efficiently explore their joint action space and more readily discover good joint policies. Our results in complex coordination tasks show that teams of agents trained with NS-MERL perform significantly better than agents trained solely with task-specific fitnesses.

**BibTeX** to cite:
```
@inproceedings{aydeniz2023novelty,
  title={Novelty seeking multiagent evolutionary reinforcement learning},
  author={Aydeniz, Ayhan Alp and Loftin, Robert and Tumer, Kagan},
  booktitle={Proceedings of the Genetic and Evolutionary Computation Conference},
  pages={402--410},
  year={2023}
}
```
