This repository provides the code used in a manuscript titled ['Supervised temporal link prediction in large-scale real-world networks'](https://github.com/gerritjandebruin/SNAM2021-paper).

Temporal networks used in this study are downloaded from http://konect.cc/ or http://snap.stanford.edu.

The structure of the directories is as follows:
- code: Contains all code that is not part of tlp package.
- data: Folder where all downloads and intermediate results are stored.
- teexgraph: External dependency to calculate the diameters and shortest path lengths in a network really fast. See .gitmodules.
- tlp: Here most of my code resides. I import this folder as a package in most Jupyter Notebooks.
- cleanup.sh: Clean up Python/ iPython cache folders.
- environment.yml: Can be used to create Python environment with Conda.
- install.sh: Install teexGraph.
- spec-list.txt: Can be used to create exactly the same Python environment.


## TODO for genetic programming:
- [ ] Alter get_features_gp to save results before applying time_strategies
- [ ] Create a loop in run_single.py after get_features_gp that tries different time_strategies
    - [ ] Apply GP time_stategy to get_features_gp result (Singular!)
    - [ ] Evaluate performance of function using get_performance.predict
    - [ ] Keep track of fittest functions per dataset
