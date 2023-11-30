# Copied from: run.sh
TEMP_PATH='temp'

mkdir -p $TEMP_PATH
git clone git@github.com:franktakes/teexgraph.git $TEMP_PATH
git --git-dir $TEMP_PATH checkout -b old-state 0c4ebef4ee938aa842bf40d1aec8a66d95fd8a82
(cd $TEMP_PATH && make listener)


# Adjusted for single run
##
# General settings
#
# Set the index_network variable to the desired network index
INDEX_NETWORK=1
NSWAP_PERC = -100
# Set the path to the edgelist.pkl file for the specified network
EDGE_LIST_PATH="data/${INDEX_NETWORK}/edgelist.pkl"


# Run the get_performance command for the specified network
python -m src.get_performance single $INDEX_NETWORK $EDGE_LIST_PATH