# Input one of the 30 networks ./run_single.sh [1-30]
TEMP_PATH='temp'

EDGELIST_PATH='data/${$1}/edgelists.pkl'
NSWAP_PERC=101
FEATURE_PATH='data/${$1}/${$NSWAP_PERC}/'


mkdir -p $TEMP_PATH
git clone git@github.com:franktakes/teexgraph.git $TEMP_PATH
git --git-dir $TEMP_PATH checkout -b old-state 0c4ebef4ee938aa842bf40d1aec8a66d95fd8a82
(cd $TEMP_PATH && make listener)

python -m src.get_edgelist single $1 $EDGELIST_PATH
# python -m src.rewire single $1 $NSWAP_PERC
# python -m src.get_samples single $1 $NSWAP_PERC
# python -m src.get_features single $1 $FEATURE_PATH
# python -m src.get_performance single $1 $NSWAP_PERC
