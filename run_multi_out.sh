# test run
python multi_output.py --seed=13 --num_layers=2 --num_outputs=1 --lr=0.0001 --leap=10 --num_workers=4 --optimizer=Adam --num_iterations=1000

# leap 3 experiments on 2 layer net
python multi_output.py --seed=13 --num_layers=2 --num_outputs=1 --lr=0.0000001 --leap=3 --optimizer=Adam --num_workers=4 --num_iterations=1000000
python multi_output.py --seed=13 --num_layers=2 --num_outputs=1 --lr=0.00000002 --leap=3 --optimizer=Adam --num_workers=4 --num_iterations=1000000

# leap 2 experiments on 2 layer net
python multi_output.py --seed=13 --num_layers=2 --num_outputs=2 --lr=0.0000001 --leap=2 --optimizer=Adam --num_workers=4 --num_iterations=1000000
python multi_output.py --seed=13 --num_layers=2 --num_outputs=2 --lr=0.00000002 --leap=2 --optimizer=Adam --num_workers=4 --num_iterations=1000000
