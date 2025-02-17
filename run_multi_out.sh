# test run
# python multi_output.py --seed=13 --num_layers=2 --num_outputs=1 --lr=0.0001 --leap=10 --dimension=256 --num_workers=4 --optimizer=Adam --num_iterations=1000

# 2 layer experiments 

# # leap 3 experiments on 2 layer net
# python multi_output.py --seed=29 --num_layers=2 --num_outputs=1 --lr=0.0001 --leap=3 --optimizer=Adam --num_workers=4 --dimension=256 --num_iterations=100000
# python multi_output.py --seed=42 --num_layers=2 --num_outputs=1 --lr=0.0001 --leap=3 --optimizer=Adam --num_workers=4 --dimension=256 --num_iterations=100000
# python multi_output.py --seed=65 --num_layers=2 --num_outputs=1 --lr=0.0001 --leap=3 --optimizer=Adam --num_workers=4 --dimension=256 --num_iterations=100000


# # leap 2 experiments on 2 layer net
# python multi_output.py --seed=29 --num_layers=2 --num_outputs=2 --lr=0.0001 --leap=2 --optimizer=Adam --num_workers=4 --dimension=256 --num_iterations=100000
# python multi_output.py --seed=42 --num_layers=2 --num_outputs=2 --lr=0.0001 --leap=2 --optimizer=Adam --num_workers=4 --dimension=256 --num_iterations=100000
# python multi_output.py --seed=65 --num_layers=2 --num_outputs=2 --lr=0.0001 --leap=2 --optimizer=Adam --num_workers=4 --dimension=256 --num_iterations=100000

# #3 layer experiments
# # test run
# python multi_output.py --seed=13 --num_layers=3 --num_outputs=1 --lr=0.0001 --leap=10 --dimension=256 --num_workers=4 --optimizer=Adam --num_iterations=1000

# # leap 3 experiments on 2 layer net
# python multi_output.py --seed=13 --num_layers=3 --num_outputs=1 --lr=0.0001 --leap=3 --optimizer=Adam --num_workers=4 --dimension=256 --num_iterations=100000
# python multi_output.py --seed=35 --num_layers=3 --num_outputs=1 --lr=0.0001 --leap=3 --optimizer=Adam --num_workers=4 --dimension=256 --num_iterations=100000
# python multi_output.py --seed=43 --num_layers=3 --num_outputs=1 --lr=0.0001 --leap=3 --optimizer=Adam --num_workers=4 --dimension=256 --num_iterations=100000
# python multi_output.py --seed=65 --num_layers=3 --num_outputs=1 --lr=0.0001 --leap=3 --optimizer=Adam --num_workers=4 --dimension=256 --num_iterations=100000

# # python multi_output.py --seed=13 --num_layers=3 --num_outputs=1 --lr=0.00001 --leap=3 --optimizer=Adam --num_workers=4 --dimension=256 --num_iterations=100000


# # leap 2 experiments on 2 layer net
# python multi_output.py --seed=13 --num_layers=3 --num_outputs=2 --lr=0.0001 --leap=2 --optimizer=Adam --num_workers=4 --dimension=256 --num_iterations=100000
# python multi_output.py --seed=35 --num_layers=3 --num_outputs=2 --lr=0.0001 --leap=2 --optimizer=Adam --num_workers=4 --dimension=256 --num_iterations=100000
# python multi_output.py --seed=43 --num_layers=3 --num_outputs=2 --lr=0.0001 --leap=2 --optimizer=Adam --num_workers=4 --dimension=256 --num_iterations=100000
# python multi_output.py --seed=65 --num_layers=3 --num_outputs=2 --lr=0.0001 --leap=2 --optimizer=Adam --num_workers=4 --dimension=256 --num_iterations=100000

# python multi_output.py --seed=13 --num_layers=3 --num_outputs=2 --lr=0.00001 --leap=2 --optimizer=Adam --num_workers=4 --dimension=256 --num_iterations=100000


# # 2 -layer scaling law
# python multi_output.py --seed=29 --num_layers=2 --num_outputs=2 --lr=0.0004 --leap=2 --optimizer=Adam --num_workers=4 --dimension=128 --num_iterations=25000
# python multi_output.py --seed=42 --num_layers=2 --num_outputs=2 --lr=0.0004 --leap=2 --optimizer=Adam --num_workers=4 --dimension=128 --num_iterations=25000
# python multi_output.py --seed=65 --num_layers=2 --num_outputs=2 --lr=0.0004 --leap=2 --optimizer=Adam --num_workers=4 --dimension=128 --num_iterations=25000

# python multi_output.py --seed=29 --num_layers=2 --num_outputs=2 --lr=0.0016 --leap=2 --optimizer=Adam --num_workers=4 --dimension=64 --num_iterations=10000
# python multi_output.py --seed=42 --num_layers=2 --num_outputs=2 --lr=0.0016 --leap=2 --optimizer=Adam --num_workers=4 --dimension=64 --num_iterations=10000
# python multi_output.py --seed=65 --num_layers=2 --num_outputs=2 --lr=0.0016 --leap=2 --optimizer=Adam --num_workers=4 --dimension=64 --num_iterations=10000

# python multi_output.py --seed=29 --num_layers=2 --num_outputs=2 --lr=0.000025 --leap=2 --optimizer=Adam --num_workers=4 --dimension=512 --num_iterations=1000000
# python multi_output.py --seed=42 --num_layers=2 --num_outputs=2 --lr=0.000025 --leap=2 --optimizer=Adam --num_workers=4 --dimension=512 --num_iterations=1000000
# python multi_output.py --seed=65 --num_layers=2 --num_outputs=2 --lr=0.000025 --leap=2 --optimizer=Adam --num_workers=4 --dimension=512 --num_iterations=1000000

# 2-layer step-size tuning

# python multi_output.py --seed=29 --num_layers=2 --num_outputs=2 --lr=0.0025 --leap=2 --optimizer=Adam --num_workers=4 --dimension=64 --num_iterations=2000
# python multi_output.py --seed=42 --num_layers=2 --num_outputs=2 --lr=0.0025 --leap=2 --optimizer=Adam --num_workers=4 --dimension=64 --num_iterations=2000
# python multi_output.py --seed=65 --num_layers=2 --num_outputs=2 --lr=0.0025 --leap=2 --optimizer=Adam --num_workers=4 --dimension=64 --num_iterations=2000

# python multi_output.py --seed=29 --num_layers=2 --num_outputs=2 --lr=0.005 --leap=2 --optimizer=Adam --num_workers=4 --dimension=64 --num_iterations=2000
# python multi_output.py --seed=42 --num_layers=2 --num_outputs=2 --lr=0.005 --leap=2 --optimizer=Adam --num_workers=4 --dimension=64 --num_iterations=2000
# python multi_output.py --seed=65 --num_layers=2 --num_outputs=2 --lr=0.005 --leap=2 --optimizer=Adam --num_workers=4 --dimension=64 --num_iterations=2000

# python multi_output.py --seed=29 --num_layers=2 --num_outputs=2 --lr=0.01 --leap=2 --optimizer=Adam --num_workers=4 --dimension=64 --num_iterations=2000
# python multi_output.py --seed=42 --num_layers=2 --num_outputs=2 --lr=0.01 --leap=2 --optimizer=Adam --num_workers=4 --dimension=64 --num_iterations=2000
# python multi_output.py --seed=65 --num_layers=2 --num_outputs=2 --lr=0.01 --leap=2 --optimizer=Adam --num_workers=4 --dimension=64 --num_iterations=2000

# python multi_output.py --seed=29 --num_layers=2 --num_outputs=2 --lr=0.02 --leap=2 --optimizer=Adam --num_workers=4 --dimension=64 --num_iterations=2000
# python multi_output.py --seed=42 --num_layers=2 --num_outputs=2 --lr=0.02 --leap=2 --optimizer=Adam --num_workers=4 --dimension=64 --num_iterations=2000
# python multi_output.py --seed=65 --num_layers=2 --num_outputs=2 --lr=0.02 --leap=2 --optimizer=Adam --num_workers=4 --dimension=64 --num_iterations=2000

python multi_output.py --seed=29 --num_layers=2 --num_outputs=2 --lr=0.05 --leap=2 --optimizer=Adam --num_workers=4 --dimension=64 --num_iterations=2000
python multi_output.py --seed=42 --num_layers=2 --num_outputs=2 --lr=0.05 --leap=2 --optimizer=Adam --num_workers=4 --dimension=64 --num_iterations=2000
python multi_output.py --seed=65 --num_layers=2 --num_outputs=2 --lr=0.05 --leap=2 --optimizer=Adam --num_workers=4 --dimension=64 --num_iterations=2000

python multi_output.py --seed=29 --num_layers=2 --num_outputs=2 --lr=0.1 --leap=2 --optimizer=Adam --num_workers=4 --dimension=64 --num_iterations=2000
python multi_output.py --seed=42 --num_layers=2 --num_outputs=2 --lr=0.1 --leap=2 --optimizer=Adam --num_workers=4 --dimension=64 --num_iterations=2000
python multi_output.py --seed=65 --num_layers=2 --num_outputs=2 --lr=0.1 --leap=2 --optimizer=Adam --num_workers=4 --dimension=64 --num_iterations=2000
