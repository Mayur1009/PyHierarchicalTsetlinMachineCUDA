from PyHierarchicalTsetlinMachineCUDA.tm import TsetlinMachine
import numpy as np
from time import time
import PyHierarchicalTsetlinMachineCUDA.tm as tm
import argparse

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--runs", default=10, type=int)
    parser.add_argument("--number-of-clauses", default=2000, type=int)
    parser.add_argument("--T", default=9000, type=int)
    parser.add_argument("--s", default=44.0, type=float)
    parser.add_argument("--q", default=1.0, type=float)
    parser.add_argument("--boost", default=1, type=int)
    parser.add_argument("--number-of-state-bits", default=7, type=int)
    parser.add_argument("--number-of-alternatives", default=60, type=int)
    parser.add_argument("--constant-update-p", action='store_true')
    parser.add_argument('--binary-inference', action='store_true')
    parser.add_argument('--vanilla', action='store_true')
    parser.add_argument('--and-group-normalization', action='store_true')
    parser.add_argument('--no-clipping', action='store_true')

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

args = default_args()

data = np.loadtxt("./examples/hex_data.txt").astype(np.uint32)
X_train = data[:int(len(data)*0.8),0:-1]
Y_train = data[:int(len(data)*0.8),-1]

X_test = data[int(len(data)*0.8):,0:-1]
Y_test = data[int(len(data)*0.8):,-1]

f = open("hex_statistics_%d_%d_%.2f_%d_%d_%d_%d_%d_%d_%d.txt" % (args.number_of_clauses, args.T, args.s, args.number_of_state_bits, args.vanilla, args.and_group_normalization, args.constant_update_p, args.binary_inference, args.number_of_alternatives, args.no_clipping), "w")

for r in range(args.runs):
    seed = np.random.randint(10000)

    if args.vanilla:
        tsetlin_machine = TsetlinMachine(
            args.number_of_clauses * args.number_of_alternatives,
            args.T,
            args.s,
            weighted_clauses=False,
            number_of_state_bits=args.number_of_state_bits,
            boost_true_positive_feedback=args.boost,
            binary_inference=args.binary_inference,
            constant_update_p=args.constant_update_p,
            and_group_normalization=args.and_group_normalization,
            seed=seed,
            no_clipping=args.no_clipping,
            hierarchy_structure=(
                (tm.AND_GROUP, 288),
                (tm.AND_GROUP, 1)
            )
        )
    else:
        tsetlin_machine = TsetlinMachine(
            args.number_of_clauses,
            args.T,
            args.s,
            weighted_clauses=False,
            number_of_state_bits=args.number_of_state_bits,
            boost_true_positive_feedback=args.boost,
            binary_inference=args.binary_inference,
            constant_update_p=args.constant_update_p,
            and_group_normalization=args.and_group_normalization,
            seed=seed,
            no_clipping=args.no_clipping,
            hierarchy_structure=(
                (tm.AND_GROUP, 72),
                (tm.OR_ALTERNATIVES, args.number_of_alternatives),
                (tm.AND_GROUP, 4)
            )
        )

    print("\nAccuracy over %d epochs:\n" % (args.epochs))
    for e in range(args.epochs):
        start_training = time()
        for b in range(10):
            tsetlin_machine.fit(X_train[b*len(Y_train)//10:(b+1)*len(Y_train)//10], Y_train[b*len(Y_train)//10:(b+1)*len(Y_train)//10])
        stop_training = time()

        start_testing = time()
        result_testing = 100*(tsetlin_machine.predict(X_test) == Y_test).mean()
        stop_testing = time()

        #result_training = 100*(tsetlin_machine.predict(X_train) == Y_train).mean()

        print("#%d/%d Testing Accuracy: %.2f%% Training Time: %.2fs Testing Time: %.2fs" % (r+1, e+1, result_testing, stop_training-start_training, stop_testing-start_testing))
        f.write("%d %d %.2f\n" % (r, e, result_testing))
        f.flush()

f.close()