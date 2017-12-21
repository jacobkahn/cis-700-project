from generate import generate_general, separate_train_test
from model import DeepLearningClient, PerceptronClient, LocalClassifierClient
import numpy as np
import unicodecsv as csv
import model
import multiprocessing

def run_noisy_sim(seq_length, size, cnum, noise, shared_data):
    res = model.run(seq_length, size, num_constraints=cnum, noise=noise)
    shared_data.append(res)

def run_soft_sim(seq_length, size, cnum, soft, shared_data):
    res = model.run(seq_length, size, num_constraints=cnum, soft=soft)
    shared_data.append(res)

def run_neutral_sim(seq_length, size, cnum, shared_data):
    res = model.run(seq_length, size, num_constraints=cnum)
    shared_data.append(res)

NOISE_OUTNAME = "noise_results.csv"
def noise(_):
    # shared data structure for all results
    manager = multiprocessing.Manager()
    shared_results_list = manager.list()
    jobs = []

    seq_lengths = [10]
    constraint_list = [0, 5, 10, 20]
    training_sizes = range(100, 2001, 100)
    for seq_length in seq_lengths:
        for idx1, cnum in enumerate(constraint_list):
            for idx2, size in enumerate(training_sizes):
                # [inputs, outputs, constraints] = generate_general(seq_length, size, cnum, noise=True)
                # [train, test] = separate_train_test(inputs, outputs)
                # dl_acc = DeepLearningClient(train, test, seq_length).run()[1]
                # lc_acc = LocalClassifierClient(train, test, seq_length).run()
                # p_acc = PerceptronClient(train, test, seq_length).run(constraints)[1]
                # Run all selected simulations in model
                print("Starting noisy process with parameters:")
                print(seq_length, size, cnum, True, shared_results_list)
                p = multiprocessing.Process(target=run_noisy_sim, args=(seq_length, size, cnum, True, shared_results_list))
                jobs.append(p)
                p.start()
                # res = model.run(seq_length, size, num_constraints=cnum, noise=True)
                # print "res is "
                # print res

    # wait on all jobs
    for job in jobs:
        job.join()

    # write results to file
    print("All noisy results:")
    print(shared_results_list)

    with open(NOISE_OUTNAME, "a+") as outfile:
        writer = csv.DictWriter(outfile, shared_results_list[0].keys())
        writer.writeheader()
        for result in shared_results_list:
            writer.writerow({k: str(v) for k,v in result.items()})



SOFT_OUTNAME = "soft_results.csv"
def soft(_):
    # shared data structure for all results
    manager = multiprocessing.Manager()
    shared_results_list = manager.list()
    jobs = []

    seq_lengths = [10]
    constraint_list = [5, 10, 20]
    training_sizes = range(100, 2001, 100)
    for seq_length in seq_lengths:
        for idx1, cnum in enumerate(constraint_list):
            for idx2, size in enumerate(training_sizes):
                # [inputs, outputs, constraints] = generate_general(seq_length, size, cnum, soft=True)
                # [train, test] = separate_train_test(inputs, outputs)
                # dl_acc = DeepLearningClient(train, test, seq_length).run()[1]
                # lc_acc = LocalClassifierClient(train, test, seq_length).run()
                # p_acc = PerceptronClient(train, test, seq_length).run(constraints)[1]
                # outfile = open(SOFT_OUTNAME, "a+")
                # outfile.write(str(cnum)+','+str(size)+','+str(p_acc)+','+str(dl_acc)+','+str(lc_acc) + '\n')
                # outfile.close()
                # Run all selected simulations in model
                print("Starting soft process with parameters:")
                print(seq_length, size, cnum, True, shared_results_list)
                p = multiprocessing.Process(target=run_soft_sim, args=(seq_length, size, cnum, True, shared_results_list))
                jobs.append(p)
                p.start()
                # res = model.run(seq_length, size, num_constraints=cnum, soft=True)
                # print "res is "
                # print res

    # wait on all jobs
    for job in jobs:
        job.join()

    # write results to file
    print("All soft results:")
    print(shared_results_list)

    with open(SOFT_OUTNAME, "a+") as outfile:
        writer = csv.DictWriter(outfile, shared_results_list[0].keys())
        writer.writeheader()
        for result in shared_results_list:
            writer.writerow({k: str(v) for k,v in result.items()})


NEUTRAL_OUTNAME = "neutral_results.csv"
def neutral(_):
    print('at beginning')
    # shared data structure for all results
    manager = multiprocessing.Manager()
    shared_results_list = manager.list()
    jobs = []

    seq_lengths = [10]
    constraint_list = [0, 5, 10, 20]
    training_sizes = range(100, 2001, 100)
    print('before loop')
    constraint_jobs = []
    for seq_length in seq_lengths:
        for idx2, size in enumerate(training_sizes):
            for idx1, cnum in enumerate(constraint_list):
                # Run all selected simulations in model
                print("Starting neutral process with parameters:")
                print(seq_length, size, cnum, shared_results_list)
                p = multiprocessing.Process(target=run_neutral_sim, args=(seq_length, size, cnum, shared_results_list))
                # jobs.append(p)
                constraint_jobs.append(p)
                p.start()
            if idx2 % 2 == 1:
                for job in constraint_jobs:
                    job.join()
                constraint_jobs = []

    # wait on all jobs
    for job in jobs:
        job.join()

    # write results to file
    print("All neutral results:")
    print(shared_results_list)

    with open(NEUTRAL_OUTNAME, "a+") as outfile:
        writer = csv.DictWriter(outfile, shared_results_list[0].keys())
        writer.writeheader()
        for result in shared_results_list:
            writer.writerow({k: str(v) for k,v in result.items()})




if __name__ == "__main__":
    # 10, 20
    neutral_proc = multiprocessing.Process(target=neutral, args=(0,))
    # noise_proc = multiprocessing.Process(target=noise, args=(0,))
    # soft_proc = multiprocessing.Process(target=soft, args=(0,))

    # Run sequentially so we don't break the machine
    neutral_proc.start()
    neutral_proc.join()

    # noise_proc.start()
    # noise_proc.join()

    # soft_proc.start()
    # soft_proc.join()
