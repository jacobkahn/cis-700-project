from generate import generate_general, separate_train_test
from model import DeepLearningClient, PerceptronClient, LocalClassifierClient
import numpy as np

outname = "noise_results.csv"
def noise(seq_length):
    constraint_list = [0, 10]
    training_sizes = range(500, 2100, 500)
    for cnum in constraint_list:
        for size in training_sizes:
            [inputs, outputs, constraints] = generate_general(seq_length, size, cnum, noise=True)
            [train, test] = separate_train_test(inputs, outputs)
            dl_acc = DeepLearningClient(train, test, seq_length).run()[1]
            lc_acc = LocalClassifierClient(train, test, seq_length).run()
            p_acc = PerceptronClient(train, test, seq_length).run(constraints)[1]
            outfile = open(outname, "a+")
            outfile.write(str(cnum)+','+str(size)+','+str(p_acc)+','+str(dl_acc)+','+str(lc_acc) + '\n')
            outfile.close()

soft_outname = "soft_results.csv"
def soft(seq_length):
    constraint_list = [10]
    training_sizes = range(500, 2100, 500)
    for cnum in constraint_list:
        for size in training_sizes:
            [inputs, outputs, constraints] = generate_general(seq_length, size, cnum, soft=True)
            [train, test] = separate_train_test(inputs, outputs)
            dl_acc = DeepLearningClient(train, test, seq_length).run()[1]
            lc_acc = LocalClassifierClient(train, test, seq_length).run()
            p_acc = PerceptronClient(train, test, seq_length).run(constraints)[1]
            outfile = open(soft_outname, "a+")
            outfile.write(str(cnum)+','+str(size)+','+str(p_acc)+','+str(dl_acc)+','+str(lc_acc) + '\n')
            outfile.close()
