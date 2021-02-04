import pandas as pd
from itertools import chain
import sys
from io import StringIO
import numpy as np
from Performance_measures import confusion_metrics_basic, Micro_calculate_measures, Macro_calculate_measures_basic

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
N_CLASSSES = 4
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def evaluate_rules_boundary(rules, x_test, y_test_one):
    # Flatten the rule_set list into a 1D list
    # print(rules)
    y_test = [np.where(r == 1)[0][0] for r in y_test_one]
    y_pred = np.zeros(len(y_test), dtype=int)
    # rules = list(chain.from_iterable(rules))
    uncovered_sample = 0
    t = 0
    for x in x_test:
        flg = False
        # sum_one = 0
        # sum_zero = 0
        sum = np.zeros(N_CLASSSES)
        # flg_rule = False
        for rl in rules:
            # Write each rule into a string then execute it
            # f = open('rule_code.py', 'w')
            str_rule = 'if ('
            # cnt_non_match = 0
            rl_lbl = rl[-1]
            rl = rl[:-1]
            i = 3
            while i < (len(rl) - 1):
                try:
                    rl[i] = str(x[int(rl[i])])
                    i += 6
                except Exception as E:
                    print(E)
            try:
                # Generate rule string to be written to a file
                str_rule += ' '.join(rl) + '):\n\t' + 'print(True)\nelse:\n\tprint(False)'
                # f.write(str_rule)
                # f.close()
                '''Save stdout into a file'''
                # Store the reference, in case you want to show things again in standard output
                old_stdout = sys.stdout
                # This variable will store everything that is sent to the standard output
                result = StringIO()
                sys.stdout = result
                # Here we execute the rule instructions, and everything that they will send to standard output will be stored on "result"
                # exec(open("rule_code.py").read())
                # Redirect again the std output to screen
                exec(str_rule)
                sys.stdout = old_stdout
                # Then, get the stdout like a string and process it!

                flg_rule = result.getvalue()
                flg_rule = flg_rule.replace('\n', '')

            except Exception as E:
                print(E)
            # If the rule covers the whole sample
            if (flg_rule == 'True'):
                flg = True
                # 1 0 sample
                sum[int(rl_lbl)] += 1
                # if (rl_lbl == 0):
                #     sum_zero += 1
                # # 0 1 sample
                # else:
                #     sum_one += 1

        # if the current sample is not covered by any of the rules
        if (flg == False):
            uncovered_sample += 1
            continue
        # if the current sample is covered by at least one of the rules
        else:
            rule_vote = np.argmax(sum)

        ## In the rules, output 0 means (1 0) Malicious
        # In the rules, output 1 means (0 1) Benign
        y_pred[t] = rule_vote
        t += 1
        # if (t==10):
        #     break
    mcm, tp_mean, tn_mean, fp_mean, fn_mean = confusion_metrics_basic(y_test, y_pred)
    out_measures = Micro_calculate_measures(tp_mean, tn_mean, fp_mean, fn_mean, uncovered_sample)
    pr, rc, f1 = Macro_calculate_measures_basic(y_test, y_pred)

    np.savetxt("Experiments/Micro_Test_Conf_CapsRule.csv", mcm, delimiter=',', fmt='%s')
    np.savetxt("Experiments/Micro_Test_Measures_CapsRules.csv", out_measures.to_numpy(), delimiter=',', fmt='%s')
    f = open("Experiments/Macro_Test_Results_CapsRules.txt", 'w')
    f_str = "PR:" + str(pr) + " RC:" + str(rc) + " F1:" + str(f1)
    f.write(f_str)
    f.close()
    np.savetxt("Experiments/y_true_CapsRule_Test.csv", y_test, delimiter=',')
    np.savetxt("Experiments/y_pre_CapsRule_Test.csv", y_pred, delimiter=',')